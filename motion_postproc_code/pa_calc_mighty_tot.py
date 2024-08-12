#!/usr/local/bin/python3

# @file pa_calc_mighty_tot_v03.py
#
# @brief Calculate physical activity intensity for infants.
#
# Calculate based on a single ankle-worn accelerometer. Infants should be
# pre-ambulatory) not yet walking.
#
# Need time (s) and 3D accelerometer data (m/s2) sampled  at 20 Hz. Code can be
# modified for higher sampling rates.the approach should work the same, i.e.
# intergrating over 2 seconds, regardless of sampling rate.
#
# Based on the following paper:
#
# Ghazi MA, Zhou J, Havens KL, Smith BA. Accelerometer Thresholds for Estimating
# Physical Activity Intensity Levels in Infants: A Preliminary Study. Sensors.
# 2024; 24(14):4436
#
# Based on pa_calc_mighty_tot_v03 which in turn is based on mighty_tot_calc_v01
#
# Computation steps:
#
# -Calculate magnitude
#
# -Calculate baseline, a_G
# ****************  TRY DO DO ALL OF THESE IN ONE LOOP? *****************
# -Calculate gravity independent magnitude of acceleration, a_IND,
# -Scale the acceleration by leg length, a_ADJ
# -Calculate computed quantity (jerk or accel), left
# -Assign label based on threshold
# ****************  TRY DO DO ALL OF THESE IN ONE LOOP? *****************
# -Compile statistics (num points in each category, percentage activity)
#
# Known bugs:
# -Last line of raw output file may have just zeros
#
# TODO: limit how far to go for baseline?
#
# Created 04 Apr 2024
# Modified 03 May 2024 (V02)
# -Fixed bug in time calc
# -Conversion to and from date/time int versus separate
# -Start/end time
# -Start/end time checks
# -Calculate and save duration of activity regions
# -File write summary
# -File write all results
# -File write error log
# -Clean up parameters for all file writes
# -Update to use right threshold only
# -Separate functions for load/view and process
# Modified 20 Jul 2024 (V03)
# -Added load from .tsv
# -Added full results write (raw)
# -Removed unused commented code
# -TODO: merged perocessing messaged, error mesages
#
# @author Mustafa Ghazi
# @editor Jinseok Oh

import time
from datetime import datetime, timezone
import json
import numpy as np
# import pandas as pd
# import pytz
import h5py
import pyarrow
from pyarrow import csv

# ****************************************************************************
# Global variables, user input, CAN EDIT
# ****************************************************************************
# File I/O options
# fileNameNoExtension = "PA001_6020352_L_CALIBRATED"  # same 3 day above
# PA calculation parameters
# computedQttyOption = "acceleration"  # acceleration or jerk
optimizationTypeOption = "PAP"  # TP or PAP
# infantLegLengthCm = 26.5
baselineLegLengthCm = 26.5
# Start processing from this time onwards
# start time, epoch time UTC time zone
# userInputStartYear = 2024
# userInputStartMonth = 3
# userInputStartDay = 8 # should be 8
# userInputStartHour = 11 # should be 10
# userInputStartMinute = 15
# userInputStartSecond = 2
# Do not process beyond this time
# end time, epoch time UTC time zone
# userInputEndYear = 2024
# userInputEndMonth = 3
# userInputEndDay = 8 # should be 11
# userInputEndHour = 12 # should be 10
# userInputEndMinute = 15
# userInputEndSecond = 1
# End global variables, user input, CAN EDIT
# ****************************************************************************


# ****************************************************************************
#  Global constants, DO NOT EDIT
# ****************************************************************************
# Thresholds, based in single leg
# jerk, TP optimization
constThreshLowerJerkTP = 11.0
constThreshUpperJerkTP = 17.0
# jerk, PAP optimization
constThreshLowerJerkPAP = 7.0
constThreshUpperJerkPAP = 24.0
# acceleration, TP optimization
constThreshLowerAccelTP = 0.4
constThreshUpperAccelTP = 0.8
# acceleration, PAP optimization
constThreshLowerAccelPAP = 0.3
constThreshUpperAccelPAP = 1.0
constRefLegLength3To5MonthCm = 26.5  # Compiled from Snyder et al.
# End global constants,  DO NOT EDIT
# ****************************************************************************

# ****************************************************************************
# Global variables to use, DO NOT EDIT
# ****************************************************************************
# File I/O options
fileNameAppendRaw = "RAW"  # time, type, 0.05 s interval
fileNameAppendBouts = "BOUTS"  # start time, end time, duration, type
fileNameAppendSummary = "SUMMARY"  # Time, pct stats
fileNameAppendErrorLog = "LOG"  # all errors and messages produced by code
inputFileExtension = ".tsv"  # [JO] For HBCD
outputFileExtension = ".txt"
inputDir = "data"
outputDir = "output"
# Start/end time options
fileReadErrorFlag = 0  # 0 if no error, 1 if error
dataInputErrorFlag = 0  # 0 if no error, 1 if error
printBufferStr = "********************\nMIGHTY TOT OUTPUT LOG\n********************"
# End global variables to use, DO NOT EDIT
# ****************************************************************************


# @brief Add a string to the print buffer
#
# @param thisString String to print
def printToPrintBuffer(thisString):
    global printBufferStr
    printBufferStr = printBufferStr + '\n' + thisString
    displayPrintBuffer()


# @brief Display the full print buffer to the display area
def displayPrintBuffer():
    global printBufferStr
    print(printBufferStr)


def getDatetimeNowString():
    myDateTimeNow = datetime.now()
    # myOutStr = myDateTimeNow.year + '_' + myDateTimeNow.month + '_' + myDateTimeNow.day + '_' + myDateTimeNow.hour + '_' + myDateTimeNow.minute + '_' + myDateTimeNow.second
    myOutStr = "%04d-%02d-%02d %02d:%02d:%02d" % (myDateTimeNow.year, myDateTimeNow.month, myDateTimeNow.day, myDateTimeNow.hour, myDateTimeNow.minute, myDateTimeNow.second)
    return myOutStr


def dateTimeEpochToRegularDateTime(dateTimeEpochInt):
    myDateTimeNow = datetime.utcfromtimestamp(dateTimeEpochInt)
    myOutStr = "%04d-%02d-%02d %02d:%02d:%02d" % (myDateTimeNow.year, myDateTimeNow.month, myDateTimeNow.day, myDateTimeNow.hour, myDateTimeNow.minute, myDateTimeNow.second)
    return myOutStr

# @brief A physical activity series is a series that holds information about recorded
# movement data, computed quantity, and its classification
#
# index, time, computed qty total, computed qtty left, computed qtty right, accel mag left, accel mag right
#
class PhysActSeries:
    
    def __init__(self, numDataPts):
        self.numDataPts = numDataPts
        self.index = np.zeros((numDataPts,1), dtype=int) # Nx1 array of index based on source data, int (no units)
        self.time = np.zeros((numDataPts,1)) # Nx1 array of UTC time float (seconds)
        self.classification = 999.0*np.ones((numDataPts,1)) # Nx1 array of classification, 0.0, 3.0, 6.0 (light, sedentary, moderate-to-vigorous) float (no units)
        self.computedQttyTotal = np.zeros((numDataPts,1)) # Nx1 array of computed accel qtty, float (computed qtty units)
        self.computedQttyLeft = np.zeros((numDataPts,1)) # Nx1 array of computed accel qtty, float (computed qtty units)
        self.computedQttyRight = np.zeros((numDataPts,1)) # Nx1 array of computed accel qtty, float (computed qtty units)
        self.accelMagLeft = np.zeros((numDataPts,1)) # Nx1 array of acceleration magnitude left leg, float (m/s2)
        self.accelMagRight = np.zeros((numDataPts,1)) # Nx1 array of acceleration magnitude right leg, float (m/s2)
        
 
# @brief An activity region is a series that holds information on constant
# activity periods, e.g. [1-2.5 seconds, light], [2.5-6 seconds, sedentaty],
# [6.0-7.5 seonds, MV], and so on
#
# start time,  end time, start index, end index, classification
#
class ActivityRegions:
 
    def __init__(self, numDataPts):
        self.numDataPts = numDataPts
        self.idxStart = np.zeros((numDataPts,1), dtype=int) # Nx1 array of start index based on source data, int (no units)
        self.idxEnd = np.zeros((numDataPts,1), dtype=int) # Nx1 array of end index based on source data, int (no units)
        self.timeStart = np.zeros((numDataPts,1)) # Nx1 array of start time based on source data, UTC time float (seconds)
        self.timeEnd = np.zeros((numDataPts,1)) # Nx1 array of end time based on source data, UTC time float (seconds)
        self.classification = 999.0*np.ones((numDataPts,1)) # Nx1 array of classification, 0.0, 3.0, 6.0 (light, sedentary, moderate-to-vigorous) float (no units)
 
    def printString(self, thisClass, thisStartIdx, thisEndIdx, thisStartTime, thisEndTime):
        printToPrintBuffer(str(thisClass) + ": " + str(thisStartIdx) + "-" + str(thisEndIdx) + "\t" + str(thisStartTime)  + "-" + str(thisEndTime))


# @brief Load the sensor data into the PhysActSeries object
#
# @return Physical activity series object with index, time, and accel magnitude populated
#
def loadPhysActSeries(timeArr, accelMag):

    numDataPtsHere = len(timeArr)
    myAccelObj = PhysActSeries(numDataPtsHere)
    i = 0
    while(i < numDataPtsHere):
        myAccelObj.index[i,0] = i
        myAccelObj.time[i,0] = timeArr[i,0]
        myAccelObj.accelMagLeft[i,0] = accelMag[i,0] # always fill in the left one when using single accelerometer data

        i = i + 1

    return myAccelObj


# @brief Load from PIDF (h5) file
#
# @param thisFilePath File path, name, extension to read file from
#
# @return Unit for sensor acceleration, string
# @return Sensor sampling rate, float (Hz)
# @return Time in epoch, n x 1 array float array (sec)
# @return Acceleration, n x 3 float array (m/s2)
# @return Gyroscope, n x 3 float array (deg/s)
# @return Temperature, n x 3 float array (C)
#
def loadDataFromPIDFFile(thisFilePath):

    hfRead = h5py.File(thisFilePath, 'r')

    # Copy the data before closing h5py file
    sensorAccelUnit = hfRead['sensor/1/accelerometer'].attrs['unit']  # string

    sensorSamplingRateObj = hfRead['sensor/1/sampling_rate']
    sensorTimeObj = hfRead['sensor/1/time']  # n x 1
    sensorAccelXYZObj = hfRead['sensor/1/accelerometer']  # n x 3
    sensorGyroXYZObj = hfRead['sensor/1/gyroscope']  # n x 3
    sensorTempObj = hfRead['sensor/1/temperature']  # n x 1

    # Don't need to read the other quantities

    inSensorAccelUnit = sensorAccelUnit
    # load the object data to numpy array
    inSensorSamplingRate = sensorSamplingRateObj[()]  # as this is a scalar
    inSensorTime = sensorTimeObj[:]
    # [:] for newer h5py, use .value for older versions of h5py library
    inSensorAccelXYZ = sensorAccelXYZObj[:]
    inSensorGyroXYZ = sensorGyroXYZObj[:]
    inSensorTemp = sensorTempObj[:]

    hfRead.close()

    return inSensorAccelUnit, inSensorSamplingRate, inSensorTime, inSensorAccelXYZ, inSensorGyroXYZ, inSensorTemp


# @brief Load from tab separated text (.tsv) file
#
# File has 4 columns: t, ax, ay, az
#
# @param thisFilePath File path, name, extension to read file from
#
# @return Time in epoch, n x 1 array float array (sec)
# @return Acceleration, n x 3 float array (m/s2)
#
# [JO] pyarrow module (arrow.apache.org/docs/python/index.html)
# is faster in reading large files. 
def loadDataFromTsvFile(thisFilePath):
    read_options = csv.ReadOptions(autogenerate_column_names=True,
                                   skip_rows_after_names=1)
    parse_options = csv.ParseOptions(delimiter='\t')

    # tsvRead = np.loadtxt(thisFilePath, delimiter='\t', dtype=float)
    # [JO] tsvRead will now be pyarrow.Table - a collection of top-level
    # names, equal length Arrow arrays.
    tsvRead = csv.read_csv(thisFilePath,
                           read_options=read_options,
                           parse_options=parse_options)
    # [JO] You can convert this to a NumPy array.
    tsvRead = np.asarray(tsvRead)

    # load the object data to numpy array
    # [JO] getColFromNx4Array() is not necessary - just use .reshape method
    # of NumPy.
    # inSensorTime = getColFromNx4Array(tsvRead) # this is copied over
    inSensorTime = tsvRead[:, 0].reshape(-1, 1)
    inSensorAccelXYZ = tsvRead[:,1:4] # need to be careful with slicing here

    return inSensorTime, inSensorAccelXYZ


# @brief Given nx4 array get first column as nx1 array
#
# This is needed because slice and transpose does not work
#
# @param inArray The nx1 array for which first column is needed
#
# @return First column as nx1 array
#
def getColFromNx4Array(inArray):

    numRows = len(inArray)
    outArray = np.zeros((numRows,1))
    i = 0
    while (i < numRows):
        outArray[i,0] = inArray[i,0]
        i = i + 1

    return outArray


# @brief Calculate vector magnitude of acceleration
#
# @param accelData Nx3 array of accelerometer data, float, (m/s2)
#
def calcAccelMag(myArr):
    numPts = len(myArr)
    mag = np.zeros((numPts,1))
    k = 0

    while (k < numPts):
        mag[k,0] = np.sqrt(myArr[k,0]*myArr[k,0] + myArr[k,1]*myArr[k,1] + myArr[k,2]*myArr[k,2])
        k = k + 1

    return mag


# @brief Find a baseline acceleration using segments of non-movement
#
# TODO: Keep track of this and see if baseline drifts over time?
#
# @param accelData Nx1 array of accelerometer magnitude data, float, (m/s2)
# @param sampleSize Window size to group the data into for computing mean baseline, int (no units)
# @param devRange Deviation range, i.e. maximum acceptable difference between min and max to consider as static (no movement), float (m/s2)
# @param maxIndexLimit Maximum number of samples up to which to search for baseline (no units)
#
#
# @return baseline
#
def getVariableBaseline(accelData, sampleSize, devRange, maxIndexLimit):

    endIdx = len(accelData)
    baseline = 0.0;
    #printToPrintBuffer(numDataPtsHere)
    # go in blocks of 10/sampleSize, how many blocks of 10/sampleSize
    i = 0; # count the blocks of data

    tempSum = 0.0;
    tempMean = 0.0;
    sumOfMeans = 0.0; # store a baseline value every time it is detected
    countOfMeans = 0 # count how many times baeline is detected

    i = 0 + sampleSize # start a bit later to allow for back calculation

    # first, cycle through and compute the baseline
    while (i <= endIdx):
        # go through block by block, and see if the min/max difference is below 
        tempDiff = abs(np.amax(accelData[i-sampleSize:i]) - np.amin(accelData[i-sampleSize:i]))
        if tempDiff <= devRange:
            tempMean = np.sum(accelData[i-sampleSize:i])/float(sampleSize)
            sumOfMeans = sumOfMeans + tempMean
            countOfMeans = countOfMeans + 1

        i = i + sampleSize

    # Use median if unable to compute
    if countOfMeans > 1:
        baseline = sumOfMeans/float(countOfMeans)
    else:
        baseline = np.median(accelData[0:endIdx])
        printToPrintBuffer("WARNING: Could not find a quiet period for baseline, using median")


    return baseline


# @brief Given a value in an array, search for its index the array
#
# Expect value to be near the start. Go one index over if exact match is
# not found.
#
# @param myVal The value to search for, float
# @param myArr Nx1 array in which to search, float
#
# @return Index, int
#
def findArrayElementNearStart(myVal, myArr):
    numDataPts = len(myArr)
    i = 0
    while i < numDataPts:
        if myArr[i] >= float(myVal):
            break

        i = i + 1

    return i


# @brief Given a value in an array, search for its index the array
#
# Expect value to be near the end. Go one index under if exact match is
# not found.
#
# @param myVal The value to search for, float
# @param myArr Nx1 array in which to search, float
#
# @return Index, int
#
def findArrayElementNearEnd(myVal, myArr):
    numDataPts = len(myArr)
    i = numDataPts - 2 # 1 extra less as there may be a bug with last time value = 0
    while i >= 0:
        if myArr[i] <= float(myVal):
            break

        i = i - 1

    return i


# @brief Get a subset of the time series data that excludes extra recorded time that is not needed
#
# @param accelObject Full data, PhysActSeries
# @param startTime Start time for subset, UTC time int (seconds)
# @param startTime End time for subset, UTC time int (seconds)
#
# TODO: fix start time, end time selection
#
# @return Subset of the data, PhysActSeries
#
def selectiveWindow(accelObject, startTime, endTime):
    # startIdx = 0 # from the source object # this override is obsolete
    # endIdx = accelObject.numDataPts # from the source object # this override is obsolete
    startIdx = findArrayElementNearStart(startTime, accelObject.time)
    endIdx = findArrayElementNearEnd(endTime, accelObject.time)
    printToPrintBuffer("Start index: " + str(startIdx) + " End Index: " + str(endIdx))
    newTimeSeries = np.zeros((endIdx - startIdx + 1, 1))
    newAccelMagSeries = np.zeros((endIdx - startIdx + 1, 1))
    i = 0  # destination array

    j = startIdx  # source array
    while j < endIdx:

        newTimeSeries[i, 0] = accelObject.time[j, 0]
        newAccelMagSeries[i, 0] = accelObject.accelMagLeft[j, 0]
        i = i + 1
        j = j + 1

    accelObjectSubset = loadPhysActSeries(newTimeSeries, newAccelMagSeries) # load into physical activity series object for easier processing

    return accelObjectSubset


# @brief Given Nx1 array, add the absolute values of all the elements
#
# Useful for integrating
#
def sumAbs(myArr):

    numDataPtsHere = len(myArr);
    i = 0;
    sumAbs = 0.0;
    while i < numDataPtsHere:
        sumAbs = sumAbs + abs(myArr[i]);
        i = i + 1;

    return sumAbs;


# @brief Given Nx1 array, get the integral of the jerk-time curve
#
#
def integratedJerk(myArr):

    # jerk = da/dt
    # integrate jerk = da/dt*dt = da = (a_i - a_i-1)
    numDataPtsHere = len(myArr)
    i = 1  # skip the first one
    sumJerk = 0.0
    while i < numDataPtsHere:
        sumJerk = sumJerk + abs(myArr[i] - myArr[i-1])
        i = i + 1

    return sumJerk

# @brief Calculate the score based on thresholds and computed quantity
#
# Quantity could be anything, e.g., median, integral
#
# Score 0 if quantity <= threshLow, 3.0 if threshLow < quantity <= threshHigh,
# 6.0 if quantity > threshHigh
#
# @param calcQtty Some computed quantity (don't care about units)
# @param threshLow Lower threshold to use for scoring calcQtty (units must be same as calcQtty)
# @param threshHigh Higher threshold to use for scoring calcQtty (units must be same as calcQtty)
#
def getClassificationComputedQtty(computedQtty, threshLow, threshHigh):

    calcQttyArrScore = 0.0
    if ((computedQtty > threshLow) and (computedQtty <= threshHigh)):
        calcQttyArrScore = 3.0
    elif computedQtty > threshHigh:
        calcQttyArrScore = 6.0

    return calcQttyArrScore


# @brief Compute physical activity class (acceleration-time) given gravity-included acceleration and baselines
#
# -Calculate gravity independent magnitude of acceleration, a_IND
# -Scale the acceleration by leg length, a_ADJ
# -Calculate computed quantity (jerk or accel)
# -Calculate total computed quantity
# -Assign label based on threshold
#
# Computed quantity is area under the accel-time curve, i.e
# a_(i-n)*dT + ... a_(i-1)*dT + a_i*dT or simply sum of a * dT, i.e.,
# (a_(i-n) + ... a_(i-1) + a_i)*dT
#
# Generally working with 20 Hz sampling rate, and 2 second window, so a window
# length of 40 and time step of 0.050 s should be used. Classification labels
# are 0.0 sedentary, 3.0 light, 6.0 moderate-to-vigorous, 999.0 undefined. The
# undefined class is only so at the first (window length - 1) points.
#
# @param accelObject PhysActSeries with index, time, accelMagLeft, accelMagRight filled in (m/s2)
# @param baselineAccel Baseline acceleration value, float, (m/s2)
# @param thisLegLengthCm Leg length as measured for this individual, float (cm)
# @param baselineLegLengthCm Standard leg length to scale accel data, float (cm)
# @param windowLen Window length for computed qtty, in terms of number of data points, use 40 @ 20 Hz sampling rate, int, (dimensionless)
# @param dT Time step size, use 0.050 s @ 20 Hz sampling rate, int (s)
# @param threshLower Threshold between sedentary and light physical activity, in terms of accel-time integral, float (m/s)
# @param threshUpper Threshold between light and moderate-to-vigorous physical activity, in terms of accel-time integral, float (m/s)
#
# @return PhysActSeries with computed quantities and classification also filled in
#
def computePhysActivityAccelTime(accelObject, baselineAccel, thisLegLengthCm, baselineLegLengthCm, windowLen, dT, threshLower, threshUpper):

    numDataPts = accelObject.numDataPts

    thisScaleFactor = thisLegLengthCm/baselineLegLengthCm # short legs, small scale factor - long legs, larger scale factor
    # short legs, small scale factor, increase newAccel - longer legs, larger scale factor, decrease newAccel

    #splitting into two while loops to  eliminate if/else check, making things a bit faster
    # for the length-1 of the first window, we just adjust the accel, we don't compute accel/jerk or classification
    i = 0
    while i < (windowLen-1):
        # Calculate gravity independent magnitude of acceleration, a_IND, left and right
        # Scale the acceleration by leg length, a_ADJ, left and right
        accelObject.accelMagLeft[i,0] = (accelObject.accelMagLeft[i,0] - baselineAccel)/thisScaleFactor

        accelObject.classification[i,0] = 999.0 # undefined classification

        i = i + 1

    # for the rest of the data, we compute everything...adjust the accel, compute accel/jerk, classification
    i = windowLen - 1
    while i < numDataPts:
        # Calculate gravity independent magnitude of acceleration, a_IND, left and right
        # Scale the acceleration by leg length, a_ADJ, left and right
        accelObject.accelMagLeft[i,0] = (accelObject.accelMagLeft[i,0] - baselineAccel)/thisScaleFactor

        # Calculate computed quantity (jerk or accel), left and right
        accelObject.computedQttyLeft[i,0] = sumAbs(accelObject.accelMagLeft[i-windowLen:i,0])*dT # returns in (m/s)

        # Calculate total quantity quantity (left + right)
        accelObject.computedQttyTotal[i,0] = accelObject.computedQttyLeft[i,0] + 0.0

        # Assign label based on threshold
        accelObject.classification[i,0] = getClassificationComputedQtty(accelObject.computedQttyTotal[i,0], threshLower, threshUpper)

        i = i + 1

    return accelObject


# @brief Compute physical activity class (jerk-time) given gravity-included acceleration
#
# -Calculate gravity independent magnitude of acceleration, a_IND, left and right
# -Scale the acceleration by leg length, a_ADJ, left and right
# -Calculate computed quantity (jerk or accel), left and right
# -Calculate total computed quantity (left + right)
# -Assign label based on threshold
#
# Computed quantity is area under the jerk-time curve, i.e
# if jerk = da/dt
# integral jerk = d(a_i-n)/dt*dt + ... + da_i/dt*dt
# i.e., integral jerk = d(a_i-n) + ... + da_i
#
# Generally working with 20 Hz sampling rate, and 2 second window, so a window
# length of 40 and time step of 0.50 s should be used. Classification labels
# are 0.0 sedentary, 3.0 light, 6.0 moderate-to-vigorous, 999.0 undefined. The
# undefined class is only so at the first (window length - 1) points
#
# @param accelObject PhysActSeries with index, time, accelMagLeft, accelMagRight filled in (m/s2)
# @param baselineAccel Baseline acceleration value, left leg, float, (m/s2)
# @param thisLegLengthCm Leg length as measured for this individual, float (cm)
# @param baselineLegLengthCm Standard leg length to scale accel data, float (cm)
# @param windowLen Window length for computed qtty, in terms of number of data points, use 40 @ 20 Hz sampling rate, int, (dimensionless)
# @param dT Time step size, use 0.50 s @ 20 Hz sampling rate, int (s)
# @param threshLower Threshold between sedentary and light physical activity, in terms of jerk-time integral, float (m/s)
# @param threshUpper Threshold between light and moderate-to-vigorous physical activity, in terms of jerk-time integral, float (m/s)
#
# @return PhysActSeries with computed quantities and classification also filled in
#
def computePhysActivityJerkTime(accelObject, baselineAccel, thisLegLengthCm, baselineLegLengthCm, windowLen, dT, threshLower, threshUpper):

    numDataPts = accelObject.numDataPts

    thisScaleFactor = thisLegLengthCm/baselineLegLengthCm # short legs, small scale factor - long legs, larger scale factor
    # short legs, small scale factor, increase newAccel - longer legs, larger scale factor, decrease newAccel

    #splitting into two while loops to  eliminate if/else check, making things a bit faster
    # for the length-1 of the first window, we just adjust the accel, we don't compute accel/jerk or classification
    i = 0
    while (i < windowLen-1):
        # Calculate gravity independent magnitude of acceleration, a_IND, left and right
        # Scale the acceleration by leg length, a_ADJ, left and right
        accelObject.accelMagLeft[i,0] = (accelObject.accelMagLeft[i,0] - baselineAccel)/thisScaleFactor

        accelObject.classification[i,0] = 999.0 # undefined classification

        i = i + 1

    # for the rest of the data, we compute everything...adjust the accel, compute accel/jerk, classification
    i = windowLen-1
    while i < numDataPts:
        # Calculate gravity independent magnitude of acceleration, a_IND, left and right
        # Scale the acceleration by leg length, a_ADJ, left and right
        accelObject.accelMagLeft[i,0] = (accelObject.accelMagLeft[i,0] - baselineAccel)/thisScaleFactor

        # Calculate computed quantity (jerk or accel), left and right
        accelObject.computedQttyLeft[i,0] = integratedJerk(accelObject.accelMagLeft[i-windowLen:i,0]) # returns in (m/s2)

        # Calculate total quantity quantity (left + right)
        accelObject.computedQttyTotal[i,0] = accelObject.computedQttyLeft[i,0] + 0.0

        # Assign label based on threshold
        accelObject.classification[i,0] = getClassificationComputedQtty(accelObject.computedQttyTotal[i,0], threshLower, threshUpper)

        i = i + 1

    return accelObject


# @brief Compute physical activity class given gravity-included acceleration
#
# -Calculate gravity independent magnitude of acceleration, a_IND, left and right
# -Scale the acceleration by leg length, a_ADJ, left and right
# -Calculate computed quantity (jerk or accel), left and right
# -Calculate total computed quantity (left + right)
# -Assign label based on threshold
#
# Generally working with 20 Hz sampling rate, and 2 second window, so a window
# length of 40 and time step of 0.050 s should be used. Classification labels
# are 0.0 sedentary, 3.0 light, 6.0 moderate-to-vigorous, 999.0 undefined. The
# undefined class is only so at the first (window length - 1) points
#
# @param computedQtty Computed quantity to use for activity classification, can be "jerk" or "acceleration" string
# @param optimizationType Type of optimization used to compute threshold, use "TP" for maximum true positive rate or "PAP" for predictive activity proportion, string
# @param accelObject PhysActSeries with index, time, accelMagLeft, accelMagRight filled in (m/s2)
# @param baselineAccel Baseline acceleration value, leg, float, (m/s2)
# @param thisLegLengthCm Leg length as measured for this individual, float (cm)
# @param baselineLegLengthCm Standard leg length to scale accel data, float (cm)
# @param windowLen Window length for computed qtty, in terms of number of data points, use 40 @ 20 Hz sampling rate, int, (dimensionless)
# @param dT Time step size, use 0.050 s @ 20 Hz sampling rate, int (s)
#
# @return PhysActSeries with computed quantities and classification also filled in
#
def computePhysicalActivity(computedQtty, optimizationType, accelObject, baselineAccel, thisLegLengthCm, baselineLegLengthCm, windowLen, dT):

    # Using these globally defined thresholds
    global constThreshLowerJerkTP
    global constThreshUpperJerkTP
    global constThreshLowerJerkPAP
    global constThreshUpperJerkPAP
    global constThreshLowerAccelTP
    global constThreshUpperAccelTP
    global constThreshLowerAccelPAP
    global constThreshUpperAccelPAP

    if((computedQtty == "jerk") and (optimizationType == "TP")):
        accelObject = computePhysActivityJerkTime(accelObject, baselineAccel, thisLegLengthCm, baselineLegLengthCm, windowLen, dT, constThreshLowerJerkTP, constThreshUpperJerkTP)

    elif((computedQtty == "jerk") and (optimizationType == "PAP")):
        accelObject = computePhysActivityJerkTime(accelObject, baselineAccel, thisLegLengthCm, baselineLegLengthCm, windowLen, dT, constThreshLowerJerkPAP, constThreshUpperJerkPAP)

    elif((computedQtty == "acceleration") and (optimizationType == "TP")):
        accelObject = computePhysActivityAccelTime(accelObject, baselineAccel, thisLegLengthCm, baselineLegLengthCm, windowLen, dT, constThreshLowerAccelTP, constThreshUpperAccelTP)

    elif((computedQtty == "acceleration") and (optimizationType == "PAP")):
        accelObject = computePhysActivityAccelTime(accelObject, baselineAccel, thisLegLengthCm, baselineLegLengthCm, windowLen, dT, constThreshLowerAccelPAP, constThreshUpperAccelPAP)

    else:
        printToPrintBuffer("ERROR: Incorrect selection of computedQtty and/or optimizationType (computePhysicalActivity)")

    return accelObject

# A 3-class statistics is an object that contains sumary statstics, e.g., count
# and percentage for each of the classes
#
class ThreeClassStats:

    def __init__(self):

        self.countTotal = 0         # total number of events, int (dimensionelss)
        self.countSedentary = 0     # number of events rated as sedentary by algortihm, int (dimensionelss)
        self.countLight = 0         # number of events rated as light by algorithm, int (dimensionelss)
        self.countMV = 0            # number of events rated as moderate-to-vigorous by algorithm, int (dimensionelss)
        self.countUndefined = 0     # number of events rated as not defined by algorithm, int (dimensionelss)

        self.countTotalPct = 0.0        # countTotal as a percentage of the total number of events, used as a check later, float (percent)
        self.countSedentaryPct = 0.0    # countSedentary as a percentage of the total number of events, float (percent)
        self.countLightPct = 0.0        # countLight as a percentage of the total number of events, float (percent)
        self.countMVPct = 0.0           # countMV as a percentage of the total number of events, float (percent)
        self.countUndefinedPct = 0.0    # countUndefined as a percentage of the total number of events, float (percent)

        self.timeTotalSec = 0.0         # total time for the entire period, float (seconds)
        self.timeSedentarySec = 0.0     # time spent in sedentary activity, float (seconds)
        self.timeLightSec = 0.0         # time spent in light activity, float (seconds)
        self.timeMVSec = 0.0            # time spent in MV activity, float (seconds)
        self.timeUndefinedSec = 0.0     # time spent in activity that was undefined, float (seconds)


    # @brief Count how many instances of each of the classes as rated by the algorithm
    def countClasses(self, seriesData):

        numDataPts = seriesData.numDataPts
        i = 0
        while i < numDataPts:

            self.countTotal = self.countTotal + 1

            if(isEqualClassification036(seriesData.classification[i,0], 0.0)):
                self.countSedentary = self.countSedentary + 1
            elif(isEqualClassification036(seriesData.classification[i,0], 3.0)):
                self.countLight = self.countLight + 1
            elif(isEqualClassification036(seriesData.classification[i,0], 6.0)):
                self.countMV = self.countMV + 1
            else:
                self.countUndefined = self.countUndefined + 1

            i = i + 1


    # @brief Calculate percentage rated by algorithm, as percentage of total counts in the entire data set
    #
    def calculatePercentage(self):

        if(self.countTotal > 0):
            self.countTotalPct = float(self.countTotal)/float(self.countSedentary + self.countLight + self.countMV + self.countUndefined)*100.0
            self.countSedentaryPct = float(self.countSedentary)/float(self.countSedentary + self.countLight + self.countMV + self.countUndefined)*100.0
            self.countLightPct = float(self.countLight)/float(self.countSedentary + self.countLight + self.countMV + self.countUndefined)*100.0
            self.countMVPct = float(self.countMV)/float(self.countSedentary + self.countLight + self.countMV + self.countUndefined)*100.0
            self.countUndefinedPct = float(self.countUndefined)/float(self.countSedentary + self.countLight + self.countMV + self.countUndefined)*100.0
        else:
            printToPrintBuffer("ERROR: ThreeClassStats.calculatePercentage() Cannot calculate percentage. countTotal = 0. Did you run countClasses() first?")


    # @brief Calculate the time spend on each activity classification.
    #
    # Instead of trying to count and compile time through every instance, it
    # is simpler to calculate the time based on percentage count and total time.
    #
    def calculateTime(self, seriesData):

        numDataPts = seriesData.numDataPts
        if( (self.countTotal > 0) or (self.countTotalPct < 1.0) ):
            # TODO: Fix the bug where thelast time instance is zero for some reason
            # self.timeTotalSec = seriesData.time[numDataPts-1,0] - seriesData.time[0,0] # ideally sbtract from last instance but it is zero
            self.timeTotalSec = seriesData.time[numDataPts-2,0] - seriesData.time[0,0]
            self.timeSedentarySec = self.countSedentaryPct*self.timeTotalSec/100.0
            self.timeLightSec = self.countLightPct*self.timeTotalSec/100.0
            self.timeMVSec = self.countMVPct*self.timeTotalSec/100.0
            self.timeUndefinedSec = self.countUndefinedPct*self.timeTotalSec/100.0

        else:
            printToPrintBuffer("ERROR: ThreeClassStats.calculateTime() Cannot calculate time. countTotal = 0 or countTotalPct < 1. Did you run countClasses() and calculatePercentage() first?")


    # JO: adapting this function to prepare a json format output.
    def printThreeClassStats(self):

        threeClassStatsStringDict = {
                "Counts": {
                    "total": self.countTotal,
                    "sedentary": self.countSedentary,
                    "light": self.countLight,
                    "MV": self.countMV,
                    "undefined": self.countUndefined
                    },
                "Percent": {
                    "total": self.countTotalPct,
                    "sedentary": self.countSedentaryPct,
                    "light": self.countLightPct,
                    "MV": self.countMVPct,
                    "undefined": self.countUndefinedPct
                    },
                "Time (min)": {
                    "total": self.timeTotalSec/60.0,
                    "sedentary": self.timeSedentarySec/60.0,
                    "light": self.timeLightSec/60.0,
                    "MV": self.timeMVSec/60.0,
                    "undefined": self.timeUndefinedSec/60.0
                    }
                }

        countsHeaderStr = "\nCounts\ntotal\tsedentary\tlight\tMV\tundefined\n"
        countsDataStr = str(self.countTotal) + "\t" + str(self.countSedentary) + "\t" + str(self.countLight) + "\t" + str(self.countMV) + "\t" + str(self.countUndefined)

        pctHeaderStr = "\nPercent\ntotal\tsedentary\tlight\tMV\tundefined\n"
        pctDataStr = str(self.countTotalPct) + "\t" + str(self.countSedentaryPct) + "\t" + str(self.countLightPct) + "\t" + str(self.countMVPct) + "\t" + str(self.countUndefinedPct)

        timeHeaderStr = "\nTime (min)\ntotal\tsedentary\tlight\tMV\tundefined\n"
        timeDataStr = str(self.timeTotalSec/60.0) + "\t" + str(self.timeSedentarySec/60.0) + "\t" + str(self.timeLightSec/60.0) + "\t" + str(self.timeMVSec/60.0) + "\t" + str(self.timeUndefinedSec/60.0)

        allThreeClassStatsString = countsHeaderStr + countsDataStr + '\n' + pctHeaderStr + pctDataStr + '\n' + timeHeaderStr + timeDataStr
        printToPrintBuffer(allThreeClassStatsString)

        return threeClassStatsStringDict


# @brief Compare and see if the two classification codes (floats) are equal
#
# @param FIrst classification code, float (diemsnionless)
# @param Second clasification code, float (dimensionless)
#
# @return True if equal, false if not equal
#
def isEqualClassification036(float1, float2):

    if abs(float1 - float2) < 0.1:
        return True
    else:
        return False


# @brief Collect adjacent identical activity instances in time series into groups (will have empty rows at end).
#
# if we have elemnts
# time [0 1 2 3 4 5 6 7 8 9 10]
# qty [0 0 0 1 0 0 2 2 1 1 1]
# Then group adjacent similar elements
# [0-2 3 4-5 6-7  8-10]
# [0 1 0 2 1]
#
# Note that this will return an object with a lot of "blanks" at the end and
# those need to be removed/truncated.
#
# Test status: pass
#
# @param PhysActSeries object holding the entire movement series, with classification
#
# @return number of activity regions, ActivityRegions object with lots of empty rows at end
#
def getGroupedActivityRegionsLong(expandedDataLong):

    nMax = expandedDataLong.numDataPts
    # pick out the intervals in a larger array
    compressedDataLong = ActivityRegions(nMax) # this stores the individual regions of actitivity, e.g. MV from 3-10 seconds
    counterActivityRegion = 0
    # first data point marks the start of the
    prevClassification = expandedDataLong.classification[0,0]
    prevClassificationIndex = expandedDataLong.index[0,0]
    prevClassificationTime = expandedDataLong.time[0,0]
    # save the start of the first activity region
    compressedDataLong.classification[0,0]  = prevClassification
    compressedDataLong.idxStart[0,0] = prevClassificationIndex
    compressedDataLong.timeStart[0,0] = prevClassificationTime
    i = 1 # loop from the second data point onwards
    while i < nMax:
        # currClassification = classification[i,1]
        currClassification = expandedDataLong.classification[i,0]
        currClassificationIndex = expandedDataLong.index[i,0]
        currClassificationTime = expandedDataLong.time[i,0]
        # shift only if current class is different from precious cativity, or we have reached the last data point
        if(isEqualClassification036(prevClassification, currClassification) == False) or (i == (nMax-1)):
            # printToPrintBuffer(str(prevClassification) + ": " + " " + "-" + str(prevClassificationIndex))
            # wrap up previous
            compressedDataLong.idxEnd[counterActivityRegion,0] = prevClassificationIndex
            compressedDataLong.timeEnd[counterActivityRegion,0] = prevClassificationTime

            # update count
            counterActivityRegion = counterActivityRegion + 1

            # start next, can't do this if at last point
            if i < (nMax - 1):
                compressedDataLong.classification[counterActivityRegion,0] = currClassification
                compressedDataLong.idxStart[counterActivityRegion,0] = currClassificationIndex
                compressedDataLong.timeStart[counterActivityRegion,0] = currClassificationTime


            # compressedDataLong.printString(compressedDataLong.classification[counterActivityRegion-1,0], compressedDataLong.idxStart[counterActivityRegion-1,0], compressedDataLong.idxEnd[counterActivityRegion-1,0], compressedDataLong.timeStart[counterActivityRegion-1,0], compressedDataLong.timeEnd[counterActivityRegion-1,0])

        prevClassification = currClassification
        prevClassificationIndex = currClassificationIndex
        prevClassificationTime = currClassificationTime

        i = i + 1

    return counterActivityRegion, compressedDataLong


# @brief Collect adjacent identical activity instances in time series into groups.
#
# if we have elemnts
# time [0 1 2 3 4 5 6 7 8 9 10]
# qty [0 0 0 1 0 0 2 2 1 1 1]
# Then group adjacent similar elements
# [0-2 3 4-5 6-7  8-10]
# [0 1 0 2 1]
#
# Test status:
# TODO: Istead of cycling again through shortened, simply change the number of activty opbjects
#
# @param PhysActSeries object holding the entire movement series, with classification
#
# @return ActivityRegions object that holds grouped identical activity instances
#
def getGroupedActivityRegions(accelObject):
    # nMax = numDataPts
    # shortDatTemp = ActivityRegions(nMax) # this stores the individual regions of activity, e.g. MV from 3-10 seconds
    numActivityRegions, activityRegionsExtraZeros = getGroupedActivityRegionsLong(accelObject) # Object/arrays of activity regions, a lot of them blank at the end
    #activityRegionsNoExtras = getGroupedActivityRegionsShortened(numActivityRegions, activityRegionsExtraZeros)  # Object/arrays of activity regions, with end blanks removed
    # instead of copying over to a shorter object, and wasting time, just set the variable for no. of data points to the smaller number
    activityRegionsExtraZeros.numDataPts = numActivityRegions # # Object/arrays of activity regions, a lot of them blank at the end

    return activityRegionsExtraZeros


# @brief Compile the summary statistics (count, percentage, time spent, etc.)
#
# @param PhysActSeries object holding the entire movement series, with classification
#
# @return Object with statistics in it
# @return Formatted string with statistics in it, good for writing to file
#
def getSummaryStatistics(accelObject):

    myStatistics = ThreeClassStats() # create the object
    myStatistics.countClasses(accelObject) # work on it
    myStatistics.calculatePercentage() # work on it
    myStatistics.calculateTime(accelObject) # work on it
    myStatisticsPrintStr = myStatistics.printThreeClassStats() # work on it TODO: this should return something to print

    return myStatistics, myStatisticsPrintStr


# @brief  Write the string for the summary stats to file
#
# @param toWriteStr The string to be written to file, this contains the summary stats
# @param outputDirStr The output directory string, e.g., "output/results"
# @param fNameStr Original (source) file name string
# @param fNameAppendStr What to append to file name string, e.g. "output"
# @param fExtensionStr Output file extension string, e.g. ".txt"
#
# @return Void
#
def writeSummaryStatsToFile(toWriteStr, outputDirStr, fNameStr, fNameAppendStr, fExtensionStr):

    textFileWriter = open(outputDirStr + "/" + fNameStr + "_" + fNameAppendStr + fExtensionStr, 'w')
    textFileWriter.write(toWriteStr)
    textFileWriter.close()


# @brief  Write the activity bouts (regions) to file
#
# Take the object and write it in comma separated format. Write in the format
# start time, end time, duration, classification.
#
# @param myGroupedActivityRegionsObject The activity regions object
# @param outputDirStr The output directory string, e.g., "output/results"
# @param fNameStr Original (source) file name string
# @param fNameAppendStr What to append to file name string, e.g. "output"
# @param fExtensionStr Output file extension string, e.g. ".txt"
#
# @return Void
#
def writeActivityBoutsToFile(myGroupedActivityRegionsObject, outputDirStr, fNameStr, fNameAppendStr, fExtensionStr):

    maxNum = myGroupedActivityRegionsObject.numDataPts
    # i = 0
    # bufferString = "start_time_sec,end_time_sec,duration_sec,classification\n"
    # while i < maxNum:
    #     bufferString = bufferString + str(myGroupedActivityRegionsObject.timeStart[i,0]) + "," + str(myGroupedActivityRegionsObject.timeEnd[i,0]) + "," + str(myGroupedActivityRegionsObject.timeEnd[i,0] - myGroupedActivityRegionsObject.timeStart[i,0]) + "," + str(myGroupedActivityRegionsObject.classification[i,0]) + "\n"
    #     i = i + 1
    # # printToPrintBuffer(bufferString)
    # textFileWriter = open(outputDirStr + "/" + fNameStr + "_" + fNameAppendStr + fExtensionStr, 'w')
    # textFileWriter.write(bufferString)
    # textFileWriter.close()
    bouts = pyarrow.table([myGroupedActivityRegionsObject.timeStart[0:maxNum, 0],
                           myGroupedActivityRegionsObject.timeEnd[0:maxNum, 0],
                           myGroupedActivityRegionsObject.timeEnd[0:maxNum, 0] - myGroupedActivityRegionsObject.timeStart[0:maxNum, 0],
                           myGroupedActivityRegionsObject.classification[0:maxNum, 0]],
                          names = ["start_time_sec", "end_time_sec", "duration_sec", "classification"])
    csv.write_csv(bouts,
                  outputDirStr + "/" + fNameStr + "_" + fNameAppendStr + fExtensionStr,
                  write_options = csv.WriteOptions(include_header=True, delimiter='\t'))


# @brief  Write the all instances of activity classifications to file
#
# Take the object and write it in comma separated format. Write in the format
# time, classification.
#
# @param myPhysActSeriesObject The PhysActSeriesObject regions object
# @param outputDirStr The output directory string, e.g., "output/results"
# @param fNameStr Original (source) file name string
# @param fNameAppendStr What to append to file name string, e.g. "output"
# @param fExtensionStr Output file extension string, e.g. ".txt"
#
# @return Void
#
def writeActivityAllToFile(myPhysActSeriesObject, outputDirStr, fNameStr, fNameAppendStr, fExtensionStr):

    try:
        textFileWriteStartTime = time.process_time()
        # outSingleArray = np.hstack((myPhysActSeriesObject.time,myPhysActSeriesObject.classification))
        outTable = pyarrow.table({'epoch_time': myPhysActSeriesObject.time[:,0].round(5),
                                  'class': myPhysActSeriesObject.classification[:,0].astype('int')})
        # %.5f means non scientific, simple float number, with 5 decimal places
        # for epoch time (sec)  3 digits after decimal should be enough of precision
        # for acceleration (m/s2)  5 digits after decimal should be enough precision
        # np.savetxt(outputDirStr + "/" + fNameStr + "_" + fNameAppendStr + fExtensionStr, outSingleArray, fmt='%.5f', delimiter=',', newline='\n')
        csv.write_csv(outTable,
                      outputDirStr + "/" + fNameStr + "_" + fNameAppendStr + fExtensionStr,
                      write_options=csv.WriteOptions(include_header=True, delimiter='\t')
                      )
        textFileWriteEndTime = time.process_time()
        printToPrintBuffer("INFO: Time to write all instances of activity (raw): " + str(textFileWriteEndTime - textFileWriteStartTime) + " seconds")
    except BaseException as e:
        printToPrintBuffer(f"ERROR: Failed to write instances of activity (raw) to file; {str(e)}")


# @brief  Write the string for the error log to file
#
# @param toWriteStr The string to be written to file, this contains the error log
# @param outputDirStr The output directory string, e.g., "output/results"
# @param fNameStr Original (source) file name string
# @param fNameAppendStr What to append to file name string, e.g. "output"
# @param fExtensionStr Output file extension string, e.g. ".txt"
#
# @return Void
#
def writeErrorLogToFile(outputDirStr, fNameStr, fNameAppendStr, fExtensionStr):
    global printBufferStr
    writeSummaryStatsToFile(printBufferStr, outputDirStr, fNameStr, fNameAppendStr, fExtensionStr) # this function just writes a string to file


# @brief Show the user the start and end times from file
#
def displayTimeFromFile(startTime, EndTime):
    printToPrintBuffer("Start time (epoch):" + str(startTime) +", " + "End time (epoch):" + str(EndTime) )
    printToPrintBuffer("Start time:" + dateTimeEpochToRegularDateTime(startTime) +", " + "End time:" + dateTimeEpochToRegularDateTime(EndTime) )


# @brief Try to calculate the start time requested by the user
#
def calcUserStartTime(userInStartYear, userInStartMonth, userInStartDay, userInStartHour, userInStartMinute, userInStartSecond):
    global dataInputErrorFlag
    thisStartTime = 0
    
    try:
        # need to specify utc timezone otherwise it defaults to system timezone
        thisStartTime = datetime(userInStartYear, userInStartMonth, userInStartDay, userInStartHour, userInStartMinute, userInStartSecond, tzinfo=timezone.utc)
        #userStartTimeUTC = userStartTime.replace()
    except:
        # need to specify utc timezone otherwise it defaults to system timezone
        printToPrintBuffer("ERROR: Start date or time out of range. Maybe the day and month don't match.")
        dataInputErrorFlag = 1

    return thisStartTime


# @brief Try to calculate the end time requested by the user
#    
def calcUserEndTime(userInEndYear, userInEndMonth, userInEndDay, userInEndHour, userInEndMinute, userInEndSecond):
    global dataInputErrorFlag
    thisEndTime = 0

    try:
        thisEndTime = datetime(userInEndYear, userInEndMonth, userInEndDay, userInEndHour, userInEndMinute, userInEndSecond, tzinfo=timezone.utc)
    except:
        printToPrintBuffer("ERROR: End date or time out of range. Maybe the day and month don't match.")
        dataInputErrorFlag = 1

    return thisEndTime


# @brief Load data file and display file start/end time
#
# @param fNameInputDir The file path without the file name and the last '/'
# @param fNameNoExtension The file name without the extension
# @param fExtension The complete file extension with "." e.g. ".txt"
#
# @return nx1 array time in epoch (sec)
# @return nx3 array acceleration (m/sec2)
# @return Earliest possible time from file in epoch (sec)
# @return Last possible time from file in epoch (sec)
#
def loadFileDisplayFile(fNameInputDir, fNameNoExtension, fExtension):
    """[JO] fNameInputDir to be passed as a pathlib.Path object or a string"""

    global fileReadErrorFlag

    try:
        printToPrintBuffer(getDatetimeNowString())
        printToPrintBuffer("INFO: Reading input file " + fNameNoExtension + fExtension)
        if fExtension == ".hdf5":
            sensorAccelUnit, sensorSamplingRate, sensorTime, sensorAccelXYZ, sensorGyroXYZ, sensorTemp = loadDataFromPIDFFile(
                str(fNameInputDir / (fNameNoExtension + fExtension)))
        elif fExtension == ".tsv":
            sensorTime, sensorAccelXYZ = loadDataFromTsvFile(
                str(fNameInputDir / (fNameNoExtension + fExtension)))

        # time limits based on the time recorded from the data
        # epoch time UTC time zone
        thisMinFileTime = sensorTime[0, 0] + 1.0
        thisMaxFileTime = sensorTime[len(sensorTime)-1, 0] - 1.0

        displayTimeFromFile(thisMinFileTime, thisMaxFileTime)
        fileReadErrorFlag = 0  # 0 if no error, 1 if error
        return sensorTime, sensorAccelXYZ, thisMinFileTime, thisMaxFileTime
    except BaseException as e:
        # [JO] changed from the original print statement to print
        # the actual error message
        printToPrintBuffer(f"Error: {str(e)}")
        fileReadErrorFlag = 1  # 0 if no error, 1 if error
        # return garbage
        return 0.0, 0.0, 0.0, 0.0


# @brief Given user input, process the data, save results
#
# Also perform a few checks in the process, mainly that the user input time
# is within range of the start time and end time from the file
#
# @param fileReadSensorTime nx1 array time in epoch (sec)
# @param fileReadSensorAccelXYZ nx3 array acceleration (m/sec2) 
# @param minFileTime Earliest possible time from file in epoch (sec)
# @param maxFileTime Last possible time from file in epoch (sec)
#
# @return Void
#
# [JO] Adapting the function parameters to work with HBCD batch processing
# protocol
# @param fileReadSensorTime same as the original variable
# @param fileReadSensorAccelXYZ same as the original variable
# @param computedQttyOption "acceleration" or "jerk"
# @param infantLegLengthCm an age-based estimate
# @param outputDir path to save output
# @param fileNameNoExtension output file name
def processDataSaveResults(fileReadSensorTime, fileReadSensorAccelXYZ, computedQttyOption,
                           infantLegLengthCm, outputDir, fileNameNoExtension):
    
    # error flags
    global fileReadErrorFlag  # 0 if no error, 1 if error
    global dataInputErrorFlag  # 0 if no error, 1 if error

    global constRefLegLength3To5MonthCm

    # file io stuff
    # global fileNameNoExtension  # same 3 day above
    global fileNameAppendRaw # time, type, 0.05 s interval
    global fileNameAppendBouts # start time, end time, type
    global fileNameAppendSummary  # OVer pct and time
    global fileNameAppendErrorLog  # All messages and errors from code
    # global inputFileExtension
    global outputFileExtension
    # global inputDir
    # global outputDir

    # user options
    # global computedQttyOption
    global optimizationTypeOption
    # global infantLegLengthCm

    # user input
    # start time, epoch time UTC time zone
    # global userInputStartYear
    # global userInputStartMonth
    # global userInputStartDay # should be 8
    # global userInputStartHour # should be 10
    # global userInputStartMinute
    # global userInputStartSecond
    # end time, epoch time UTC time zone
    # global userInputEndYear
    # global userInputEndMonth
    # global userInputEndDay  # should be 11
    # global userInputEndHour  # should be 10
    # global userInputEndMinute
    # global userInputEndSecond

    windowLen = 40  # number of data points in computation window, use 40 (20 Hz for 2 sec)
    deltaTimeSec = 0.05  # sampling time step (sec)

    printToPrintBuffer("INFO: User selected computed quantity " + computedQttyOption + ", optimization type " + optimizationTypeOption +", leg length " + str(infantLegLengthCm) + " cm")
    # Only process if file was read in successfully
    if fileReadErrorFlag == 0:

        # [JO] disabling this option, as the processing will be done on the entire dataset
        # Calculate times from user input, if there is a problem, the global dataInputErrorFlag will be set
        # userStartTime = calcUserStartTime(userInputStartYear, userInputStartMonth, userInputStartDay, userInputStartHour, userInputStartMinute, userInputStartSecond)
        # userEndTime = calcUserEndTime(userInputEndYear, userInputEndMonth, userInputEndDay, userInputEndHour, userInputEndMinute, userInputEndSecond)

        # if all good so far, check if time within range
        # if(dataInputErrorFlag == 0):
        #     printToPrintBuffer("INFO: User entered start time (epoch): " + str(int(userStartTime.timestamp())))
        #     printToPrintBuffer("INFO: User entered end time (epoch): " + str(int(userEndTime.timestamp())))

        #     if(int(userStartTime.timestamp()) < int(minFileTime)):
        #        printToPrintBuffer("ERROR: User entered start time is earlier than first time point in data")
        #        dataInputErrorFlag = 1
        #     if(int(userEndTime.timestamp()) > int(maxFileTime)):
        #         printToPrintBuffer("ERROR: User entered end time is later than last time point in data")
        #         dataInputErrorFlag = 1

        # if all good so far, do all the processing
        if dataInputErrorFlag == 0:
            # Do some preliminary calculation and get a subset of the data
            printToPrintBuffer("INFO: Computing magnitude and baseline for full data")
            accelMag = calcAccelMag(fileReadSensorAccelXYZ) # compute acceleration vector magnitude
            accelObject = loadPhysActSeries(fileReadSensorTime, accelMag) # load into physical activity series object for easier processing
            accelMagBaseline = getVariableBaseline(accelMag, 10, 0.10, 86400) # compute baseline acceleration a_G (gravity component)
            printToPrintBuffer("INFO: Acceleration baseline (m/s2): " + str(accelMagBaseline))

            # Before processing, select a subset or window of data based on user selection
            printToPrintBuffer("INFO: Selecting subset of the data")
            # accelObjectSelectedRegion = selectiveWindow(accelObject, int(userStartTime.timestamp()), int(userEndTime.timestamp()))

            # Now do all of these together
            # Calculate gravity independent magnitude of acceleration, a_IND,
            # Scale the acceleration by leg length, a_ADJ
            # Calculate computed quantity (jerk or accel), left
            # Assign label based on threshold
            printToPrintBuffer("INFO: Computing physical activity")
            # [JO] previously the third parameter was accelObjectSelectedRegion
            #  - replacing it with accelObject
            accelObjectSelectedRegion = computePhysicalActivity(computedQttyOption, optimizationTypeOption, accelObject, accelMagBaseline, infantLegLengthCm, constRefLegLength3To5MonthCm, windowLen, deltaTimeSec)

            # Now save the results to file
            # Output 1, summary
            printToPrintBuffer("INFO: Computing summary of results and writing to file")
            summaryStatsObject, summaryStatsString = getSummaryStatistics(accelObjectSelectedRegion)  # this is a string
            # JO: json output prepared
            json_object = json.dumps(summaryStatsString, indent=4)
            writeSummaryStatsToFile(json_object, str(outputDir), fileNameNoExtension, fileNameAppendSummary, '.json')

            # Output 2, bouts of activity
            printToPrintBuffer("INFO: Computing bouts of activity and writing to file")
            groupedActivityRegionsObject = getGroupedActivityRegions(accelObjectSelectedRegion)  # find activity regions
            writeActivityBoutsToFile(groupedActivityRegionsObject, str(outputDir), fileNameNoExtension, fileNameAppendBouts, '.tsv')

            # Output 3, raw time/activity classification
            printToPrintBuffer("INFO: Writing all activity labels to file")
            writeActivityAllToFile(accelObjectSelectedRegion, str(outputDir), fileNameNoExtension, fileNameAppendRaw, '.tsv') # accelObject PhysActSeries with index, time, accelMagLeft, accelMagRight filled in (m/s2)

        else:
            printToPrintBuffer("ERROR: Cannot compute physical activity")

    # Error log will be written no matter what as it can be helpful for debugging
    # Output 4, full error log
    writeErrorLogToFile(str(outputDir), fileNameNoExtension, fileNameAppendErrorLog, outputFileExtension)


# General procedure
# Compute baseline acceleration, getVariableBaseline()
# Compute physical activity given gravity-included acceleration and baselines computePhysicalActivity()
# Get group activity regions getGroupedActivityRegions()
# Get summary statistics getSummaryStatistics()

# Calculate magnitude
# Calculate baseline, a_G left and right
# ****************  TRY DO DO ALL OF THESE IN ONE LOOP? *****************
# Calculate gravity independent magnitude of acceleration, a_IND,
# Scale the acceleration by leg length, a_ADJ
# Calculate computed quantity (jerk or accel), left
# Assign label based on threshold
# ****************  TRY DO DO ALL OF THESE IN ONE LOOP? *****************
# Compile statistics (num points in each category, percentage activity)
#
#
def mainFunction(inputDir, inputFileNameNoExtension, outputFileNameNoExtension,
                 computedQttyOption, infantLegLengthCm):
    """
    [JO] Originally the function `mainFunction()` did not take any parameter.
    This has been modified.

    Parameters
    ----------
        inputDir: pathlib.Path
            BIDS-compatible output folder ('.../motion')
            This is the folder where tsv files of accelerometer data (L/R)
            calibrated and resampled at 20 Hz are stored

        inputFileNameNoExtension : str
            Name of a tsv file (DO NOT ATTACH '.tsv')

        outputFileNameNoExtension : str
            Prefix of the output files (ex. sub-xxxxx_ses-V0x)

        computedQttyOption : str
            "acceleration" or "jerk"

        infantLegLengthCm : float
            Refer to the variable `infantLegLengthCmDict` of run.py

    Returns
    -------
        None (Physical Acitivty analysis results are saved in txt files)
    """

    # file io stuff
    # global fileNameNoExtension  # same 3 day above
    global inputFileExtension
    # global inputDir

    outputDir = inputDir / 'PA'

    # [JO] loadFileDisplayFile returns four variables, including
    # minInputFileTime and maxInputFileTime.
    # We don't need these two, so replacing them with '_'.
    # STEP 1: LOAD FILE, DISPLAY FILE START/END TIME
    fileReadSensorTime, fileReadSensorAccelXYZ, _, _ = loadFileDisplayFile(inputDir, inputFileNameNoExtension, inputFileExtension)

    #  STEP 2: PROCESS FILE, SAVE RESULTS
    processDataSaveResults(fileReadSensorTime, fileReadSensorAccelXYZ,
                           computedQttyOption, infantLegLengthCm,
                           outputDir, outputFileNameNoExtension)

    return 0
# **** END ****
