"""
Written by Jinseok Oh, Ph.D.
2022/9/13 - present (as of 2023/8/25)

base.py is a python script porting MakeDataStructure_v2.m (+ α),
    a file prepared to extract data from APDM OPAL V2 sensors.
The class - BaseProcess - is inherited by other classes,
accommodating the need of processing data from different sensors.

© 2023-2024 Infant Neuromotor Control Laboratory. All rights reserved.
"""
from dataclasses import dataclass, field
import re
from datetime import datetime, timedelta, timezone
from itertools import chain, repeat
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, firwin, lfilter
from scipy.interpolate import interp1d, CubicSpline
import pytz
from EntropyHub import SampEn, FuzzEn


@dataclass
class Processed:
    """
    Notes
    -----
    Store preprocessed measures from raw IMU data.
    All measures are side-specific (Left vs. Right)

    1. accmags: detrended linear acceleration magnitude
    2. velmags: detrended angular velocity magnitude
    3. thresholds: Positive vs. Negative
    4. over_accth: indicator if a value is above or below a threshold
                   (0: No; 1: Yes)
    5. th_crossed: indicator if a value crossed a threshold
                   ( 1: over the positive threshold
                    -1: under the negative threshold
                     0: otherwise)

    4 and 5 are updated AFTER 1-3 are obtained.
    """
    accmags: dict
    velmags: dict
    thresholds: dict
    over_accth: dict = field(init=False)
    under_naccth: dict = field(init=False)
    th_crossed: dict = field(init=False)

    def __post_init__(self):
        if len(self.accmags.keys()) == 2:
            lpos_temp = self.accmags['lmag'] > self.thresholds['laccth']
            lneg_temp = self.accmags['lmag'] < self.thresholds['lnaccth']
            rpos_temp = self.accmags['rmag'] > self.thresholds['raccth']
            rneg_temp = self.accmags['rmag'] < self.thresholds['rnaccth']

            self.over_accth = {'L': lpos_temp, 'R': rpos_temp}
            self.under_naccth = {'L': lneg_temp, 'R': rneg_temp}

            self.th_crossed = {
                    'L': lpos_temp + lneg_temp * -1,
                    'R': rpos_temp + rneg_temp * -1}
        else:
            # For a single sensor use, the identifier is 'U' and 'umag'
            lpos_temp = self.accmags['umag'] > self.thresholds['accth']
            lneg_temp = self.accmags['umag'] < self.thresholds['naccth']
            self.over_accth = {'U': lpos_temp}  # [U]nidentified
            self.under_naccth = {'U': lneg_temp}
            self.th_crossed = {'U': lpos_temp + lneg_temp * -1}


@dataclass
class SubjectInfo:
    """
    Notes
    -----
    Store subject-specific information.
    (8/25/23) Should I prepare a .json sidecar that's BIDS compatible?

    1. fname: .cwa or .h5 file name (.csv for movesense active)
    2. record_times: timestamps of the first and the last VALID data points
    3. fs: sampling frequency (Hz)
    4. timezone: timezone of the location where sensors were setup
    5. label_r: string used to indicate the right side (required for Opal V2)
    6. rowidx: indices of the rows to be processed
    7. recordlen: length of a recording in the number of data points
    """
    fname: list
    record_times: dict
    fs: int
    timezone: str
    label_r: str | None
    rowidx: list | None
    recordlen: dict


@dataclass
class CalibFileInfo:
    """
    Notes
    -----
    Store miscellaneous information.

    1. fname: .cwa or .h5 file name (CALIBRATION FILES)
    2. fs: sampliing frequency (Hz)
    3. timezone: same as SubjectInfo.timezone
    """
    fname: list
    fs: int
    timezone: str


class CalibProcess:
    """
    An object that saves estimated measurement related errors.

    Inherited by classes: Ax6Calib, apdmCalib, MovesenseCalib
    """
    def __str__(self):
        return self._name

    def __repr__(self):
        ret = f"{self._name}("
        for k in self._kw:
            ret += f"{k}={self._kw[k]!r}, "
        if ret[-1] != "(":
            ret = ret[:-2]
        ret += ")"
        return ret

    def __eq__(self, other):
        if isinstance(other, type(self)):
            return self._kw == other._kw
        return False

    def __init__(self, absolute=False, g_thresholds=None,
                 winlen=5, stdcut=0.02, **kwargs):
        """
        Parameters
        ----------
        absolute: Boolean
            Default is False; if True, misalignment level is calculated
            based on absolute deviation from 0 g.

        g_thresholds: list
            The lower and the upper limits of measured g.
            Default is None (bcz of the warning: list is a dangerous default),
            and [0.9, 1.1] will be used.

        winlen: int | list | None
            Length of a window to measure offset along each axis in seconds.
            Default is 5 for all three axes.
            If a single integer is given, it will assume the same window length for
            all three axes.

        stdcut: float
            Cutoff of the standard deviation of the values within a window.
            Default is 0.02.
        """
        self._name = self.__class__.__name__
        self._kw = kwargs
        self.absolute = absolute
        if g_thresholds is None:
            self.g_thresholds = [0.9, 1.1]
        else:
            self.g_thresholds = g_thresholds
        if winlen is None:
            self.winlen = {'x': 5, 'y': 5, 'z': 5}
        elif isinstance(winlen, list):
            if len(winlen) == 1:
                self.winlen = {key: winlen[0] for key in ['x', 'y', 'z']}
            elif len(winlen) == 3:
                self.winlen = dict(zip(['x', 'y', 'z'], winlen))
            else:
                raise ValueError("winlen should be a list of 3, or an integer")
        else:
            self.winlen = {key: winlen for key in ['x', 'y', 'z']}
        self.stdcut = stdcut

        self.info = CalibFileInfo(
            fname=['no_filename'],
            fs=None,
            timezone='')

    def find_window_both(self, arr, axis, winlen=None):
        """ This is a wrapper to use find_window with a dict obj """
        thrs = self.g_thresholds
        sns = self.find_sns(arr, thrs[0], thrs[1])
        # If axis value is not provided, or not one of 'x', 'y', or 'z',
        # raise an error
        if any((axis is None, axis.lower() not in ['x', 'y', 'z'])):
            raise ValueError("axis value incorrect - pick one: 'x', 'y', 'z'")
        # start with the initial threshold values.
        if winlen is None:
            winlen = self.winlen[axis.lower()]
        print(f"winlen, {axis.upper()}-axis: {winlen}")
        pwin, nwin = map(self.find_window,
                         [arr, arr],
                         list(sns.values()),
                         [winlen, winlen])
        if all((pwin is not None, nwin is not None)):
            # update the winlen...
            self.winlen[axis.lower()] = winlen
            return dict(zip(['p', 'n'], [pwin, nwin]))  # p: pos; n: neg

        print(f"""No window found to calculate offset with 
                  current threshold values: 
                  {thrs[0]:.2f}, {thrs[1]:.2f}""")
        thrs_low = thrs[0]
        thrs_high = thrs[1]
        while all((thrs_low > 0.75, thrs_high < 1.25)):
            thrs_low = thrs_low - 0.05
            thrs_high = thrs_high + 0.05
            print(f"""Searching windows with new thresholds:
{thrs_low:.2f}, {thrs_high:.2f}
                    """)
            sns = self.find_sns(arr, thrs_low, thrs_high)
            print(f"winlen: {winlen}")
            pwin, nwin = map(self.find_window,
                             [arr, arr],
                             list(sns.values()),
                             [winlen, winlen])
            # Should break the loop if pwin and nwin are all good
            if all((pwin is not None, nwin is not None)):
                break
        if any((pwin is None, nwin is None)):
            # If you're still not satisfied shrink the window length by 1s.
            print(f"""Windows were not found within the four attempts.
A new search begins with the reduced window length:
{winlen} - 1 = {int(winlen - 1)}
            """)
            new_winlen = int(winlen - 1)
            # If new_win is 0, stop and return an empty numpy array
            # for the missed measurement.
            if new_winlen < 1:
                if pwin is None:
                    warnings.warn(f"winlen is shorter than 1s ({winlen}s). ",
                                  f"Calibration output for {axis.lower()} axis: ",
                                  "positive will be NaNs.")
                    pwin = np.empty(0, int)
                if nwin is None:
                    warnings.warn(f"winlen is shorter than 1s ({winlen}s). ",
                                  f"Calibration output for {axis.lower()} axis: ",
                                  "negative will be NaNs.")
                    nwin = np.empty(0, int)
                return dict(zip(['p', 'n'], [pwin, nwin]))

            # Recursion
            return self.find_window_both(arr, axis, winlen=new_winlen)

        return dict(zip(['p', 'n'], [pwin, nwin]))  # positive & negative

    def find_window(self, arr, sampnums, winlen=None):
        """
        A function to find sample numbers of THE window (length = winlen).
        The window should have adjacent points, and the std of the points
        should be no greater than a cut-off for std (default: 0.02).

        Parameters
        ----------
        arr: NumPy array
             An array of measured gravitational acceleration along an axis.

        sampnums: NumPy array
             An array of sample numbers corresponding to the periods
             when one of the three (X, Y, or Z) axes was measuring +/-1g.

        winlen: int | None
            Length of the window

        Returns
        -------
        x: NumPy array | None
             An array of sample numbers. The difference of the two adjacent
             array elements will ALWAYS be one, and the standard deviation
             of the entire array will be less than a cut-off (stdcut).
             If you don't find any such array, return None.
        """
        ia = 0
        if winlen is None:
            winlen = self.winlen
        ib = self.info.fs * winlen
        # (8/7/23) ib should not be greater than the end of sampnums...
        while ib < len(sampnums):
            # Continuity should be kept!
            # 0.02 is experimental - could adjust later (6/22/23)
            if all((all(np.diff(sampnums[ia:ib]) == 1),
                    np.std(arr[sampnums[ia:ib]]) < self.stdcut)):
                break
            ia += 10
            ib += 10    # push back by 10 data points.
        if ib > len(sampnums):
            return None
        return sampnums[ia:ib]

    def find_sns(self, arr, thr_low, thr_high):
        """
        A function to find SampleNumberS (sns)

        Paramters
        ---------
        arr: NumPy array
             An array of measured gravitational acceleration along an axis.

        thr_low: float
             Lower threshold of 1g. It's expected to be self.g_thresholds[0].

        thr_high: float
             Upper threshold of 1g. It's expected to be self.g_thresholds[1].

        Returns
        -------
        x: dict
             'p' = arr sample numbers between thr_high and thr_low.
             'n' = arr sample numbers between -(thr_low) and -(thr_high).
        """
        idp = np.where(np.logical_and(arr < thr_high, arr > thr_low))[0]
        idn = np.where(np.logical_and(arr < -1*thr_low, arr > -1*thr_high))[0]
        return {'p': idp, 'n': idn}

    def get_gs(self, ndarr):
        """
        A function to calculate gain, offset, andmisalignment

        Sometimes calibration is not done properly. If so,
        NA values should be filled...

        Parameters
        ----------
        ndarr: NumPy ndarray
               Raw measurement of +/-1g along the measurement axis.

        Returns
        -------
        x: dict
               offset: measurement offset along each axis
               misalign: misalignment along each axis
               processed: +/- 1g measurements minus axis-specific offset
               gs_orig: original measurement of +/-1g along each axis
               gs: corrected measurement of +/-1g along each axis
               gs_boost: misalignment compensated gs. Not used.
               samp_num: sample numbers of the WINDOWS (return of find_windows)
        """
        absolute = self.absolute
        x_w, y_w, z_w = map(self.find_window_both,
                            [ndarr[:, 0], ndarr[:, 1], ndarr[:, 2]],
                            ['x', 'y', 'z'])
        # non-g sample numbers
        # For example, x_P(ositive)N(on)G is the concatenated regions
        # where y or z axis was measuring positive g (1 g)
        x_png = sorted(np.concatenate((y_w['p'], z_w['p'])))
        x_nng = sorted(np.concatenate((y_w['n'], z_w['n'])))
        y_png = sorted(np.concatenate((x_w['p'], z_w['p'])))
        y_nng = sorted(np.concatenate((x_w['n'], z_w['n'])))
        z_png = sorted(np.concatenate((x_w['p'], y_w['p'])))
        z_nng = sorted(np.concatenate((x_w['n'], y_w['n'])))
        x_non_g = sorted(np.concatenate((x_png, x_nng)))
        y_non_g = sorted(np.concatenate((y_png, y_nng)))
        z_non_g = sorted(np.concatenate((z_png, z_nng)))

        # Correct method of calculating offset - 6/25/23 (GEL)
        offset = list(map(np.mean,
                          [ndarr[x_non_g, 0],
                           ndarr[y_non_g, 1],
                           ndarr[z_non_g, 2]]))
        # measured g values, offset NOT removed
        gs_orig = list(map(np.mean,
                           [ndarr[x_w['n'], 0],
                            ndarr[x_w['p'], 0],
                            ndarr[y_w['n'], 1],
                            ndarr[y_w['p'], 1],
                            ndarr[z_w['n'], 2],
                            ndarr[z_w['p'], 2]]))
        # Remove offsets
        arr2 = ndarr - np.array(offset)
        # Then check for the amount of misalignment
        # Use only the positive values... doesn't matter much
        if absolute:
            xm, ym, zm = map(lambda x: np.mean(abs(x)),
                             [arr2[x_png, 0], arr2[y_png, 1], arr2[z_png, 2]])
        else:
            xm, ym, zm = map(np.mean,
                             [arr2[x_png, 0], arr2[y_png, 1], arr2[z_png, 2]])
        # measured g values, offset removed
        gs = list(map(np.mean,
                      [arr2[x_w['n'], 0],
                       arr2[x_w['p'], 0],
                       arr2[y_w['n'], 1],
                       arr2[y_w['p'], 1],
                       arr2[z_w['n'], 2],
                       arr2[z_w['p'], 2]]))
        # misalignment reflected (percentage boost)
        gs_boost = [gs[0]*(1+xm), gs[1]*(1+xm),
                    gs[2]*(1+ym), gs[3]*(1+ym),
                    gs[4]*(1+zm), gs[5]*(1+zm)]

        return {'offset': offset,
                'misalign': [xm, ym, zm],
                'processed': arr2,
                'gs_orig': gs_orig,
                'gs': gs,
                'gs_boost': gs_boost,
                'samp_num': {'x': x_w, 'y': y_w, 'z': z_w}}


class BaseProcess:
    """
    An object that saves preprocessing outcome of the raw IMU data.

    Inherited by classes including: axivity.Ax6, apdm.OpalV2
    """
    def __str__(self):
        return self._name

    def __repr__(self):
        ret = f"{self._name}("
        for k in self._kw:
            ret += f"{k}={self._kw[k]!r}, "
        if ret[-1] != "(":
            ret = ret[:-2]
        ret += ")"
        return ret

    def __eq__(self, other):
        if isinstance(other, type(self)):
            return self._kw == other._kw
        return False

    def __init__(self, **kwargs):
        """ Parameters will be inherited-class specific """
        self._name = self.__class__.__name__
        self._kw = kwargs

        # Placehodler - will be updated
        self.info = SubjectInfo(
                fname={'L': None, 'R': None},
                record_times={'L': None, 'R': None},
                fs=0,
                timezone='',
                label_r=None,
                rowidx=None,
                recordlen={'L': 0, 'R': 0})

        # Update if file names are provided?
        if 'Lfilename' in kwargs:
            self.info.fname['L'] = str(kwargs.get('Lfilename'))
        if 'Rfilename' in kwargs:
            self.info.fname['R'] = str(kwargs.get('Rfilename'))

        self.measures = Processed(
            accmags={'lmag': np.array([0, 0, 0, 0, 0]),
                     'rmag': np.array([0, 0, 0, 0, 0])},
            velmags={'lmag': np.array([0, 0, 0, 0, 0]),
                     'rmag': np.array([0, 0, 0, 0, 0])},
            thresholds={'laccth': 1,
                        'lnaccth': -1,
                        'raccth': 1,
                        'rnaccth': -1})

    def _calc_datetime(self, timestamp):
        """
        A function to convert a timestamp into a datetime instance

        Parameters
        ----------
        timestamp: int | str
             a timestamp in units of microseconds since 1970-1-1-0:00 UTC
             For Opal sensors, directly transform this to a datetime object
             For Axivity sensors, dtype is numpy.datetime64.
             If you use OmConvert, timestamp is a string.

        Returns
        -------
        datetime_utc: datetime
             a datetime instance converted from the time_stamp
        """
        if self._name in ['OpalV2', 'OpalV1', 'OpalV2Single']:
            ts = timestamp/1e6
            datetime_utc = datetime.fromtimestamp(ts, timezone.utc)
        elif self._name in ['Ax6', 'Ax6Single']:
            ts = np.array(timestamp*1e3, dtype='datetime64[ms]')
            dt_local = pd.to_datetime(ts).tz_localize(self.info.timezone)
            datetime_utc = dt_local.astimezone(pytz.utc)
        elif self._name in ['Ax6OmGui']:
            # If you use OmConvert, time stamp format is
            # %Y-%m-%d %H:%M:%S.%f
            dt_ntz = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S.%f")
            dt_local = pd.to_datetime(dt_ntz).tz_localize(self.info.timezone)
            datetime_utc = dt_local.astimezone(pytz.utc)
        return datetime_utc

    def _get_mag(self, sensors, row_idx=None, det_option='median'):
        """
        A function to calculate the magnitude of a provided dtype.

        Parameters
        ----------
        sensors: dict
             itmes are three-column arrays of acc/gyro data
        row_idx: list | None
             Recording start and end indices (each int, for OpalV2)
        det_option: str
             'median' or 'customfunc'
             Subtract median or 1(g) to detrend

        Returns
        -------
        out: dict
             detrended acceleration or angular velocity norm
             keys = ['lmag', 'rmag']
        """
        if det_option not in ['median', 'customfunc']:
            det_option = 'median'
            print("Unknown detrending option - setting it to [median]")

        # Axivity sensors differ in length.
        # Let's create another function and use map()
        # 7/28/23, Sanity check - print median
        def linalg_norm(arr, row_idx):
            nrow = arr.shape[0]

            if row_idx is None:
                row_idx = list(range(nrow))  # targeting the entire dataset
            mag = np.linalg.norm(arr[row_idx], axis=1)
            print('median: ', np.median(mag))
            return mag

        # Use np.linalg.norm... amazing!
        mags = map(linalg_norm,
                   list(sensors.values()),
                   [row_idx, row_idx])

        # MATLAB's detrend function is not used,
        #   so we can consider that the default option
        #   for detrending the data is subtracting the median
        if det_option == 'median':
            out = map(lambda x: x - np.median(x), mags)
        else:
            # detrend by subtracting 1g
            out = map(lambda x: x - np.array([9.80665]), mags)

        # if only one sensor is processed
        if len(sensors.keys())-1:
            return dict(zip(['lmag', 'rmag'], out))
        return {'umag': list(out)[0]}

    @staticmethod
    def correct_gain(arr, gains):
        """
        A function that adjusts measurements, correcting for
        the gain error.

        Parameters
        ----------
        arr: numpy.array
             raw accelerometer data
        gains: numpy.array
             [xgain, ygain, zgain]
             list of axis specific gains

        Returns
        -------
        x: array
             offset removed accelerometer data
        """
        newx = [x/gains[0] for x in arr[:, 0]]
        newy = [y/gains[1] for y in arr[:, 1]]
        newz = [z/gains[2] for z in arr[:, 2]]

        return np.column_stack((newx, newy, newz))

    @staticmethod
    def low_pass(fc, fs, arr, window='hamming'):
        """
        A function to low-pass filter the data, using the provided
        cut-off frequency (fc).
        As it's a simple first-order low-pass filter,
        limit the window choice to hamming.
        """
        h = firwin(2, cutoff=fc, window=window, fs=fs)
        return lfilter(h, 1, arr)

    @staticmethod
    def resample_to(timearr, arr, intp_option, re_fs):
        """
        Downsample the data to a resampling frequency (re_fs).
        If rsf = 20(Hz), it's the sampling frequency of
        Opal V2. Interpolation option (intp_option) is provided
        """
        assert (intp_option in ['cubic', 'linear']),\
                "intp_option should be cubic or linear"
        if intp_option == 'cubic':
            intp = CubicSpline(timearr, arr, axis=0)
        else:
            intp = interp1d(timearr, arr, axis=0)

        nt = np.round(np.arange(timearr[0], timearr[-1], 1/re_fs), 2)
        return intp(nt)

    def _calc_ind_threshold(self, maxprop):
        """
        ind_threshold = mean(peak_heights) - C * std(peak_heights)

        C = 1       Smith et al. (2015)
            0.5     Trujillo-Priego et al. (2017), ivan=True
        """
        return np.mean(maxprop['peak_heights'])\
                - np.std(maxprop['peak_heights'])

    # This is equivalent to the MATLAB script
    def _get_ind_acc_threshold(self, accmags, reject=3.2501, height=1.0):
        """
        A function to find individual thresholds

        Parameters
        ----------
        accmags: dict
             detrended acceleration norms (L/R)
        reject: float, optional
             a cut-off; values below this number will be included
        height: float, optional
             minimal height that defines a peak

        Returns
        -------
        thresholds: dict
             laccth: left positive threshold
             lnaccth: left negative threshold
             raccth: right positive threshold
             rnaccth: right negative threshold
        """
        mags2 = accmags.copy()
        posvals = map(lambda x: [u if 0 < u < reject else 0 for u in x],
                      mags2.values())
        negvals = map(lambda x: [abs(u) if -reject < u < 0 else 0 for u in x],
                      mags2.values())

        pnpks = map(lambda x: [find_peaks(u, height=height)[1] for u in x],
                    [posvals, negvals])
        accths = map(self._calc_ind_threshold, chain(*pnpks))

        if len(accmags.keys())-1:
            tkeys = ['laccth', 'raccth', 'lnaccth', 'rnaccth']
            signvec = [1, 1, -1, -1]
        else:
            tkeys = ['accth', 'naccth']
            signvec = [1, -1]
        return dict(zip(tkeys, np.multiply(signvec, list(accths))))

    def _get_ind_acc_threshold2(self, accmags, reject=3.2501, height=1.0):
        """
        A function to find individual thresholds, modified so that
        the procedure is identical to what Smith et al. (2015) describes

        Parameters and Returns are all the same to the original version
        """
        mags2 = accmags.copy()

        def thresholds_new(mag):
            # make positive values 0
            magconvt = [x if x < 0 else 0 for x in mag]
            # then rectify
            magrect = np.array([abs(y) for y in magconvt])
            loc, pks = find_peaks(mag, height=height)
            locn, pksn = find_peaks(magrect, height=height)
            hehe = loc[np.where(pks['peak_heights'] < reject)[0]]
            hehen = locn[np.where(pksn['peak_heights'] < reject)[0]]
            pos_th = np.mean(mag[hehe]) - np.std(mag[hehe])
            neg_th = np.mean(magrect[hehen]) - np.std(magrect[hehen])
            return [pos_th, neg_th]

        accths = np.concatenate(list(map(thresholds_new, mags2.values())))

        if len(accmags.keys())-1:
            tkeys = ['laccth', 'lnaccth', 'raccth', 'rnaccth']
            signvec = [1, -1, 1, -1]
        else:
            tkeys = ['accth', 'naccth']
            signvec = [1, -1]
        return dict(zip(tkeys, np.multiply(signvec, accths)))


    def _prep_row_idx(self, sensorts, in_en_dts, **kwargs):
        """
        A function to return two datetime instances that correspond to
            the start and the end of recording.

        Parameters
        ----------
        sensorts: array-like
             If preprocessing Opal data, the type is "h5py._hl.group.Group[0]
             Otherwise, it will be an array.
        in_en_dts: list
             times (in UTC) sensors were donned (don_t) and doffed (doff_t)
        btnstatus: int
             For OpalV1, if a button press is recorded, find the index
             of the first instance.

        Returns
        -------
        row_idx: list or None
             If donned and doffed times are found from the REDCap export,
             (or any other source) return a list of two indices of datapoints
             that each corresponds to the start and the end of the recording.
             If no time is found, return None.
        """
        # index where button is first pressed (True for Opal V1 only)
        if 'btnstatus' in kwargs:
            indexed1 = kwargs.get('btnstatus')
        else:
            indexed1 = 0

        if in_en_dts is not None:
            # d_in_micro is the list of TWO datetime.timedelta objects
            # The first element of this list shows the time difference
            #   between the start of the time recorded in sensors and don_t.
            # The second element is the analogous for doff_t.
            # The sampling frequency of the APDM OPAL sensor is 20Hz,
            #   so each data point is 1/20 seconds or 5e4 microseconds.
            # Therefore, if the time difference is represented in the
            #   microseconds unit and divided by 50000 (with a bit of
            #   rounding) you get how many data points don_t and doff_t
            #   are away from the index 0.
            # Consequently, two indices will be searched:
            #   1) startidx = data index that corresponds to the don_t
            #   2) endidx = data index that corresponds to the doff_t.
            #       For this one, there are occasions where the
            #       REDCap reports are 'inaccurate', meaning that the
            #       sensors were turned off long before the reported
            #       doff_t and this exception needs to be handled by
            #       simply taking the last value of the time series
            if indexed1:
                # overwrite whatever's given as the start of the recording
                # If indexed1 is 1 (True), that means you're processing
                # Opal V1 sensor data
                in_en_dts[0] = self._calc_datetime(sensorts[indexed1])
            d_in_micro = list(map(lambda x: x - self._calc_datetime(
                sensorts[0]), in_en_dts))
            convert = 1e6    # conversion constant: 1 second = 1e6 μs
            microlen = (1/self.info.fs) * convert    # duration of a data in μs
            poten = np.ceil((d_in_micro[1].days*86400*convert
                             + d_in_micro[1].seconds*convert
                             + d_in_micro[1].microseconds)/microlen)\
                                   .astype('int')
            if poten < sensorts.shape[0]:
                indices = list(map(
                    lambda x: round(
                        (x.days*86400*convert +
                            x.seconds*convert + x.microseconds)/microlen),
                    d_in_micro))
            else:
                indices = []
                indices.extend([round((d_in_micro[0].seconds*convert
                                      + d_in_micro[0].microseconds)/microlen),
                                sensorts.shape[0]-1])

            # This will be one input to self._get_mag
            row_idx = list(range(indices[0], indices[1]))
        else:
            if indexed1:
                print("No recording start and end time provided. \
                        Data starts from the first click of the button.")
                row_idx = list(range(indexed1, sensorts.shape[0]-1))
            else:
                row_idx = None
                print("No recording start and end time provided. \
                        Analysis done on the entire recording")
        return row_idx

    def _get_count(self):
        """
        A function to get counts. The count of a data point is
            the sign of the acceleration norm at the data point.
        If the angular velocity norm at the data point is less than 0,
            then the count is also 0.

        Parameters
        ----------
        None

        Returns
        -------
        acounts2: list
             count values (L/U/R)
        """
        raw_counts = map(np.sign, self.measures.accmags.values())
        acounts = [np.multiply(x, y) for x, y in
                   zip(raw_counts,
                       map(lambda x: np.greater(x, 0),
                           self.measures.velmags.values()))]
        return acounts

    def _get_cntc(self, side='L'):
        """
        A function to get counts and tcounts

        Parameters
        ----------
        None

        Returns
        -------
        tcounts: dict
             keys: count, tcount
        """
        acounts = self._get_count()    # accmags or angvels

        # Let's start with collecting all acc values over the threshold
        # The output of np.where would be a tuple - so take the first value
        # I do this to reduce the preprocessing time...
        temp = map(lambda x: np.where(x)[0],
                   [self.measures.over_accth[side],
                    self.measures.under_naccth[side]])

        # angular velocity should be taken into account...
        # let's just make sure that the detrended angvel[i] > 0
        if side == 'R':
            angvel_gt = np.nonzero(acounts[1])[0]
        else:
            angvel_gt = np.nonzero(acounts[0])[0]

        over_posth, under_negth = map(lambda x:
                                      np.intersect1d(x, angvel_gt), temp)

        def mark_tcount(over_th_arr, acounts, pos=True):
            """
            Parameters
            ----------
            over_th_arr: numpy array
                 indices of data points over a threshold
                 (ex: over_posth_l)
            acounts: dict
                 output of self._get_count(accmags, angvels)

            Returns
            -------
            t_count: numpy array
                 nonzero counts that crossed
                 a positive or negative threshold
            """
            corrsign = -1
            if pos:
                corrsign = 1
            arr_len = len(acounts)
            t_count = np.zeros(arr_len)
            for j in over_th_arr:
                # "over_th_arr" has the indices of the acceleration values
                #   that are over a threshold (left or right).
                # [j] gives one of those indices.
                # It could be that the Tcount value at the current index
                #   may have been set by the previous index
                #   (ex. [j-2] satisfied the "else" condition so Tcount[j] = 1 or -1)
                # If so, skip to the next index.
                # Also, if the next index is not zero, then the current
                #   Tcount[j] is considered as a redundant count and
                #   marked off (this is from the original MATLAB code).
                #   So we can skip such indices here.
                if all((j < (arr_len-2), all(t_count[j:j+2] != corrsign))):
                    if all(acounts[j:j+3] == corrsign):
                        t_count[j+2] = corrsign
                    else:
                        t_count[j] = corrsign

            nz_tcount = np.nonzero(t_count)[0]      # non-zero Tcounts
            nztc_len = len(nz_tcount)

            # Remove duplicates
            for i, j in enumerate(nz_tcount):
                if (i <= (nztc_len-2)) and (nz_tcount[i+1] == (j+1)):
                    t_count[j] = 0
                elif (i == (nztc_len-1)) and (nz_tcount[i-1] == (j-1)):
                    t_count[j-1] = 0

            return t_count

        if side == 'R':
            usecount = acounts[1]
        else:
            usecount = acounts[0]
        ttcounts = mark_tcount(over_posth, usecount, pos=True)
        tntcounts = mark_tcount(under_negth, usecount, pos=False)
        return {side: [usecount, ttcounts + tntcounts]}

    def get_mov(self, side='L', ttdist=8):
        """
        A function to get movement counts

        Parameters:
            side: str
                'L'eft, 'R'ight, or 'U'nilateral

        Returns:
            tmov: list
                movement counts (L/R)
        """

        assert (side in ['L', 'R', 'U']), "side should be 'L', 'R', or 'U'"

        # tcounts is a dictionary (keys: 'L', 'R')
        # Each value is a list of TWO lists (counts, tcounts)
        tcounts = self._get_cntc(side)

        # index | count | tcount | th_crossed
        arr_a = np.column_stack((np.arange(len(tcounts[side][0])),
                                 tcounts[side][0],  # counts
                                 tcounts[side][1],  # tcounts
                                 self.measures.th_crossed[side])).astype(int)
        arr_b = arr_a[np.nonzero(arr_a[:, 2])[0], :]
        # (8/21/23) Adding one more column to save diff(movstart, first_tc)
        movidx = np.zeros((arr_b.shape[0], 4), dtype=int)

        maxmov_dt = 1.5

        # arr_b[0,] would be the row with the first nonzero tcount.
        # Smith et al. (2015) has this quote:
        #   "The start of a movement was defined as simultaneous accceleration
        #    above a magnitude threshold and angular velocity greater than 0."
        # So I originally thought that the start of a movement should be
        #   a data point with tcount value 1 or -1
        # However, a typicaly movement's acceleration profile would rather be
        #   sinusoidal. Probably we could search backwards a little more and
        #   define the "start of a movement" as the data point that precedes
        #   a point with the nonzero tcount (first_tc) in time and has the same
        #   count value to that of first_tc.
        for i in range(arr_b.shape[0]-1):
            pairdiff = np.diff(arr_b[i:i+2, :], axis=0).ravel()
            # Two Tcounts are different (-1 vs. 1)
            # Rolling back to the version: Dev 29, 2022
            # Apr 26, 2023:
            #   If Ax6 (sampling rate: 25 samples/sec) is used,
            #   the difference in data points between the two opposite-sign
            #   tcounts should be 8*25/20 = 10.
            #   Also, a mov duration is no longer than 1.5 sec.
            #   OpalV2 at 20 samples/sec: 30 points
            #   Ax6 at 25 samples/sec: 38 points (round up 37.5)
            # Jul 28, 2023:
            #   Making this distance between two Tcounts adjustable >> ttdist
            #   ttdist default = 8
            tcount_diff = int(ttdist*self.info.fs/20)
            if all((pairdiff[2] != 0, pairdiff[0] <= tcount_diff)):
                sidx = arr_b[i+1, 0]     # second threshold crossing
                if arr_b[i, 3] == arr_b[i, 2]:  # if th_cross == t_count
                    first_tc = arr_b[i, 0]
                else:
                    if arr_a[arr_b[i, 0]-1, 3] == arr_b[i, 2]:
                        first_tc = arr_b[i, 0]-1
                    else:
                        first_tc = arr_b[i, 0]-2
                # Feb 9, 2023: fstep = 15 - sidx + first_tc
                # Apr 26, 2023:
                #   15 (=30/2) was increased from 8.
                #   30 is 1.5*sampling rate (20 for OpalV2)
                fstep = int(maxmov_dt*self.info.fs/2 + 1) - sidx + first_tc
                if 0 < fstep < first_tc:
                # if all((0 < fstep < first_tc,
                #    arr_a[(first_tc-1), 3] == arr_a[first_tc, 3])):
                    # diffcnt is the index of the point whose count value is
                    # different from that of first_tc, implying that
                    # the baseline is 'crossed'. This should NOT happen.
                    # Therefore, find one point behind sidx and set that as
                    # the start of a movement
                    # (1/3/22) Let's revise so that diffcnt would be
                    # the index of the FIRST point whose cross_th value is
                    # the same as that of first_tc and is within k data points
                    # from the first_tc where k = max(3, fstep)
                    # k = min(3, fstep)
                    # (2/2/23) No. Use fstep, and run another while loop
                    k = 0
                    # (8/21/23) If count, not th_crossed, is different in
                    # sign with the first tcount of a movement...
                    # while (arr_a[(first_tc-k),3] == arr_a[first_tc, 3]):
                    while (arr_a[(first_tc-k), 1] == arr_a[first_tc, 3]):
                        k += 1
                        if k > fstep:
                            break
                    # (8/1/24) finding the min value
                    # ex. 'L' -> 'lmag'
                    sidestr = side.lower() + 'mag'
                    k2 = np.argmin(abs(self.measures.accmags[sidestr][first_tc-k:first_tc]))
                    movstart = first_tc - k + k2 + 1
                    start_tc_diff = k - k2 - 1     # difference in data points
                else:
                    movstart = first_tc
                    start_tc_diff = 0

                addi = int(movstart + np.ceil(maxmov_dt*self.info.fs))
                try:
                    # movend: the first point that has the "count" value
                    # whose sign is the opposite to that of sidx's
                    # "tcount" value
                    movend = np.where(
                            arr_a[sidx:addi, 1] == -arr_a[sidx, 2])[0][0]
                    # If you move by movend from sidx and that point crossed
                    # a threshold (-1 or 1, nonzero), check one point further,
                    # and see if that point also crossed the same threshold.
                    # If not, the end of a movement should be one back.
                    # Otherwise, take that as the movend.
                    # The end of a movement should be one back
                    # (2/2/23) What if you forget about it, and just take it?
                    movidx[i] = [movstart, sidx, sidx + movend, start_tc_diff]
                except:
                    continue

        movidx_nz = movidx[np.nonzero(movidx[:, 0])[0], :]

        for i in range(1, movidx_nz.shape[0]):
            # elif condition met previously then this condition is True.
            if not any(movidx_nz[i-1, :]):
                j = i
                while j < movidx_nz.shape[0]:
                    if movidx_nz[j, 0] < movidx_nz[i-2, 2]:
                        # movidx_nz[j, :] = np.zeros(3)
                        movidx_nz[j, :] = np.zeros(4)
                        j += 1
                    else:
                        break
            # If the following movement's start index is
            # less than its precending movement's end index,
            # you should discard the current movement.
            # (8/22/23) However this should be reconsidered now...
            # Adjust a bit more... see if this current movement
            # can be modified by looking at the movstart location.
            # If movstart of the i-th movement can be pushed back
            # to first_tc, then push it back and skip.
            elif movidx_nz[i, 0] < movidx_nz[i-1, 2]:
                if movidx_nz[i, 3] > 0:
                    backstep = movidx_nz[i, 3]
                    ith_st = movidx_nz[i, 0]
                    broken = 0
                    while backstep > -1:
                        if ith_st >= movidx_nz[i-1, 2]:
                            movidx_nz[i, 0] = ith_st
                            broken = 1
                            break
                        ith_st += 1
                        backstep -= 1
                    # If while loop was not broken...
                    if not broken:
                        movidx_nz[i,] = np.zeros(4)
                else:
                    movidx_nz[i,] = np.zeros(4)
                # movidx_nz[i,] = np.zeros(3)

        movidx_nz2 = movidx_nz[np.nonzero(movidx_nz[:, 0])[0],]

        return movidx_nz2

    def acc_per_mov(self, side='L', movmat=None):
        """
        A function to calculate average acceleration per movement
            and the peak acceleration per movement

        Parameters
        ----------
        side: str
             'L' ,'R', or 'U'
        movmat: None | numpy array
             matrix that stores movements' indices

        Returns
        -------
        acc_arr: numpy ndarray
             [mov start | ave accmag for mov[i] | peak accmag for mov[i]]
        """
        assert (side in ['L', 'R', 'U']), "side should be 'L', 'R', or 'U'"

        if movmat is not None:
            movidx = movmat
        else:
            movidx = self.get_mov(side)

        if side == 'L':
            accvec = np.abs(self.measures.accmags['lmag'].copy())
        elif side == 'U':
            accvec = np.abs(self.measures.accmags['umag'].copy())
        else:
            accvec = np.abs(self.measures.accmags['rmag'].copy())

        acc_arr = np.array([[x, np.mean(accvec[x:y]), max(accvec[x:y])]
                            for x, y in zip(movidx[:, 0], movidx[:, 2]+1)])

        return acc_arr

    def plot_segment(self, time_passed, duration=20, side='L', movmat=None,
                     title=None, show=True):
        """
        A function to let user visually check movement counts

        Parameters
        ----------
        time_passed: float
             time passed from the start of the recording in seconds

        duration: int
             duration in seconds, default is 20

        side: str
             'L', 'R', or 'U'

        movmat: numpy array
            an array whose columns are movement start, midpoint, and end

        title: str
            if custom plot title is requested, use this

        Returns
        -------
        A diagnostic plot to check movement counts
        """
        assert (side in ['L', 'R', 'U']), "side should be 'L', 'R', or 'U'"

        if movmat is not None:
            movidx = movmat
        else:
            movidx = self.get_mov(side)

        if side == 'L':
            labels = ['lmag', 'laccth', 'lnaccth']
        elif side == 'U':
            labels = ['umag', 'accth', 'naccth']
        else:
            labels = ['rmag', 'raccth', 'rnaccth']

        # Jan 31, 23 / WHY did I do this? (checking rowidx is None)
        # Feb 09, 23 / I think this can be remove
        # if self.info.rowidx is not None:
            # new_t = self.info.record_times[0]\
            #         + timedelta(seconds=time_passed)
            # end_t = self.info.record_times[1]
        startidx = int(time_passed * self.info.fs)
        endidx = startidx + int(duration * self.info.fs) + 1
        mov_st = np.where(movidx[:, 0] >= startidx)[0]
        mov_fi = np.where(movidx[:, 2] <= endidx)[0]

        _, ax = plt.subplots(1)
        accline, = ax.plot(self.measures.accmags[labels[0]][startidx:endidx],
                marker='o', c='pink', label='acceleration')
        pthline = ax.axhline(y=self.measures.thresholds[labels[1]],
                c='k', linestyle='dotted', label='positive threshold')
        nthline = ax.axhline(y=self.measures.thresholds[labels[2]],
                c='k', linestyle='dashed', label='negative threshold')
        ax.axhline(y=0, c='r')  # baseline
        # If Ax6 or Active, convert from 1 deg/s to 0.017453 rad/s
        # rad_convert = 0.017453 if self._name in ['Ax6', 'Ax6Single'] else 1
        velline, = ax.plot(
                self.measures.velmags[labels[0]][startidx:endidx],
                c='deepskyblue', linestyle='dashdot', label='angular velocity')
        ax.legend(handles=[accline, pthline, nthline, velline])

        if all((mov_st.size, mov_fi.size)):
            if mov_st[0] == mov_fi[-1]:
                mov_lens = movidx[mov_st[0], 2] - movidx[mov_st[0], 0]
            else:
                fi2 = mov_fi[-1] + 1
                mov_lens = movidx[mov_st[0]:fi2, 2] - movidx[mov_st[0]:fi2, 0]
            if mov_lens.size:
                if mov_lens.size == 1:
                    hull = np.arange(movidx[mov_st[0], 0],
                                     movidx[mov_st[0], 0] + mov_lens + 1)
                    hl, = ax.plot(hull - startidx,
                            self.measures.accmags[labels[0]][hull],
                            c='g', linewidth=2, label='movement')
                    ax.legend(handles=[accline, pthline, nthline,
                                         velline, hl])
                else:
                    hull = [np.arange(x, (x+mov_lens[i]+1))
                            for i, x in enumerate(movidx[mov_st[0]:fi2, 0])]
                    hl, = ax.plot(hull[0] - startidx,
                            self.measures.accmags[labels[0]][hull[0]],
                            c='g', linewidth=2, label='movement')
                    ax.legend(handles=[accline, pthline, nthline, velline, hl])
                    # 9/6/2024 - mark the overlapping point
                    prevend = (hull[0] - startidx)[-1]
                    for j in range(1, len(hull)):
                        new_xs = hull[j] - startidx
                        ax.plot(new_xs,
                                self.measures.accmags[labels[0]][hull[j]],
                                c='g', linewidth=2)
                        if new_xs[0] == prevend:
                            ax.scatter(x=new_xs[0],
                                       y=self.measures.accmags[labels[0]][hull[j]][0],
                                       color='red',
                                       marker='s')
                        prevend = new_xs[-1]
        # (11/9/23) Shutting off this feature
        # if title is None:
        #    title = f"{duration}s from "\
                #            f"{(self.info.record_times[side][0] + timedelta(seconds=time_passed)).ctime()}"\
                #    f" UTC\n(recording ended at {self.info.record_times[side][1].ctime()} UTC)"
        # ax.set_title(title)
        ax.set_xlabel("Time (sec)")
        ax.set_ylabel("Acc. magnitude (m/s^2)")
        # x tick labels to indicate "seconds"

        def numfmt(x, pos):
            s = '{}'.format(x/self.info.fs)
            return s
        ax.xaxis.set_major_formatter(numfmt)

        if not show:
            plt.savefig(f'Segment-{int(time_passed)}_Duration_{int(duration)}_Plot.tiff',
                        dpi=600, format='tiff',
                        pil_kwargs={"compression": "tiff_lzw"})
        else:
            plt.show()


def get_axis_offsets(ground_gs):
    """
    A function to calculate axis/orientation specific biases

    Parameters
    ----------
    grounds_gs: list
        measured gravitational accelerations in the following order/
        x-, x+, y-, y+, z-, z+

    Returns
    -------
    x: list
        list of axis/orientation specific offsets
    """
    # If it is a list of TWO lists...
    refvec = np.array([-1, 1, -1, 1, -1, 1])
    if len(ground_gs) == 2:
        return list(map(lambda x: refvec - x, ground_gs))
    else:
        return refvec - ground_gs


def get_axis_offsets_v2(ground_gs):
    """ ground_gs are offset_removed gs """

    def calc_error(arr, idxlist):
        """ Return: a list of three gain values """
        # idx as a list...
        errs = list()
        for idx in idxlist:
            gain = np.diff(arr[idx[0]:idx[1]])[0]/2
            #offs = arr[idx[0]]+gain  # offset will be removed separately
            #errs.extend([gain, offs])
            errs.append(gain)
        return errs

    if len(ground_gs) == 2:
        # Left and Right
        lxyz, rxyz = map(calc_error,
                         list(ground_gs.values()),
                         repeat([[0, 2], [2, 4], [4, 6]], 2))
        return np.array((lxyz, rxyz))
    else:
        lxyz = calc_error(ground_gs, [[0, 2], [2, 4], [4, 6]])
        return np.array(lxyz)


def time_asleep(movmat, recordlen, fs, t0=0, t1_user=None):
    """
    A function to approximate the time the infant was inactive

    Parameters
    ----------
    movmat: numpy array
        indices of movements
    recordlen: int
        length of the recording
        ex) info.recordlen['R'] of a class object
    fs: int
        sampling rate
    t0: int
        initial time, default is 0
    t1_user: int|None
        user defined end time, default is None

    Returns:
        time_sleep: int
            estimated inactive times for left and right
            (unit: data point)
    """
    # time_asleep calculated based on the original MATLAB code...
    # If 6000 consequent data points (5 min) show:
    #   1) angular velocity less than 0.3
    #   2) acceleration norm less than 1.001
    # then this period is considered as the baby sleeping or inactive.
    # Technically, this is a bit weird... because you would 'ignore'
    # movements with negative peaks. The second condition should be applied
    # to the absolute values of the detrended acc magnitudes.

    # For Axivity sensor recording, there are just too many data points.
    # This approach would not work.

    # Another method to calculate [time asleep] is introduced in
    # Trujillo-Priego et al. (2017) - less than 3 movements in 5 min.
    # An approximation of this method could be:
    #
    #   a'  a  1   bc 2           d   3           4
    #   -----[-+-]--[-+--]---------[--+-]------[--+-]-
    #
    # Let's suppose that 1,2,3,4 are the movement counts (+).
    # Their starts and ends are marked by [ and ].
    # First, you define an anchor and a target.
    # The anchor and the target are the indices of movements separated by
    # no more than two movements.
    # From the image above, the first anchor is 1 and the first target is 2.
    # You then calculate the distance (D):
    #   one point before the start of a target (b) - start of an anchor (a)
    # If the anchor is 1, then instead of a, use a'.

    # i. If D > 6000, you increase [time asleep] by D
    #    Then the new anchor will be the previous target (2), and the new
    #    target will be target + 1 (3).
    #    D is newly defined: d-c (refer to the image above)
    #
    # ii. If FALSE, increase the target by 1 and check if the new distance
    #     b' - a' is greater than 6000 (refer to the imave below).
    #     If that's TRUE, do i. and move on.
    #     If that's FALSE, increase the target one more time do the same.
    #     (D = b'' - a'; D > 6000?)
    #
    #   a'  a  1   b  2           b'  3       b'' 4
    #   -----[-+-]--[-+--]---------[--+-]------[--+-]-
    #
    #     If you still see that D = b'' - a' < 6000, increase the anchor
    #     by 1, and repeat ii.
    #     If D > 6000, do i and move on.

    anchor = 0
    target = 1
    time_sleep = 0
    mvcnt = movmat.shape[0]
    len_5_min = 5*60*fs  # 6000 for fs=20 samples/sec

    while target < (mvcnt-1):
        if target-anchor > 3:
            anchor += 1
            t0 = movmat[anchor][0]

        t1 = movmat[target][0]-1

        if t1-t0 > len_5_min:
            time_sleep += t1-t0
            anchor = target
            t0 = movmat[anchor][0]
            target += 1
        else:
            target += 1
    # When you break out from the loop, make t0 the end of the last move
    t0 = movmat[-1][2] + 1
    if t1_user is not None:
        t1 = t1_user-1
    else:
        t1 = recordlen-1
    if t1-t0 > len_5_min:
        time_sleep += t1-t0

    return time_sleep

def cycle_filt(movmat, fs=20, thratio=0.2):
    """
    A function to detect and filter out movements look highly cyclical

    Parameters
    ----------
    movmat: numpy array
        an array of movement indices
    fs: int
        Sampling frequency (Hz)
    thratio: float
        The difference between two adjacent movements in seconds.
        Default is 0.2

    Returns
    -------
    movmat_del: numpy array
        an array of movement indices / cyclical movements rejected
    """
    threshold = int(fs*thratio)
    to_del = []
    i = 0
    while i < (movmat.shape[0]-2):
        diff = movmat[i+1, 0] - movmat[i, 2]
        if diff <= threshold:
            j = i+1   # j could be the last mov idx
            while j < (movmat.shape[0]-1):
                if (movmat[j+1, 0]-movmat[j, 2]) <= threshold:
                    j += 1
                else:
                    break
            # If more than 8 'cycles' are observed, discard the movements
            # (should it be ten or more?)
            # Discussion with Dr. Smith (2/13/23) -> increasing the number
            # testing - doubling the number to 16? -> 40
            if j-i > 39:
                to_del.extend(range(i, j+1))
                # further remove 2 movements within 12 seconds of j'th movement
                # Why two? just...
                k = j+1
                counter = 0
                while all((k < (movmat.shape[0]-1), counter < 3)):
                    if (movmat[k, 0]-movmat[j, 2]) < 12*fs:
                        to_del.append(k)
                        counter += 1
                        k += 1
                    else:
                        break
                i = k
            else:
                i = j+1
        else:
            i += 1
    movmat_del = np.delete(movmat, to_del, 0)
    return movmat_del

def rate_calc(lmovmat, rmovmat, recordlen, fs, thratio=0.2):
    """
    A function to calculate movement rate and return
    other relevant measures, such as sleep time

    Parameters
    ----------
    lmovmat, rmovmat: numpy array
        movement matrices for the left/right sides
    recordlen: int
        length of a recording; use '.info.recordlen.values()'
    fs: int
        Sampling frequency; if the movement matrices are downsampled,
        put in the frequency used for resampling.
    thratio: float
        a value to calculate the threshold value in cycle_filt function.
        difference between the end of one movement and the start of
        the next movement
        Default is 0.2 (seconds) -> 4 samples for 20 S/s, 5 samples for 25 S/s

    Returns
    -------
    x: dict
        lrate, rrate: movement rates
        lrec_hr, rrec_hr: recording length in hours
        lsleep_hr, rsleep_hr: sleep hours
    """
    lmovs_del, rmovs_del = map(cycle_filt, [lmovmat, rmovmat],
                               repeat(fs, 2),
                               repeat(thratio, 2))
    lsleep, rsleep = map(time_asleep, [lmovs_del, rmovs_del],
                         recordlen, repeat(fs, 2))
    lsleep_5m, rsleep_5m = map(lambda x: x/(60*fs) - np.mod(x/(60*fs), 5),
                               [lsleep, rsleep])
    lrec_hr = recordlen[0]/(3600*fs)
    rrec_hr = recordlen[1]/(3600*fs)
    lsleep_hr, rsleep_hr = map(lambda x: x/60, [lsleep_5m, rsleep_5m])
    lmovrate = lmovs_del.shape[0]/(lrec_hr-lsleep_hr)
    rmovrate = rmovs_del.shape[0]/(rrec_hr-rsleep_hr)
    return {'lrate': lmovrate, 'rrate': rmovrate,
            'lrec_hr': lrec_hr, 'rrec_hr': rrec_hr,
            'lsleep_hr': lsleep_hr, 'rsleep_hr': rsleep_hr}


def local_to_utc(timelist, study_tz):
    """
    A function to make a datetime object with its time in UTC

    Parameters
    ----------
    timelist: list
        In the following order:
        [Year, month, day, hour, minute, second, microsecond]
        If the list length is less than 5, it will inform the
        user to provide the argument in the proper format.
        When second and microsecond are not given, they are zeros.

    study_tz: str
        Timezone where the data collection happened
        ex. 'US/Pacific', 'America/Los_Angeles'

    Returns
    -------
    local_dt: datetime
        A datetime object with its timezone set to UTC
    """
    warning_msg = """You need to provide at least five numbers:
    Year (ex. 2023)
    Month (ex. 7)
    Day (ex. 12)
    Hour (ex. 18)
    Minute (ex. 29)"""
    assert len(timelist) >= 5, warning_msg
    local = pytz.timezone(study_tz)
    naive = datetime(*timelist)
    local_dt = local.localize(naive, is_dst=None)
    return local_dt.astimezone(pytz.utc)


def convert_to_utc(datetime_obj, site):
    local = pytz.timezone(site)
    local_dt = local.localize(datetime_obj, is_dst=None)
    utc_dt = local_dt.astimezone(pytz.utc)
    return utc_dt

def calc_entropy(arr, fuzzy=False, m=2, r=None):
    """
    A function to calculate either Sample Entropy (SampEn)
    or Fuzzy Entropy (FuzzEn) of a given array.
    This uses basic features of EntropyHub module's SampEn and
    FuzzEn functions. Regarding FuzzEn, if you're interested in
    exploring more, using EntropyHub.FuzzEn directly is recommended.
    For more detail, please check
    github.com/MattWillFlood/EntropyHub/blob/main/EntropyHub%20Guide.pdf

    Parameters
    ----------
        arr: list or NumPy.array
            the time series whose entropy value will be calculated
        fuzzy: boolean
            if True, then calculate FuzzEn; default is False
        m: int
            the size of the embedding dimension; default is 2
        r: float | tuple | None
            the radius of the neighbourhood / default is None
            This will make use of the default options of SampEn and FuzzEn
            from EntropyHub (r=0.2*SD(arr) for SampEn, r=(0.2, 2) for FuzzEn).

    Returns
    -------
        ent: float
            value of SampEn or FuzzEn
    """
    if not fuzzy:
        ent = SampEn(arr, m=m, r=r)
    else:
        if r is None:
            ent = FuzzEn(arr, m=m)
        else:
            ent = FuzzEn(arr, m=m, r=r)

    return ent


def make_start_end_datetime(redcap_csv, filename, site):
    """
    A function to make two datetime objects based on the
        entries from the REDCap export

    Parameters
    ----------
    redcap_csv: pandas DataFrame
    filename: string
        format: '/Some path/YYYYmmdd-HHMMSS_[identifier].h5'
    site: string
        where the data were collected (ex. America/Guatemala)

    Returns
    -------
    utc_don_doff: list
        site-specific don/doff times converted to UTC time
    """

    # Use the .h5 filename -> search times sensors were donned and doffed
    # A redcap_csv will be in the following format (example below)
    #
    # id    |             filename              | don_t | doff_t
    # ----------------------------------------------------------
    # LL001 | 20201223-083057_LL001_M1_Day_1.h5 | 9:20  | 17:30
    #
    # Split the filename with '/' and take the first (path_removed).
    # Then we concatenate them with the time searched from the REDCap
    #   export file to generate don_dt and doff_dt (dt: datetime object)

    path_removed = filename.split('/')[-1]

    def match_any(string, filename):
        """
        Highly likely, entries of the column: filename in redcap_csv
            could be erroneous. Therefore, to be on the safe side,
            split REDCap entry and check if more than half the splitted items
            are included in the filename.
        """
        result = False
        if isinstance(string, str):
            splitted = re.split('[-:_.]', string)
            lowered = list(map(lambda x: x.lower(), splitted))
            result = np.mean([x in filename.lower() for x in lowered]) > 0.5

        return result

    def convert_to_utc(datetime_obj, site):
        local = pytz.timezone(site)
        local_dt = local.localize(datetime_obj, is_dst=None)
        utc_dt = local_dt.astimezone(pytz.utc)
        return utc_dt

    idx = redcap_csv.filename.apply(lambda x: match_any(x, filename))

    don_n_doff = redcap_csv.loc[np.where(idx)[0][0], ['don_t', 'doff_t']]
    # don_n_doff = times.values[0]  # times is a Pandas Series

    # 2/10/23, don_n_doff could be NaN, meaning that people forgot to
    # enter times to the REDCap. NaN is "float".
    # If that's the case, return None
    if isinstance(don_n_doff[0], float):
        utc_don_doff = None
    else:
        temp = path_removed.split('-')[0]
        don_h, _ = don_n_doff[0].split(":")
        doff_h, _ = don_n_doff[1].split(":")

        don_dt = datetime.strptime('-'.join([temp, don_n_doff[0]]),
                                   '%Y%m%d-%H:%M') + timedelta(minutes=1)
        doff_temp = datetime.strptime('-'.join([temp, don_n_doff[1]]),
                                      '%Y%m%d-%H:%M')
    # There are cases where the REDCap values are spurious at best.
    # Sometimes the first time point recorded in sensors could be later
    #   than what's listed as the time_donned in the REDCap file by
    #   few microseconds.
    # This would case the problem later, as the method of the Subject class
    #   [_prep_row_idx] will 'assume' that the opposite is always true,
    #   and calculate: REDCap time_donned - sensor initial time.
    # Of course this will be a negative value, ruining everything; so add
    #   a minute to what's reported on the REDCap time donned.
    # If the time sensors were doffed was early in the morning (ex. 2 AM)
    #   you know that a day has passed.
    # The condition below, however, may not catch every possible exception.
    # Let's hope for the best that everyone removed the sensors
    #   before 2 PM the next day.

    nextday = any((all((int(doff_h) < 14,
                        abs(int(don_h) - int(doff_h)) < 10)), int(doff_h) < 12))

    if nextday:
        doff_dt = doff_temp + timedelta(days=1)
    else:
        doff_dt = doff_temp

    # site-specific don/doff times are converted to UTC time
    utc_don_doff = list(map(lambda lst: convert_to_utc(lst, site), [don_dt, doff_dt]))
    return utc_don_doff
