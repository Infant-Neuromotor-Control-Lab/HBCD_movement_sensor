#!/usr/local/bin/python3
"""
ax6_postproc.py does preprocessing of sensor recordings.

The script is loaded and used in run.py
"""
import re
from pathlib import Path
import json
# import gzip
import numpy as np
from scipy.stats import iqr
import pyarrow
from pyarrow import csv as pacsv
import antropy as ant
import axivity


# (7/18/24) Removing `study_tz`
def calc_stats(tsvdir, outdir, sub, ses,
               det_option='median',
               in_en_dts=None, fs=25,
               entropy_measure=None,
               entropy_type=None,
               **kwargs):

    """A function to calculate summary statistics and
    save the preprocessing output

    * may change in the future

    Parameters
    ----------
    tsvdir : str
        path to the directory where tsv files are located
        this should be an absolute path

    outdir : str
        path to the directory where the summary JSON file
        will be stored

    sub : str
        participant id; the first two characters are 'CH'

    ses : int
        session id in number

    study_tz : str (deprecated)
        a label specifying the time zone where the sensors were 'configured'
        the value should be cax6patible with one of `pytz.all_timezones`
        ex) America/Los_Angeles, US/Eastern, America/Denver

    det_option : str
        a label describing the detrending method of the signal
        default set to subtracting the 'median'

    in_en_dts : list | None
        a list of datetime objects specifying the times a recording
        started and ended. Default set to None

    fs : int
        Sampling frequency / default set to 25, fs used in HBCD study

    entropy_measure : str
        a label indicating which acceleration to calculate entropy from
        ex) avgacc (average acceleration per movement)
            pkacc (peak acceleration per movement)

    entropy_type : str
        a label indicating which entropy to calculate
        ex) SampEn or FuzzEn

    Returns
    -------
    1. a summary of recording (kinematic variables) prepared in JSON
    2. calibrated accelerometer data (X, Y, Z axes) in tsv
    3. calibrated and resampled (20 Hz) accelerometer data in tsv
    """

    # create sub-folders FIRST
    final_outdir = Path(outdir) / sub / ses / 'motion'
    # final_outdir.mkdir(parents=True, exist_ok=True)

    tsvfiles = [str(x) for x in tsvdir.glob('sub*') if str(x).endswith('tsv')]
    # print(f'tsvfiles: {tsvfiles}')
    # We need FOUR tsv files: {L, R} x {PRECAL, RECORDING}
    if not tsvfiles:
        raise FileNotFoundError("There is no tsv file")
    if len(tsvfiles) < 4:
        # Four files should be:
        # sub-{sub}_ses-{ses}_task-LeftLegMovement_tracksys-imu_acq-primary_motion.tsv
        # sub-{sub}_ses-{ses}_task-LeftLegMovement_tracksys-imu_acq-calibration_motion.tsv
        # sub-{sub}_ses-{ses}_task-RightLegMovement_tracksys-imu_acq-primary_motion.tsv
        # sub-{sub}_ses-{ses}_task-RightLegMovement_tracksys-imu_acq-calibration_motion.tsv
        expected_fnames = [f'sub-{sub}_ses-{ses}_task-LeftLegMovement_tracksys-imu_acq-primary_motion.tsv',
                           f'sub-{sub}_ses-{ses}_task-LeftLegMovement_tracksys-imu_acq-calibration_motion.tsv',
                           f'sub-{sub}_ses-{ses}_task-RightLegMovement_tracksys-imu_acq-primary_motion.tsv',
                           f'sub-{sub}_ses-{ses}_task-RightLegMovement_tracksys-imu_acq-calibration_motion.tsv']
        for tsv in tsvfiles:
            expected_fnames.remove(tsv)
        raise FileNotFoundError(f'File(s) Not Found: {expected_fnames}')

    # No else after raise
    for i, tsv in enumerate(tsvfiles):
        # identifier = str(tsv).rsplit('_acq-', maxsplit=1)[-1]
        print(f"tsv {i+1}: {tsv.split('/')[-1]}")
        splits = re.split('_acq-', re.split('_task-', str(tsv))[1])
        # print(f"splits: {splits}")
        identifier = ''.join((splits[0].rsplit('tracksys-imu')[0],
                              splits[1].rsplit('.tsv')[0]))
        print(f'|->> {identifier}')
        print('------------------------------')
        # match / case in Python 3.10
        match identifier:
            case "LeftLegMovement_primary_motion":
                LeftRecordingFile = tsv
            case "RightLegMovement_primary_motion":
                RightRecordingFile = tsv
            case "LeftLegMovement_calibration_motion":
                LeftCalibrationFile = tsv
            case "RightLegMovement_calibration_motion":
                RightCalibrationFile = tsv

    print('+----------------------------------------+')
    print('+   4 [motion] files should be present   +')
    print('+     (lfile, rfile, lcalib, rcalib)     +')
    print('+----------------------------------------+')
    print(f"Recording file Left : {LeftRecordingFile.split('/')[-1]}")
    print(f"Recording file Right : {RightRecordingFile.split('/')[-1]}")
    print(f"Calibration file Left : {LeftCalibrationFile.split('/')[-1]}")
    print(f"Calibration file Right : {RightCalibrationFile.split('/')[-1]}\n")

    # Calibration output prepared
    print('+----------------------------+')
    print('+  Offset estimation begins  +')
    print('+----------------------------+')
    calib = axivity.Ax6Calib(LeftCalibrationFile,
                             calib2=RightCalibrationFile,
                             fs=fs,)
    print(calib.off_gs)
    print('------------------------------')

    # Creates an 'Ax6' class instance, `ax6proc`.
    # Assuming that the sampling frequency is 25.
    ax6proc = axivity.Ax6(LeftRecordingFile, RightRecordingFile,
                          fs=fs,
                          offset=calib.offset,
                          gs=calib.off_gs,
                          det_option=det_option,
                          in_en_dts=in_en_dts,
                          **kwargs)

    # (7/22/24) write some preprocessed files...
    # 1) calibrated accelerometer files
    g_val = 9.80665

    # Options for writing tsv files (accelerations)
    write_option = pacsv.WriteOptions(include_header=True, delimiter='\t')
    left_column_names = ["imu_latency", "LeftAnkle_Accel_x",
                         "LeftAnkle_Accel_y", "LeftAnkle_Accel_z"]
    right_column_names = ["imu_latency", "RightAnkle_Accel_x",
                          "RightAnkle_Accel_y", "RightAnkle_Accel_z"]

    # Do this if you're asked to adjust the interval
    if kwargs.get('fs_handling') == 'corrected':
        leftleg_calib_tsv = final_outdir / ('_'.join((sub, ses, 'leg-left',
                                                      'desc-calibrated',
                                                      f'recording-{fs}',
                                                      'motion'))+'.tsv')
        rightleg_calib_tsv = final_outdir / ('_'.join((sub, ses, 'leg-right',
                                                       'desc-calibrated',
                                                       f'recording-{fs}',
                                                       'motion'))+'.tsv')
        # No resampling done; use ax6proc.l_timevec and ax6proc.r_timevec
        # Remember that l_timevec and r_timevec are each 'elapsed_time' of
        # the left and the right sensor, respectively.
        # The interval between the two adjacent points would be 'approximately'
        # 0.04 seconds, but never guaranteed.
        left_pa_table = pyarrow.table([ax6proc.l_timevec,
                                       ax6proc.calibrated[0][:, 0] * g_val,
                                       ax6proc.calibrated[0][:, 1] * g_val,
                                       ax6proc.calibrated[0][:, 2] * g_val],
                                      names=left_column_names)
        right_pa_table = pyarrow.table([ax6proc.r_timevec,
                                        ax6proc.calibrated[1][:, 0] * g_val,
                                        ax6proc.calibrated[1][:, 1] * g_val,
                                        ax6proc.calibrated[1][:, 2] * g_val],
                                       names=right_column_names)
        pacsv.write_csv(left_pa_table, leftleg_calib_tsv,
                        write_options=write_option)
        pacsv.write_csv(right_pa_table, rightleg_calib_tsv,
                        write_options=write_option)

        print('+-------------------------------------------------------------------------+')
        print('+  Calibrated accelerometer data exported:                                +')
        print(f'+   {sub}_{ses}_leg-left_desc-calibrated_recording-{fs}_motion.tsv   +')
        print(f'+   {sub}_{ses}_leg_right_desc-calibrated_recording-{fs}_motion.tsv  +')
        print('+-------------------------------------------------------------------------+')

        del left_pa_table, right_pa_table

    # calibrated + resampled accelerometer files
    leftleg_calib_re20_tsv = final_outdir / ('_'.join((sub, ses, 'leg-left',
                                                       'desc-calibrated',
                                                       'recording-20',
                                                       'motion'))+'.tsv')
    rightleg_calib_re20_tsv = final_outdir / ('_'.join((sub, ses, 'leg-right',
                                                        'desc-calibrated',
                                                        'recording-20',
                                                        'motion'))+'.tsv')
    # place the 'elapsed time' vector... 0.05 = 1/20
    # Error reported (10/2/2024) - floating point error
    # ex. >> np.shape(ax6.proc_Ax6__acc20[0]) = 14742317
    #     >> 14743217 * 0.05 = 737115.8500000001
    #     >> np.arange(0, 737115.8500000001, 0.05) will yield 14742318,
    #     one sample more than the original number of samples,
    #     because the range includes 737115.85 - that's not what's wanted.
    #     This will cause an error, because pyarrow.table will expect
    #     incoming data to have 14742318 rows, but will receive
    #     14742317 rows.
    # Rounding up the number (ex. np.round(14743217 * 0.05, 3)) would
    # address this error.

    left_nrow = np.round(np.shape(ax6proc._Ax6__acc20[0])[0]*0.05, 3)
    right_nrow = np.round(np.shape(ax6proc._Ax6__acc20[1])[0]*0.05, 3)
    time_l_20 = np.arange(0, left_nrow, 0.05)
    time_r_20 = np.arange(0, right_nrow, 0.05)
    left_pa_table_re20 = pyarrow.table([time_l_20,
                                        ax6proc._Ax6__acc20[0][:, 0] * g_val,
                                        ax6proc._Ax6__acc20[0][:, 1] * g_val,
                                        ax6proc._Ax6__acc20[0][:, 2] * g_val],
                                       names=left_column_names)
    right_pa_table_re20 = pyarrow.table([time_r_20,
                                         ax6proc._Ax6__acc20[1][:, 0] * g_val,
                                         ax6proc._Ax6__acc20[1][:, 1] * g_val,
                                         ax6proc._Ax6__acc20[1][:, 2] * g_val],
                                        names=right_column_names)
    pacsv.write_csv(left_pa_table_re20, leftleg_calib_re20_tsv,
                    write_options=write_option)
    pacsv.write_csv(right_pa_table_re20, rightleg_calib_re20_tsv,
                    write_options=write_option)

    print('+-------------------------------------------------------------------------+')
    print('+  Calibrated, resampled accelerometer data exported:                     +')
    print(f'+   {sub}_{ses}_leg-left_desc-calibrated_recording-20_motion.tsv   +')
    print(f'+   {sub}_{ses}_leg-right_desc-calibrated_recording-20_motion.tsv  +')
    print('+-------------------------------------------------------------------------+')

    print('------------------------------')
    print('Preprocessing Completed.\nKinematic variables are going to be calculated')
    print('------------------------------')

    # Erase variables to free up some space (Don't think that useful but...)
    del left_pa_table_re20, right_pa_table_re20

    def calc_kinematics(ax6obj):
        fs = ax6obj.info.fs
        print(f'Data [re]sampled at {fs} Hz')
        print('------------------------------')

        ax6_mov = ax6obj.get_mov()
        ax6_movr = ax6obj.get_mov('R')

        ax6_mov_filt = axivity.cycle_filt(ax6_mov, fs)
        ax6_movr_filt = axivity.cycle_filt(ax6_movr, fs)

        rates = axivity.rate_calc(ax6_mov, ax6_movr,
                                  list(ax6proc.info.recordlen.values()), fs)

        movdur, movrdur = map(lambda x: (x[:, 2] - x[:, 0] + 1) * (1/fs),
                              [ax6_mov_filt, ax6_movr_filt])

        accs = ax6obj.acc_per_mov(movmat=ax6_mov_filt)
        accsr = ax6obj.acc_per_mov(movmat=ax6_movr_filt, side='R')

        return {'ax6_mov_filt': ax6_mov_filt,
                'ax6_movr_filt': ax6_movr_filt,
                'rates': rates,
                'movdur': movdur,
                'movrdur': movrdur,
                'accs': accs,
                'accsr': accsr}

    # Calculate kinematic variables
    kinematics = calc_kinematics(ax6proc)
    print('Calculation was successful - here goes the writing process')
    # Save a summary of variables into a .json file
    save_kinematics_summary(final_outdir, sub, ses, ax6proc, kinematics,
                            entropy_measure, entropy_type)

    # Repeat the process with the new sampling frequency: 20
    ax6proc.update(new_fs=20)
    kinematics = calc_kinematics(ax6proc)
    save_kinematics_summary(final_outdir, sub, ses, ax6proc, kinematics,
                            entropy_measure, entropy_type)


def save_kinematics_summary(final_outdir, sub, ses, ax6obj, kinematics,
                            entropy_measure, entropy_type):
    """
    A summary of Kinematic variables (ex. movement rate) will be
    written in a json file + entropy values as well!

    Parameters
    ----------
        final_outdir: pathlib.Path
            It should be {BIDS_output_dir} / {sub} / {ses} / 'motion'

        sub: str
            participant label (ex. sub-xxxxx)

        ses: str
            session label (ex. ses-V02)

        ax6obj: axivity.Ax6
            An object of Ax6 class of axivity module

        kinematics: dict
            A dictionary saving values of kinematic variables

        entropy_type : str | None
            a label indicating which entropy to calculate
            ex) SampEn or FuzzEn

        entropy_measure : str | None
            a label indicating which acceleration to calculate entropy from
            ex) avgacc (average acceleration per movement)
                pkacc (peak acceleration per movement)

    Returns
    -------
        None (saves a .json file in final_outdir / 'Kinematics')
    """
    outname_json = (final_outdir / 'Kinematics' / 
                    ('_'.join((sub, ses, f'desc-kinematics_recording-{ax6obj.info.fs}', 'motion'))+'.json'))
    avgaccel_L_arr = kinematics['accs'][:, 1]
    avgaccel_R_arr = kinematics['accsr'][:, 1]
    pkaccel_L_arr = kinematics['accs'][:, 2]
    pkaccel_R_arr = kinematics['accsr'][:, 2]

    # Trying to be efficient in handling the entropy-related
    # arguments
    entropy_keys = []
    entropies = []
    if entropy_measure is None:
        feed = [avgaccel_L_arr, pkaccel_L_arr, avgaccel_R_arr, pkaccel_R_arr]
        feed_label = ['Left_average_acc', 'Left_peak_acc',
                      'Right_average_acc', 'Right_peak_Acc']
    elif entropy_measure.lower() == 'avgacc':
        feed = [avgaccel_L_arr, avgaccel_R_arr]
        feed_label = ['Left_average_acc', 'Right_average_acc']
    elif entropy_measure.lower() == 'pkacc':
        feed = [pkaccel_L_arr, pkaccel_R_arr]
        feed_label = ['Left_peak_acc', 'Right_peak_acc']
    else:
        raise ValueError(f"entropy_measure is 'avgacc' or 'pkacc'. \
User input: {entropy_measure}")

    if entropy_type is None:
        fuzzTrue = [False, True]
        entype_label = ['SampEn', 'FuzzEn']
    elif entropy_type.lower() == 'sampen':
        fuzzTrue = [False]
        entype_label = ['SampEn']
    elif entropy_type.lower() == 'fuzzen':
        fuzzTrue = [True]
        entype_label = ['FuzzEn']
    else:
        raise ValueError(f"entropy_type is 'SampEn' or 'FuzzEn'. \
User input: {entropy_type}")

    # Entropy calculated for both the left and the right leg data
    # If both 'entropy_measure' and 'entropy_type' are None,
    # all four cells are obtained (total 8 entropies calculated).
    # If only 'entropy_measure' is None, either the first column
    # or the second column will be obtained (total 4 entropies).
    # If only 'entropy_type' is None, either the first row
    # or the second row will be obtained (total 4 entropies).
    #                +------------------------------------------------+
    #                |  Average acceleration  |   Peak acceleration   |
    # ---------------|------------------------------------------------|
    # Sample Entropy | entropy_measure:avgacc | entropy_measure:pkacc |
    #                | entropy_type:SampEn    | entropy_type:SampEn   |
    # ----------------------------------------------------------------|
    #  Fuzzy Entropy | entropy_measure:avgacc | entropy_measure:pkacc |
    #                | entropy_type:FuzzEn    | entropy_type:FuzzEn   |
    # ----------------------------------------------------------------+
    for i, fd in enumerate(feed):
        for j, ft in enumerate(fuzzTrue):
            # print(f"Looped in: feed[{i}], fuzzTrue[{j}]")
            if ft:
                out = axivity.calc_entropy(fd, fuzzy=ft)
                entropies.append(out[0][-1])
            else:
                # if not calculating FuzzEn, use 'antropy' package
                out = ant.sample_entropy(fd)
                entropies.append(out)
            temp_label = '_'.join((feed_label[i], entype_label[j]))
            entropy_keys.append(temp_label)
            print(f'{temp_label} is calculated')

    outdict = {
            "Participant_id": sub,
            "Session_id": ses,
            # "Left_sensor_id": sensorids[0],
            # "Right_sensor_id": sensorids[1],
            "Left_positive_threshold": ax6obj.measures.thresholds['laccth'],
            "Right_positive_threshold": ax6obj.measures.thresholds['raccth'],
            "Left_negative_threshold": ax6obj.measures.thresholds['lnaccth'],
            "Right_negative_threshold": ax6obj.measures.thresholds['rnaccth'],
            "Left_total_count": kinematics['ax6_mov_filt'].shape[0],
            "Right_total_count": kinematics['ax6_movr_filt'].shape[0],
            "Left_movement_rate": kinematics['rates']['lrate'],
            "Right_movement_rate": kinematics['rates']['rrate'],
            "Left_sleep_time": kinematics['rates']['lsleep_hr'],
            "Right_sleep_time": kinematics['rates']['rsleep_hr'],
            "Left_average_acc_median": np.median(avgaccel_L_arr),
            "Left_average_acc_iqr": iqr(avgaccel_L_arr),
            "Left_peak_acc_median": np.median(pkaccel_L_arr),
            "Left_peak_acc_iqr": iqr(pkaccel_L_arr),
            "Left_movement_duration_median": np.median(kinematics['movdur']),
            "Left_movement_duration_iqr": iqr(kinematics['movdur']),
            "Right_average_acc_median": np.median(avgaccel_R_arr),
            "Right_average_acc_iqr": iqr(avgaccel_R_arr),
            "Right_peak_acc_median": np.median(pkaccel_R_arr),
            "Right_peak_acc_iqr": iqr(pkaccel_R_arr),
            "Right_movement_duration_median": np.median(kinematics['movrdur']),
            "Right_movement_duration_iqr": iqr(kinematics['movrdur']),
            "entropies": dict(zip(entropy_keys, entropies)),
            }

    json_object = json.dumps(outdict, indent=4)

    # write to "outname"
    with open(outname_json, "w") as f:
        f.write(json_object)
    f.close()

    print(f"A summary of kinematic variables saved: {outname_json.name}")
    print("------------------------------")
