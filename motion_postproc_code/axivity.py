"""
Script prepared to process Ax6 sensor data by Jinseok Oh, Ph.D.

© 2023-2024 Infant Neuromotor Control Laboratory, All Rights Reserved
"""
from datetime import datetime
import os
from itertools import repeat
# import gzip   # gzip no longer needed, as tsv, not tsv.gz is read
import numpy as np
import pandas as pd
from pyarrow import csv
from skdh.io import ReadCwa
from openmovement.load import CwaData
from openmovement.process import OmConvert
from base import (BaseProcess, CalibProcess, cycle_filt, time_asleep,
                  get_axis_offsets, get_axis_offsets_v2, rate_calc,
                  local_to_utc, calc_entropy)
from cwa_metadata import cwa_info


class Ax6OmGui(BaseProcess):
    """ A class using openmovement's omconvert.py to analyze .cwa file
    """
    def __init__(self, dirpath, Lfilename, Rfilename, study_tz,
                 det_option='median', in_en_dts=None, **kwargs):
        """
        Parameters
        ----------
        dirpath: str
            path to the directory where the cwa files to be processed
            are stored
        Lfilename: str
            name of the [L]eft limb movement recording file
            (ex. "CHNIH0014_V02.cwa")
        Rfilename: str
            name of the [R]ight limb movement recording file
        study_tz: str
            timezone of the study site (ex. "US/Pacific")
        det_option: str
            subtracting magnitude 'median' or 1 g ('customfunc')
        in_en_dts: list or None
            list of timestamps marking the start and the end of the
            dataset to be processed. If None, entire dataset is processed.
        **kwargs
        intp_option: str
            'none', 'linear' or 'cubic' (spline)
        re_fs: int
            target [re]sampling frequency
        fs: int
            sampling frequency - please specify if it's other than 25Hz
        filter: boolean
            apply filter (True) or not (False)
        fc: int
            cut-off frequency for low-pass filtering
        """
        super().__init__()
        options = {}    # for omConvert's execution
        # Do this FIRST: fs and study_tz update
        # If resampling is requested, this should be prioritized.
        if 're_fs' in kwargs:
            options['resample'] = kwargs.get('re_fs')
            self.info.fs = options['resample']
        elif 'fs' in kwargs:
            self.info.fs = kwargs.get('fs')
        else:
            self.info.fs = 25   # Ax6 default fs is 25Hz
        self.info.timezone = study_tz
        # Now you need to specify where the OmGUIdirectory is...
        csvs_in_dir = [x for x in os.listdir(dirpath) if x.endswith('csv')]
        llcsvname = os.path.splitext(Lfilename)[0] + '.csv'
        rrcsvname = os.path.splitext(Rfilename)[0] + '.csv'
        # need to think about how to make this part simple...
        # (8/30/23) Check if binary could be "pre-built" and distributed
        # Otherwise, the solution is just to add it to the path...
        # options['executable'] = '/Users/joh/Downloads/HBCD/omconvert'

        om = OmConvert()
        if llcsvname not in csvs_in_dir:
            print("No preprocessed csv file exists: Left")
            options['csv_file'] = os.path.join(dirpath, llcsvname)
            print("omconvert will convert the cwa file.")
            LL = om.execute(os.path.join(dirpath, Lfilename), options)
        if rrcsvname not in csvs_in_dir:
            print("No preprocessed csv file exists: Right")
            options['csv_file'] = os.path.join(dirpath, rrcsvname)
            print("omconvert will convert the cwa file.")
            RR = om.execute(os.path.join(dirpath, Rfilename), options)
        # Once you're done, you need to load the csv files
        print("Reading the preprocessed csv file: Left")
        larr = pd.read_csv(os.path.join(dirpath, llcsvname))
        print("Reading the preprocessed csv file: Right")
        rarr = pd.read_csv(os.path.join(dirpath, rrcsvname))
        accs = [larr.iloc[:, [1, 2, 3]].values, rarr.iloc[:, [1, 2, 3]].values]
        gyro = [larr.iloc[:, [4, 5, 6]].values, rarr.iloc[:, [4, 5, 6]].values]
        rowidx = self._prep_row_idx(larr.iloc[:, 0], in_en_dts)
        accmags = self._get_mag(
                {x: 9.80665*y for x, y in zip(['L', 'R'], accs)},
                rowidx,
                det_option=det_option)
        velmags = self._get_mag(
                {'L': gyro[0], 'R': gyro[1]},
                rowidx)
        # convert here? deg to radian
        velmags['lmag'] = 0.017453*velmags['lmag']
        velmags['rmag'] = 0.017453*velmags['rmag']

        if 'filter' in kwargs:
            try:
                accmags_f = {
                        # cut-off (fc), hamming window
                        # first-order FIR low-pass
                        'lmag': self.low_pass(kwargs.get('fc'),
                                              kwargs.get('fs'),
                                              accmags['lmag']),
                        'rmag': self.low_pass(kwargs.get('fc'),
                                              kwargs.get('fs'),
                                              accmags['rmag'])}
            except ValueError:
                print("Missing: fc and fs for filtering the data")
        else:
            accmags_f = accmags

        thresholds = self._get_ind_acc_threshold2(accmags_f)

        # If in_en_dts was not None, rts should be curtailed.
        if rowidx is not None:
            istart = rowidx[0]
            iend = rowidx[-1]
        else:
            istart = 0
            iend = -1
        rts = {'L': list(map(datetime.strptime,
                             [larr.iloc[istart, 0], larr.iloc[iend, 0]],
                             list(repeat("%Y-%m-%d %H:%M:%S.%f", 2)))),
               'R': list(map(datetime.strptime,
                             [rarr.iloc[istart, 0], rarr.iloc[iend, 0]],
                             list(repeat("%Y-%m-%d %H:%M:%S.%f", 2))))}

        # update relevant information
        self.info.fname = [Lfilename, Rfilename]
        self.info.record_times = rts
        self.info.rowidx = rowidx
        self.info.recordlen = {'L': accmags_f['lmag'].shape[0],
                               'R': accmags_f['rmag'].shape[0]}

        self.measures.accmags = accmags_f
        self.measures.velmags = velmags
        self.measures.thresholds = thresholds
        self.measures.__post_init__()  # update fields


class Ax6(BaseProcess):
    """
    This is a class that contains preprocessed info of cwa files.
    """
    # (7/18/24) Removing `study_tz` from the parameters
    def __init__(self, Lfilename, Rfilename, fs, det_option='median',
                 in_en_dts=None, offset=None, gs=None, **kwargs):
        """
        Parameters
        ----------
        fs: int
            sampling frequency
        det_option: str
            subtracting magnitude 'median' or 1 g ('customfunc')
        in_en_dts: list or None
            list of timestamps marking the start and the end of the
            dataset to be processed. If None, entire dataset is processed.
        offset: dict(list)
            a dictionary whose items are left and right
            sensors' axis offset error amounts
        gs: list
            list of offset-removed g measures (left/right sensor)
        **kwargs
        intp_option: str
            'none', 'linear' or 'cubic' (spline)
        re_fs: int
            target [re]sampling frequency
        apply_filter: boolean
            apply filter (True) or not (False)
        fc: int
            cut-off frequency for low-pass filtering
        timezone: string
            timezone of the study site
        """
        ALLOWED_EXTENSIONS = ['cwa', 'tsv']
        super().__init__(Lfilename=Lfilename, Rfilename=Rfilename)
        # Both files should be .cwa or .tsv
        # extensions = [Lfilename.split('.')[-1], Rfilename.split('.')[-1]]
        extensions = list(map(lambda x: x.split('.')[-1],
                              self.info.fname.values()))
        # If any of the file extension is not 'cwa' or 'tsv'
        if any([x not in ALLOWED_EXTENSIONS for x in extensions]):
            raise ValueError(
                    f"Allowed extension types are: "
                    f"{ALLOWED_EXTENSIONS}."
                    f"Provided file extensions are: "
                    f"{extensions}"
                    )
        # If two file extensions are different
        if extensions[0] != extensions[1]:
            raise ValueError(
                    f"Both filenames should have the same extension, "
                    f"'.cwa' or '.tsv'. "
                    f"Provided names are {Lfilename}, "
                    f"and {Rfilename}"
                    )
        # If the extensions are the same...
        if extensions[0] == 'tsv':
            # pyarrow.csv.read_csv: takes 1/3 of time compared to np.loadtxt
            readOptions = csv.ReadOptions(column_names=[],
                                          autogenerate_column_names=True)
            parseOptions = csv.ParseOptions(delimiter='\t')
            with open(self.info.fname['L'], 'rb') as f1:
                l_tsv = np.asarray(csv.read_csv(f1,
                                                read_options=readOptions,
                                                parse_options=parseOptions))
            f1.close()
            with open(self.info.fname['R'], 'rb') as f2:
                r_tsv = np.asarray(csv.read_csv(f2,
                                                read_options=readOptions,
                                                parse_options=parseOptions))
            f2.close()
            # This is dictated by the conversion... or is it too risky?
            # elapsed_time | acc_x | acc_y | acc_z | gyro_x | gyro_y | gyro_z
            l_skdh = {'accel': l_tsv[:, 1:4],
                      'gyro': l_tsv[:, 4:7],
                      'time': l_tsv[:, 0]}
            r_skdh = {'accel': r_tsv[:, 1:4],
                      'gyro': r_tsv[:, 4:7],
                      'time': r_tsv[:, 0]}
        # If not tsv, then cwa.
        else:
            reader = ReadCwa()
            l_skdh = reader.predict(file=Lfilename)
            r_skdh = reader.predict(file=Rfilename)

        # update relevant information
        # self.info.fname = [Lfilename, Rfilename]

        # (7/18/24) Turning off this timezone...
        # Default value of self.info.timezone = ''
        # self.info.timezone = study_tz   # prioritize this step

        # Remove offset from the measurements
        if offset is not None:
            l_aligned = l_skdh['accel'] - np.array(offset['L'])
            r_aligned = r_skdh['accel'] - np.array(offset['R'])
        else:
            l_aligned = l_skdh['accel']
            r_aligned = r_skdh['accel']

        # Gain corrected...
        if gs is not None:
            offset_rm = list(map(self.correct_gain,
                                 [l_aligned, r_aligned],
                                 get_axis_offsets_v2(gs)))
        else:
            offset_rm = [l_aligned, r_aligned]

        # This is a feature originally prepared for OPAL sensor processing
        # For Axivity sensor, you use the entire recording, so rowidx = None
        # Or should we also put in the time????
        # (8/30/23) The thing is, we can implement this feature,
        # but interpolation MUST happen to do it consistently.
        # Otherwise, starting times of the two sensors are quite different,
        # and you can't have one same rowidx for both datasets (L/R)
        rowidx = None

        # (7/18/24) This is turned off - default values are None and None
        # self.info.record_times = rts
        self.info.rowidx = rowidx

        # Export offset removed files
        self.calibrated = offset_rm

        # (7/22/24) I think it's really not necessary to have interpolation
        # 'separate'. Anyone interested in resampling to a frequency other
        # than 20 Hz can just do the job on their own.
        # For the HBCD project, I will just run the analysis twice, one time
        # with the original sampling frequency (25) and the second time
        # with 20.

        # resample_to(timearr, arr, intp_option, re_fs)
        # (11/9/23) Use of 'elapsed_time' was suggested to follow
        # BIDS. If so, even when working with cwa files you better
        # interpolate using the 'elapsed_time'
        l_timevec = l_skdh['time'] - l_skdh['time'][0]
        r_timevec = r_skdh['time'] - r_skdh['time'][0]
        intp_acc = map(lambda p: self.resample_to(p[0],
                                                  p[1],
                                                  'linear',
                                                  20),
                       zip([l_timevec, r_timevec],
                           offset_rm))
        # Save these...
        self.__acc20 = list(intp_acc)
        self.l_timevec = l_timevec
        self.r_timevec = r_timevec

        intp_vel = map(lambda q: self.resample_to(q[0],
                                                  q[1],
                                                  'linear',
                                                  20),
                       zip([l_timevec, r_timevec],
                           [l_skdh['gyro'], r_skdh['gyro']]))
        accmags20 = self._get_mag(
                {x: 9.80665*y for x, y in zip(['L', 'R'], self.__acc20)},
                det_option=det_option)
        velmags20 = self._get_mag(
                dict(zip(['L', 'R'], intp_vel)))
        # 25Hz samples
        # (7/25/24) Let's also resample data to 25Hz...
        if 'fs_handling' in kwargs:
            fs_handling = kwargs.get('fs_handling')
            if fs_handling == 'raw':
                accmags = self._get_mag(
                        {x: 9.80665*y for x, y in zip(['L', 'R'], offset_rm)},
                        rowidx,
                        det_option=det_option)
                velmags = self._get_mag(
                        {'L': l_skdh['gyro'], 'R': r_skdh['gyro']},
                        rowidx)
            elif fs_handling == 'corrected':
                intp_acc25 = map(lambda p: self.resample_to(p[0],
                                                            p[1],
                                                            'linear',
                                                            fs),
                                 zip([l_timevec, r_timevec],
                                     offset_rm))
                intp_vel25 = map(lambda p: self.resample_to(p[0],
                                                            p[1],
                                                            'linear',
                                                            fs),
                                 zip([l_timevec, r_timevec],
                                     [l_skdh['gyro'], r_skdh['gyro']]))
                accmags = self._get_mag(
                        {x: 9.80665*y for x, y in zip(['L', 'R'], intp_acc25)},
                        det_option=det_option)
                velmags = self._get_mag(
                        dict(zip(['L', 'R'], intp_vel25)))
            else:
                raise ValueError('fs_handling is either [raw] or [corrected]')
        else:
            accmags = self._get_mag(
                    {x: 9.80665*y for x, y in zip(['L', 'R'], offset_rm)},
                    rowidx,
                    det_option=det_option)
            velmags = self._get_mag(
                    {'L': l_skdh['gyro'], 'R': r_skdh['gyro']},
                    rowidx)

        # convert here? deg to radian
        velmags['lmag'] = 0.017453*velmags['lmag']
        velmags['rmag'] = 0.017453*velmags['rmag']

        velmags20['lmag'] = 0.017453*velmags20['lmag']
        velmags20['rmag'] = 0.017453*velmags20['rmag']

        # SAVE both cases - for an update in the future
        self.__mags = {'accmags20': accmags20,
                       'velmags20': velmags20,
                       'accmags25': accmags,
                       'velmags25': velmags}

        # Just make this a self's attribute...
        if 'apply_filter' in kwargs:
            self.apply_filter = kwargs.get('apply_filter')
        else:
            self.apply_filter = None
        if 'fc' in kwargs:
            self.fc = kwargs.get('fc')
        else:
            self.fc = None

        # if timezone is provided...
        if 'timezone' in kwargs:
            self.info.timezone = kwargs.get('timezone')

        # Ax6 can be updated using a different fs...
        self.update(new_fs=int(fs))

    def update(self, new_fs):
        accstring = 'accmags' + str(new_fs)
        velstring = 'velmags' + str(new_fs)
        if self.apply_filter is not None:
            # if you want to filter out,
            # provide fc (cut-off frequency)
            if self.fc is None:
                raise ValueError("Missing: fc for filtering the data")

            try:
                accmags_f = dict(zip(['lmag', 'rmag'],
                                     list(map(lambda p: self.low_pass(
                                         self.fc, new_fs, p),
                                              [self.__mags[accstring]['lmag'],
                                               self.__mags[accstring]['rmag']]))))
                print(accmags_f)
            except ValueError:
                print(f"fs is 20 or 25; you provided: {new_fs}")

        else:
            try:
                accmags_f = self.__mags[accstring]
            except ValueError:
                print(f"fs is 20 or 25; you provide {new_fs}")

        thresholds = self._get_ind_acc_threshold2(accmags_f)

        # (8/5/24) Originally we 'estimated' recording start and end
        # times with the timezone information provided by users.
        # We no longer use that option.
        # If raw cwa files are analyzed, just use 'cwa_metadata.py' to
        # read the metadata. If tsv files are analyzed, check '*motion.json'

        # rts = {'L': list(map(self._calc_datetime,
        #                      [l_skdh['time'][0], l_skdh['time'][-1]])),
        #        'R': list(map(self._calc_datetime,
        #                      [r_skdh['time'][0], r_skdh['time'][-1]]))}

        self.info.fs = new_fs   # Ax6 default fs is 25Hz
        self.info.recordlen = {'L': accmags_f['lmag'].shape[0],
                               'R': accmags_f['rmag'].shape[0]}

        self.measures.accmags = accmags_f
        # This should be safe...
        self.measures.velmags = self.__mags[velstring]
        self.measures.thresholds = thresholds
        self.measures.__post_init__()  # update fields


class Ax6Calib(CalibProcess):
    """
    This is a class that takes calibration cwa file(s) and stores
    offset values.
    """
    def __init__(self, calib1, fs, absolute=False, **kwargs):
        super().__init__()  # thresholds = [0.9, 1,1]
        if str(calib1).endswith('tsv'):
            temp = np.loadtxt(calib1, delimiter='\t')
        else:
            with CwaData(calib1) as cwafile:
                temp = cwafile.get_sample_values()
        ss_l = temp[:, [1, 2, 3]]
        self.absolute = absolute
        self.info.fs = int(fs)
        if 'stdcut' in kwargs:
            self.stdcut = kwargs.get('stdcut')
        if 'winlen' in kwargs:
            self.winlen = kwargs.get('winlen')

        calib1_vals = self.get_gs(ss_l)

        if 'calib2' in kwargs:
            print("""TWO (2) calibration files provided - the first file will be
            set as the left sensor calibration file""")
            calib2 = kwargs.get('calib2')
            self.info.fname = [str(calib1), str(calib2)]
            if str(calib2).endswith('tsv'):
                temp2 = np.loadtxt(calib2, delimiter='\t')
            else:
                with CwaData(calib2) as cwafile2:
                    temp2 = cwafile2.get_sample_values()
            # ss_r = reader.predict(kwargs.get('calib2'))['accel']
            ss_r = temp2[:, [1, 2, 3]]
            calib2_vals = self.get_gs(ss_r)

            self.offset = {'L': calib1_vals['offset'],
                           'R': calib2_vals['offset']}
            self.misalign = {'L': calib1_vals['misalign'],
                             'R': calib2_vals['misalign']}
            # original is the raw measurement; processed is offset-removed
            self.original = {'L': ss_l, 'R': ss_r}
            self.processed = {'L': calib1_vals['processed'],
                              'R': calib2_vals['processed']}
            self.raw_gs = {'L': calib1_vals['gs_orig'],
                           'R': calib2_vals['gs_orig']}
            self.off_gs = {'L': calib1_vals['gs'],
                           'R': calib2_vals['gs']}
            self.boost_gs = {'L': calib1_vals['gs_boost'],
                             'R': calib2_vals['gs_boost']}
            self.samp_num = {'L': calib1_vals['samp_num'],
                             'R': calib2_vals['samp_num']}
        else:
            self.info.fname = [calib1]
            self.offset = calib1_vals['offset']
            # this is just for the consistency across classes...
            self.misalign = {'L': calib1_vals['misalign']}
            self.original = ss_l
            self.processed = calib1_vals['processed']
            self.raw_gs = calib1_vals['gs_orig']
            self.off_gs = calib1_vals['gs']
            self.boost_gs = calib1_vals['gs_boost']
            self.samp_num = calib1_vals['samp_num']

class Ax6Single(BaseProcess):
    """
    Single Ax6 sensor processing
    """
    def __init__(self, filename, study_tz,
                 offset=None, gs=None, **kwargs):
        super().__init__()
        reader = ReadCwa()
        # pfizer people change the format of the arguments
        # ALL THE TIME! As of Sept. 11, 2024, it's (*, file=None, ...)
        l_skdh = reader.predict(file=filename)
        self.info.timezone = study_tz   # prioritize this step

        if offset is not None:
            l_aligned = l_skdh['accel'] - np.array(offset)
        else:
            l_aligned = l_skdh['accel']

        # get axis/orientation-wise offsets
        if gs is not None:
            offset_rm = self.correct_gain(l_aligned,
                                          get_axis_offsets_v2(gs))
        else:
            offset_rm = l_aligned

        # This is a feature originally prepared for OPAL sensor processing
        # For Axivity sensor, you use the entire recording, so rowidx = None
        # Or should we also put in the time????
        rowidx = None

        if 'intp_option' in kwargs:
            try:
                intp_acc = self.resample_to(l_skdh['time'],
                                            offset_rm,
                                            kwargs.get('intp_option'),
                                            kwargs.get('re_fs'))
                intp_vel = self.resample_to(l_skdh['time'],
                                            l_skdh['gyro'],
                                            kwargs.get('intp_option'),
                                            kwargs.get('re_fs'))
                accmags = self._get_mag(
                        {'L': 9.80665*intp_acc})
                velmags = self._get_mag(
                        {'L': intp_vel})
            except ValueError:
                print("Missing: re_fs for downsampling the data")
                # raise ValueError("Missing: re_fs for downsampling")
        else:
            accmags = self._get_mag(
                    {'L': 9.80665*offset_rm},
                    rowidx)
            velmags = self._get_mag(
                    {'L': l_skdh['gyro']},
                    rowidx)
        # convert here?
        velmags['umag'] = 0.017453*velmags['umag']

        if 'filter' in kwargs:
            try:
                accmags_f = {
                        'umag': self.low_pass(kwargs.get('fc'),
                                              kwargs.get('fs'),
                                              accmags['umag'])}
            except ValueError:
                print("Missing: fc and fs for filtering the data")
        else:
            accmags_f = accmags

        thresholds = self._get_ind_acc_threshold(accmags_f)

        rts = {'U': list(map(self._calc_datetime,
                             [l_skdh['time'][0], l_skdh['time'][-1]]))}

        # update relevant information
        self.info.fname = filename
        self.info.record_times = rts
        self.info.rowidx = rowidx
        if 'fs' in kwargs:
            self.info.fs = kwargs.get('fs')
        else:
            self.info.fs = 25   # Ax6 default fs is 25Hz
        self.info.recordlen = {'U': accmags_f['umag'].shape[0]}
        self.measures.accmags = accmags_f
        self.measures.velmags = velmags
        self.measures.thresholds = thresholds
        self.measures.__post_init__()  # update fields
