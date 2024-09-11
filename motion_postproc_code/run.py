#!/usr/local/bin/python3
"""
run.py will take the path to the directory where tsv files are stored,
do the preprocessing using the files and return the summary as a JSON file and
a gzipped Pickle file (a class object that can be loaded for future analysis).

The command to run run.py is:

    python3 run.py bids_dir output_dir analysis_level

Positional Arguments:

    bids_dir        The root directory of BIDS-structure.

    output_dir      The parent folder under which folders specifying
                    DCCID, session, and type of data are hierarchically placed.

                    Processed output will be stored in the following path:

                        output_dir/{sub}/{ses}/motion/

                    Following files will be stored:

                        *_motion.json : summary of sensor recordings. To learn
                                        more about each variable, see
                                        https://doi.org/10.3390/s150819006
                        *_pkl.gz : gzipped pickle file; can be loaded in a
                                   Python script for further manipulation

    analysis_level  always 'participant' (could be modified later...)

Optional Arguments:
    -h, --help                  show this help message and exit

    --participant_label, --participant-label    a specific participant data that need to be processed

    --session_id, --session-id                  a specific session data that need to be processed

    --interval                  'raw' or 'corrected'; this is to address the uneven sampling interval

    --pa_measure, --pa-measure  'acceleration' or 'jerk'; used for pa_calc_mighty_tot.py

    --pa_side, --pa-side        'left' or 'right'; used for pa_calc_mighty_tot.py

    --entropy_type, --entropy-type              a specific entropy type to be calculated ('SampEn' or 'FuzzEn')

    --entropy_measure, --entropy-measure        which measure to calculate an entropy ('avgacc' or 'pkacc')

    --stop_on_error, --stop-on-error        if activated, the code will try to exit when an error is encountered
"""
from pathlib import Path
import json
import argparse
import numpy as np
import pyarrow.csv as pacsv
import ax6_postproc
import pa_calc_mighty_tot as pacalc

# These are the options related to reading tsv files
readOptions = pacsv.ReadOptions(column_names=[],
                                autogenerate_column_names=True)
parseOptions = pacsv.ParseOptions(delimiter='\t')


def build_parser():
    # Configure the commands that can be fed to the command line
    help_msg = """run.py will analyze .tsv files in [tsv_file_dir] and \
    save output in [output_dir], using [DCCID] and [ses] in \
    those .tsv files."""
    parser = argparse.ArgumentParser(description=help_msg,
                                     epilog="Prepared by Jinseok Oh, Ph.D.")
    parser.add_argument("bids_dir", help="The path to the BIDS directory for your study (this is the same for all subjects)", type=str)
    parser.add_argument("output_dir", help="The path to the folder where outputs will be stored (this is the same for all subjects)", type=str)
    parser.add_argument("analysis_level", help="Should always be participant", type=str)
    # (7/18/24) dropping `study_tz`
    # parser.add_argument("study_tz", help="Timezone of the site where sensors were configured (ex. US/Pacific)", type=str)

    parser.add_argument('--participant_label', '--participant-label', help="The name/label of the subject to be processed (ex. sub-XXXXX)", type=str)
    parser.add_argument('--session_id', '--session-id', help="(optional) The name of a specific session to be processed (ex. ses-V02)", type=str)
    parser.add_argument('--interval', help="(optional) The label to correct or not the uneven sampling interval (raw or corrected)", type=str)
    parser.add_argument('--pa_measure', '--pa-measure', help="(optional) The computedQttyOption value (acceleration or jerk)", type=str)
    parser.add_argument('--pa_side', '--pa-side', help="(optional) which leg to calculate the physical activity level (Left/L or Right/R)", type=str)
    parser.add_argument('--entropy_type', '--entropy-type', help="(optional) Entropy type (SampEn or FuzzEn)", type=str)
    parser.add_argument('--entropy_measure', '--entropy-measure', help="(optional) Measure to calculate an entropy (avgacc or pkacc)", type=str)
    parser.add_argument('--stop_on_error', '--stop-on-error ', help="(optional) If activated, the code will try to exit if an error is encountered.", action='store_true')

    return parser

def main():

    parser = build_parser()
    args = parser.parse_args()

    # reassign variables to command line input
    bids_dir = Path(args.bids_dir).resolve()
    print('+-----------------------------------------------+')
    print('+ Movement sensor data batch processing begins! +')
    print('+-----------------------------------------------+')
    print(f'path for tsv folder: {bids_dir}')
    print('------------------------------')
    output_dir = Path(args.output_dir).resolve()

    # if output_dir does not exist, make one
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f'output_dir: {output_dir}')
    print('------------------------------')
    print(f'is output_dir generated? : {output_dir.exists()}')
    print('------------------------------\n')

    analysis_level = args.analysis_level
    if analysis_level != 'participant':
        raise ValueError("""Analysis level must be 'participant',
        but program received '{analysis_level}'""")

    # study time zone
    # study_tz = args.study_tz

    # Set session label
    if args.session_id:
        session_label = args.session_id
        if 'ses-' not in session_label:
            session_label = 'ses-' + session_label
    else:
        session_label = None

    # Find participants to try running
    if args.participant_label:
        participant_split = args.participant_label.split(' ')
        participants = []
        for temp_participant in participant_split:
            filepath = bids_dir / temp_participant
            participants.append(filepath)
    else:
        participants = list(bids_dir.glob('sub-*'))

    # 'Correct' uneven sampling rate?
    if args.interval:
        fs_handling = args.interval
    else:
        # the default should be 'raw'
        # this is already 'corrected' by skdh.io
        # if you REALLY need the interval to be
        # the same, then provide 'corrected' for
        # --interval
        fs_handling = 'raw'

    # 'acceleration' or 'jerk'
    if args.pa_measure:
        computedQttyOption = args.pa_measure
    else:
        computedQttyOption = None

    # 'L(eft)' or 'R(ight)' leg data to calculate PA
    if args.pa_side:
        pa_side = args.pa_side.lower()[0]
    else:
        pa_side = None

    # 'SampEn or 'FuzzEn' for entropy-type
    if args.entropy_type:
        entropy_label = args.entropy_type
    else:
        entropy_label = None

    # 'avgacc' or 'pkacc' for entropy-measure
    if args.entropy_measure:
        entropy_mat = args.entropy_measure
    else:
        entropy_mat = None

    # Let's save all these parameters
    param_outdict = {
            "bids_dir": str(bids_dir),
            "output_dir": str(output_dir),
            "analysis_level": analysis_level,
            "participant_label": args.participant_label,
            "session_id": session_label,
            "interval": fs_handling,
            "pa_measure": computedQttyOption,
            "pa_side": pa_side,
            "entropy_type": entropy_label,
            "entropy_measure": entropy_mat
            }
    # This will be saved as PARAMETERS.json.
    json_param = json.dumps(param_outdict, indent=4)

    # `mainFunction()` of pacalc,, the function to calculate the physical
    # activity needs one parameter, `infantLegLengthCm`.
    # An infants' leg lengths are not available, but the age in months
    # is recorded in /bids_dir/sub-xxxxxx/sub-xxxxxx_sessions.tsv.
    # Therefore we seek an estimate length from this dictionary.
    infantLegLengthCmDict = {0: 20.3, 1: 20.3, 2: 20.3,
                             3: 23.1, 4: 23.1, 5: 23.1,
                             6: 25.6, 7: 25.6, 8: 25.6,
                             9: 29.0, 10: 29.0, 11: 29.0,
                             12: 30.5, 13: 30.5, 14: 30.5, 15: 30.5,
                             16: 32.5, 17: 32.5, 18: 32.5, 19: 32.5,
                             20: 34.4, 21: 34.4, 22: 34.4, 23: 34.4}

    # Iterate through all participants
    for temp_participant in participants:
        sub = temp_participant.name
        print(f'New participant: {sub}')
        print('------------------------------')
        # Check that participant exists at the expected path
        # This should not be happening very often...
        if not temp_participant.exists():
            raise FileNotFoundError()

        # Find session/sessions
        if session_label is None:
            # 'V02', 'V03', may change to 1, 2
            sessions = list(temp_participant.glob('ses-*'))
            if not sessions:
                raise FileNotFoundError("No session specific folder exists")
        elif (temp_participant / session_label).exists():
            sessions = [temp_participant / session_label]
        else:
            raise AttributeError(f'{session_label} - incorrect session id')
        # Find the age of the infant (in month)
        # from 'bids_dir / {sub} / {sub}_sessions.tsv
        agetsv = pacsv.read_csv(temp_participant / f'{sub}_sessions.tsv',
                                parse_options=parseOptions)
        # Commented out line may be enough, if age is ALWAYS an integer.
        # People may do something like 6.2, however. I just take the
        # extra fool-proof measure.
        # sub_age = agetsv['age'][0].as_py()
        sub_age = np.ceil(agetsv['age'][0].as_py()).astype('int')
        print(f"sessions: {[y.name for y in sessions]}")
        print('------------------------------')

        # Iterate through sessions/motion
        for temp_session in sessions:
            ses = temp_session.name
            # default detrending method (median) and no in_en_dts specified
            print(f"Current working directory: {temp_session / 'motion'}")
            print('------------------------------')

            # Make sub-directories: Kinematics and PA (Physical Activity)
            tempout_dir = output_dir / sub / ses / 'motion'
            path_kinematics = tempout_dir / 'Kinematics'
            path_pa = tempout_dir / 'PA'
            path_kinematics.mkdir(parents=True, exist_ok=True)
            path_pa.mkdir(parents=True, exist_ok=True)

            with open(tempout_dir / 'PARAMETERS.json', 'w') as param_f:
                param_f.write(json_param)
                param_f.close()

            print(f"Positional/optional arguments provided are saved in : {tempout_dir / 'PARAMETERS.json'}")
            print('------------------------------')

            # Ax6 default sampling rate is 25Hz (not consistent).
            # So downsampling / interpolating the original signal at 20 Hz
            # (sampling rate of OPAL) was discussed earlier.
            # Downsampling tends to 'underestimate' the movement count.
            # (7/22/24) resampling mandated - mghazi's algorithm needs it.
            # requires data to be sampled at 20 Hz.
            # To facilitate the process, I make `calc_stats()` to save
            # calibrated tri-axial accelerometer data, resampled at 20 Hz.
            # This will later be loaded and fed to mghazi's algorithm.
            try:
                ax6_postproc.calc_stats(temp_session / 'motion',
                                        output_dir,
                                        sub,
                                        ses,
                                        fs=25,
                                        fs_handling=fs_handling,
                                        entropy_measure=entropy_mat,
                                        entropy_type=entropy_label,
                                        )
                # After this, run mghazi's algorithm
                suffix = ['leg-left', 'leg-right']
                if pa_side is not None:
                    if pa_side == 'l':
                        suffix = ['leg-left']
                    elif pa_side == 'r':
                        suffix = ['leg-right']

                if computedQttyOption is not None:
                    for sfx in suffix:
                        pacalc.mainFunction(tempout_dir,
                                            '_'.join((sub,ses,sfx,'desc-calibrated_recording-20_motion')),
                                            '_'.join((sub,ses,sfx,f'desc-{computedQttyOption}PA')),
                                            computedQttyOption,
                                            infantLegLengthCmDict[sub_age])
                else:
                    for computedQttyOption in ['acceleration', 'jerk']:
                        for sfx in suffix:
                            # inputDir, inputFileNameNoExtension,
                            # outputFileNameNoExtension, computedQttyOption,
                            # infantLegLengthCm
                            pacalc.mainFunction(tempout_dir,
                                                '_'.join((sub,ses,sfx,'desc-calibrated_recording-20_motion')),
                                                '_'.join((sub,ses,sfx,f'desc-{computedQttyOption}PA')),
                                                computedQttyOption,
                                                infantLegLengthCmDict[sub_age])

            except BaseException as e:
                if args.stop_on_error:
                    raise e
                else:
                    error_msg = f"""The movement sensor data of {sub}, {ses} \
                        were not fully processed. Please check the error message: {str(e)}\n"""
                    display_msg = f"""This message is also saved in the log:\n
                        {output_dir}/{sub}/{ses}/motion/LOG.txt\n"""
                    print(error_msg)
                    print(display_msg)
                    # create sub-folders
                    # final_outdir = output_dir / sub / ses
                    # final_outdir.mkdir(parents=True, exist_ok=True)
                    # create a log file
                    with open(f'{output_dir}/{sub}/{ses}/motion/LOG.txt', 'w') as f:
                        f.write(error_msg)
                        f.close()
                    continue


if __name__ == "__main__":
    main()
