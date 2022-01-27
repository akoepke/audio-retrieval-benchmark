import argparse
import logging
import multiprocessing as mp
import os
import subprocess
from datetime import datetime
from pathlib import Path

import tqdm
from zsvision.zs_multiproc import starmap_with_kwargs


def rename_files(initial_folder: Path):
    """
    This function makes sure all files use consistent lowercase names.
    Inputs:
        initial_folder: Location where the file to be renamed is found
    """
    audio_files = os.listdir(initial_folder / 'audios')
    for audio_file in tqdm.tqdm(audio_files):
        os.rename(initial_folder / 'audios' / audio_file, initial_folder / 'audios' / audio_file.lower())
    

def audio_to_new_sampling(audio_path: Path, dest_audio_path: Path):
    """
    This function resamples the initial audio file found at path \{audio_path\}
    and generates a new one with 16kHz sampling rate at \{dest_audio_path\}
    Inputs:
        audio_path: Location of file to be resampled
        dest_audio_path: Location of resampled file
    """
    cmd = ['ffmpeg', '-i', str(audio_path), '-ar', '16000', '-ac', '1', str(dest_audio_path)]
    print(f'Running this command {cmd}')
    subprocess.call(cmd, stdout=subprocess.PIPE)


def resample_wavs(dest_folder: Path, initial_folder: Path,
                  processes: int, logging):
    """
    This function generates a list of initial paths of files to be
    resampled and the names of the new resampled files. Then the
    audio_to_new_sampling function is called for each file in the list.
    Inputs:
        dest_folder: Location where new resampled files will be saved
        initial_folder: Location of files to be resampled
        processes: How many files are resampled at the same time
        logging: Logging module containing information about the script
    """
    (dest_folder).mkdir(parents=True, exist_ok=True)
    audio_folders = os.listdir(initial_folder / 'audios')

    kwarg_list = []

    for audio_folder in tqdm.tqdm(audio_folders):
        audio_file = os.listdir(initial_folder / 'audios' / audio_folder)[0]
        audio_path = initial_folder / 'audios' / audio_folder / audio_file
        dest_audio_path = dest_folder / audio_file.lower()
        if os.path.exists(dest_audio_path) is False:
            kwarg_list.append({'audio_path': audio_path,
                               'dest_audio_path': dest_audio_path
                                })
        else:
            logging.info(f'File {audio_path} already transformed')
    
    pool_func = audio_to_new_sampling

    if processes > 1:
        # The definition of the pool func must precede the creation of the pool
        # to ensure its pickleable.  We force the definition to occur by reassigning
        # the function.
        with mp.Pool(processes=processes) as pool:
            starmap_with_kwargs(pool=pool, func=pool_func, kwargs_iter=kwarg_list)
    else:
        for idx, kwarg in enumerate(kwarg_list):
            logging.info(f'{idx}/{len(kwarg_list)} processing kwargs ')
            pool_func(**kwarg)     


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dest_folder', type=Path,
                        help='Location where resampled files are saved')
    parser.add_argument('--initial_folder', type=Path, required=True,
                        help='Location where files to be resampled are found')
    parser.add_argument('--exp', default='rename', choices=['rename', 'resample'])
    parser.add_argument('--processes', type=int, default=60)
    args = parser.parse_args()
    os.makedirs('logs', exist_ok=True)
    logging.basicConfig(filename=f"logs/{datetime.now().strftime(r'%m%d_%H%M%S')}.log",
                        level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler())
    if args.exp == 'rename':
        rename_files(args.initial_folder)
    else:
        if args.dest_folder is not None:
            resample_wavs(args.dest_folder, args.initial_folder,
                          args.processes, logging)
        else:
            raise Exception("Need to add flag --dest_folder with folder location where resampled files should be saved")


if __name__ == "__main__":
    main()
