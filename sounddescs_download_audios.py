import argparse
import logging
import multiprocessing as mp
import os
import zipfile
from datetime import datetime
from pathlib import Path
from zipfile import BadZipFile

import tqdm
import wget
from zsvision.zs_multiproc import starmap_with_kwargs


def download_audio(download_folder: Path,
                   main_link: str,
                   entry: str):
    """
    Downloading one audio file.
    Inputs:
        download_folder: Location where audio file is downloaded
        main_link: Address from where audio content is downloaded
        entry: Full link address for the specific audio file being downloaded
    """
    audio_id = entry.split(f"{main_link}/")[1].split(".wav.zip")[0].lower()
    print(f'Downloading file {audio_id}.wav')
    (download_folder / 'zip_audios' / audio_id).mkdir(parents=True, exist_ok=True)
    try:
        wget.download(entry, str(download_folder / 'zip_audios' / audio_id))
        print(f'Successfully downloaded file {audio_id}.wav')
    except Exception as e:
        print(f'File {audio_id} could not be downloaded because of error {e}')
        with open(Path('error_files') / f'{audio_id}.txt') as f:
            f.write('\n')


def download_audios(download_folder: Path, main_link: str,
                    logging, download_file, processes: int,
                    limit: int = 0):
    """
    Downloading all audio files from given list of links.
    Inputs:
        download_folder: Location where audio file is downloaded
        main_link: Address from where audio content is downloaded
        logging: Logging module containing information about the
            progress of the code
        download_file: Path of txt file containing links for audio files
        processes: Number of processes downloading audio content at
            the same time
        limit: If not 0, downloading only the first \{limit\} adio
            files from the list
    """
    logging.info(f'Creating folder {download_folder}/zip_audios')
    (download_folder / 'zip_audios').mkdir(parents=True, exist_ok=True)
    (Path('error_files')).mkdir(parents=True, exist_ok=True)
    with open(Path('sounddescs_data') / download_file, 'r') as f:
        entries = f.read().splitlines()
    if limit != 0:
        entries = entries[:limit]
    
    kwarg_list = []
    for entry in tqdm.tqdm(entries):
        kwarg_list.append({
            "download_folder": download_folder,
            "main_link": main_link,
            "entry": entry,
        })

    pool_func = download_audio
    if processes > 1:
        # The definition of the pool func must precede the creation of the pool
        # to ensure its pickleable.  We force the definition to occur by reassigning
        # the function.
        with mp.Pool(processes=processes) as pool:
            starmap_with_kwargs(pool=pool, func=pool_func, kwargs_iter=kwarg_list)
    else:
        for idx, kwarg in enumerate(kwarg_list):
            print(f"{idx}/{len(kwarg_list)} processing kwargs ")
            pool_func(**kwarg)


def unzip_one_file(download_folder: Path, audio_id: str, zip_audios: Path):
    """
    Unzipping one zip file from 'zip_audios' and moving it to the 'audios' folder.
    Inputs:
        download_folder: Location where audio file is downloaded
        audio_id: Name of the audio file being extracted
        zip_audios: Location where the zip files are found
    """
    audio_zip = os.listdir(zip_audios / audio_id)[0]
    if os.path.exists(download_folder / "audios" / audio_id) is False:
        try:
            with zipfile.ZipFile(zip_audios / audio_id / audio_zip, 'r') as zip_ref:
                print(f"Extracting audio file {audio_id}")
                zip_ref.extractall(download_folder / "audios" / audio_id)
                audio_file = os.listdir(download_folder / "audios" / audio_id)[0]
                os.rename(download_folder / "audios" / audio_id / audio_file,
                            download_folder / "audios" / audio_id / audio_file.lower())
        except BadZipFile:
            print(f"File {audio_id} could not be unzipped because of error BadZipFile")
            with open(Path("zip_error_files") / f"{audio_id}.txt") as f:
                f.write('\n')
    else:
        print(f"Audio file {audio_id} already extracted")


def unzip_files(download_folder: Path, processes: int, logging):
    """
    Unzipping all zip files and moving them to the audios folder.
    Inputs:
        download_folder: Location where audio file is downloaded
        logging: Logging module containing information about the
            progress of the code
    """
    zip_audios = download_folder / 'zip_audios'
    existent_audio_ids = os.listdir(zip_audios)
    (download_folder / "audios").mkdir(parents=True, exist_ok=True)
    (Path('zip_error_files')).mkdir(parents=True, exist_ok=True)
    kwarg_list = []
    for audio_id in tqdm.tqdm(existent_audio_ids):
        kwarg_list.append({
            "download_folder": download_folder,
            "audio_id": audio_id,
            "zip_audios": zip_audios,
        })
    logging.info("Starting unzipping files")
    
    pool_func = unzip_one_file
    if processes > 1:
        # The definition of the pool func must precede the creation of the pool
        # to ensure its pickleable.  We force the definition to occur by reassigning
        # the function.
        with mp.Pool(processes=processes) as pool:
            starmap_with_kwargs(pool=pool, func=pool_func, kwargs_iter=kwarg_list)
    else:
        for idx, kwarg in enumerate(kwarg_list):
            print(f"{idx}/{len(kwarg_list)} processing kwargs ")
            pool_func(**kwarg)
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--download_folder", type=Path,
                        required=True)
    parser.add_argument("--download_file", type=str,
                        default="download_links.txt")
    parser.add_argument("--action", type=str, default='download',
                        choices=['download', 'unzipping'])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--processes", type=int, default=1)
    parser.add_argument("--main_link", type=str,
                        default="https://sound-effects-media.bbcrewind.co.uk/zip")
    args = parser.parse_args()
    os.makedirs('logs', exist_ok=True)
    logging.basicConfig(filename=f"logs/{datetime.now().strftime(r'%m%d_%H%M%S')}.log",
                            level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler())
    if args.action == 'download':
        download_audios(args.download_folder, args.main_link,
                        logging, args.download_file, args.processes,
                        args.limit)
    else:
        unzip_files(args.download_folder, args.processes, logging)


if __name__ == "__main__":
    main()
