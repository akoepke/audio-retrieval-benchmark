import argparse
import json
import logging
import os
import subprocess
import tqdm
import wget
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import random

def extracting_log_info(log_files, experiment, logging):
    metrics_t2v = defaultdict(list)
    metrics_v2t = defaultdict(list)
    
    for file_name in log_files:        
        output_string = f"{experiment}:\n"
        with open(Path("logs_eval") / file_name, 'r') as f:
            content_lines = f.read().splitlines()
        content_lines = content_lines[-14:]
        for line in content_lines:
            if 't2v' in line:
                metric_entry = line.split('test_t2v_metrics_')[1].split(':')[0]
                metrics_t2v[metric_entry].append(float(line.split('test_t2v_metrics_')[1].split(':')[1]))
            elif 'v2t' in line:
                metric_entry = line.split('test_v2t_metrics_')[1].split(':')[0]
                metrics_v2t[metric_entry].append(float(line.split('test_v2t_metrics_')[1].split(':')[1]))
        keys = list(metrics_t2v.keys())
    
    for key in keys:
        output_string += f"{key}_t2v: {np.mean(metrics_t2v[key]):.1f}, {np.std(metrics_t2v[key], ddof=1):.1f}\n"
    for key in keys:
        output_string += f"{key}_v2t: {np.mean(metrics_v2t[key]):.1f}, {np.std(metrics_v2t[key], ddof=1):.1f}\n"
    logging.info(output_string)
    with open(Path("logs_eval") / f"{experiment}_summary.txt", 'w') as f:
        f.write(output_string)

def run_exp(experiments, logging):
    for experiment in experiments:
        logging.info(f"Now running {experiment}")
        run_one_exp(experiment, experiments, logging)


def download_configs(experiment, trained_model_path, group_id, seed, timestamp):
    new_folder = str(trained_model_path).split('/trained_model.pth')[0]
    url_config = f"http://www.robots.ox.ac.uk/~vgg/research/collaborative-experts/data/models/{experiment}/{group_id}/{seed}/{timestamp}/config.json"
    config_path = Path(new_folder) / 'config.json'
    wget.download(url_config, out=str(config_path))
    with open(config_path, 'r') as f:
        config_content = json.load(f)
    config_content['seed'] = int(seed[-1])
    with open(config_path, 'w') as f:
        json.dump(config_content, f)


def download_models(experiment, logging, trained_model_path,
                    group_id, seed, timestamp):
    new_folder = str(trained_model_path).split('/trained_model.pth')[0]
    if os.path.exists(trained_model_path) is False:
        logging.info(f"Downloading model for {seed} since it does not exist on the local machine")
        url = f"http://www.robots.ox.ac.uk/~vgg/research/collaborative-experts/data/models/{experiment}/{group_id}/{seed}/{timestamp}/trained_model.pth"
        # import pdb; pdb.set_trace()
        Path(new_folder).mkdir(exist_ok=True, parents=True)
        wget.download(url, out=str(trained_model_path))
    else:
        logging.info(f"Model already downloaded for {experiment} seed {seed}")
    if os.path.exists(Path(new_folder) / 'config.json') is False:
        download_configs(experiment, trained_model_path, group_id, seed, timestamp)
    else:
        logging.info(f"Config already downloaded for {experiment} seed {seed}")

def run_one_exp(experiment, experiments, logging):
    if 'mmt' in experiment:
        if 'clotho' in experiment:
            ds = 'Clotho'
        elif 'audiocaps' in experiment:
            ds = 'AudioCaps'
        elif 'sounddescs' in experiment:
            ds = 'SoundDescs'
        cmd = f"python mmt/train.py --only_eval --config mmt/configs/%s_mmt.json --experiment %s"%(ds,experiment)
        subprocess.call(cmd, shell=True)
    else:
        group_id = experiments[experiment][0]

        with open('exp_to_seed_time.json', 'r') as f:
            json_dict = json.load(f)
        log_files = []
        for (group_id, seed, timestamp) in json_dict[experiment]:
            
            group_id_path = Path("data/saved/models") / experiment / group_id
            logging.info("Running evaluation on existent seeds")
            (Path("logs_eval")).mkdir(exist_ok=True, parents=True)
            trained_model_path = group_id_path / seed / timestamp / 'trained_model.pth'
            download_models(experiment, logging, trained_model_path,
                            group_id, seed, timestamp)
            config_path = group_id_path / seed / timestamp / 'config.json'
            cmd = f"python test.py --config {config_path} --resume {trained_model_path} --device 0 --eval_from_training_config >&1 | tee logs_eval/log_{group_id}_{seed}.txt"
            
            log_files.append(f"log_{group_id}_{seed}.txt")
            logging.info(cmd)
            subprocess.call(cmd, shell=True)
        logging.info("Now averaging results")
        
        extracting_log_info(log_files, experiment, logging)
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments_path", default="misc/experiments-audiocaps.json")
    parser.add_argument("--experiment", type=str, default=None)
    parser.add_argument(
        "--data_dir",
        type=Path,
        default="data",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
    )
    args = parser.parse_args()
    os.makedirs('logs', exist_ok=True)
    logging.basicConfig(filename=f"logs/{datetime.now().strftime(r'%m%d_%H%M%S')}.log",
                        level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler())
    logging.info(args)

    with open(args.experiments_path, "r") as f:
        experiments = json.load(f)
    
    if args.experiment is None:
        run_exp(experiments, logging)
    else:
        run_one_exp(args.experiment, experiments, logging)
    
    

if __name__ == "__main__":
    main()
