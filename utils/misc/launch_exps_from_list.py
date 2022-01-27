"""Launch a collection of experiments on SLURM from a text file.

EXP_LIST=audio-retrieval-exps.txt
ipy misc/launch_exps_from_list.py -- --exp_list "slurm/${EXP_LIST}" --yaspify

"""
import os
import sys
import json
import argparse
from pathlib import Path

from yaspi.yaspi import Yaspi
from utils.util import parse_grid, filter_cmd_args
from misc.aggregate_logs_and_stats import summarise

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_list", default="data/job-queues/latest.txt")
    parser.add_argument("--yaspify", action="store_true", help="launch via slurm")
    parser.add_argument("--slurm", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument('--mini_train', action="store_true")
    parser.add_argument("--use_cnodes", action="store_true")
    parser.add_argument('--train_single_epoch', action="store_true")
    parser.add_argument("--yaspi_defaults_path", type=Path,
                        default="misc/yaspi_gpu_defaults.json")
    parser.add_argument("--evaluation", type=str, default='train', choices=['train', 'test'])
    args = parser.parse_args()

    # construct list of experiments from text file
    with open(args.exp_list, "r") as f:
        custom_args = f.read().splitlines()
    # remove blank lines
    custom_args = [x for x in custom_args if x]

    if args.limit:
        custom_args = custom_args[:args.limit]

    parsed = {}
    for line in custom_args:
        parsed.update(parse_grid(line, args.evaluation))

    # flatten all parsed experiments
    custom_args = [x for group in parsed.values() for x in group]

    cmd_args = sys.argv[1:]
    remove = ["--yaspify", "--exp_list", "--use_cnodes", "--evaluation"]
    cmd_args = filter_cmd_args(cmd_args, remove=remove)
    base_cmd = f"python {args.evaluation}.py {' '.join(cmd_args)}"

    if args.yaspify:
        with open(args.yaspi_defaults_path, "r") as f:
            yaspi_defaults = json.load(f)
        if args.use_cnodes:
            yaspi_defaults.update({"partition": "compute", "gpus_per_task": 0})
        job_name = f"{Path(args.exp_list).stem}-{len(custom_args)}-exps"
        job_queue = [f'"{x}"' for x in custom_args]
        job_queue = " ".join(job_queue)
        job = Yaspi(
            cmd=base_cmd,
            job_queue=job_queue,
            job_name=job_name,
            job_array_size=len(custom_args),
            **yaspi_defaults,
        )
        job.submit(watch=True, conserve_resources=5)
    else:
        for custom_args_ in custom_args:
            base_cmd = f"{base_cmd} {custom_args_}"
            print(f"Running cmd: {base_cmd}")
            os.system(base_cmd)
    if args.evaluation =='train':
        for group_id in parsed:
            summarise(group_id=group_id)


if __name__ == "__main__":
    main()
