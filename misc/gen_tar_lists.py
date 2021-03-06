"""
ipy misc/gen_tar_lists.py -- --dataset YouCook2
"""
import copy
import json
import argparse
from typing import Dict, List, Tuple
from pathlib import Path

import tqdm
from beartype import beartype
from zsvision.zs_utils import load_json_config
from gen_readme import dataset_paths, model_specs2path


@beartype
def generate_tar_lists(
        save_dir: Path,
        experiments: Dict[str, Tuple[str, str]],
        datasets: List[str],
        refresh: bool,
):
    all_feat_paths = {}
    # import pdb; pdb.set_trace()
    for exp_name, (group_id, timestamp) in tqdm.tqdm(experiments.items()):
        rel_path = Path(group_id) / "seed-0" / timestamp / "config.json"
        config_path = Path(save_dir) / "models" / exp_name / rel_path
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
        except FileNotFoundError:
            rel_path = Path(group_id) / "seed-1" / timestamp / "config.json"
            config_path = Path(save_dir) / "models" / exp_name / rel_path
            with open(config_path, "r") as f:
                config = json.load(f)

        feat_aggregation = config["data_loader"]["args"]["feat_aggregation"]
        dataset_name = exp_name.split("-train")[0]
        if dataset_name not in [x.lower() for x in datasets]:
            continue
        if dataset_name not in all_feat_paths:
            all_feat_paths[dataset_name] = set()
        split_names = [config["data_loader"]["args"]["split_name"]]
        if "eval_settings" in config and config["eval_settings"]:
            test_split = config["eval_settings"]["data_loader"]["args"]["split_name"]
            split_names.append(test_split)
        keep = set(config["experts"]["modalities"])
        text_feat = config["experts"]["text_feat"]
        root_feat, paths = dataset_paths(dataset_name)
        modern_feat_agg = {key: val for key, val in feat_aggregation.items()
                           if key in paths["feature_names"]}
        feat_paths = model_specs2path(modern_feat_agg, keep)
        all_feat_paths[dataset_name].update({root_feat / x for x in feat_paths})
        for key, feat_list in paths["custom_paths"].items():
            for feat_path in feat_list:
                all_feat_paths[dataset_name].add(root_feat / feat_path)
        # import pdb; pdb.set_trace()
        text_paths = [root_feat / paths["text_feat_paths"][text_feat]]
        all_feat_paths[dataset_name].update(set(text_paths))
        all_feat_paths[dataset_name].add(root_feat / paths["raw_captions_path"])
        if "dict_youtube_mapping_path" in paths:
            all_feat_paths[dataset_name].add(
                root_feat / paths["dict_youtube_mapping_path"])
        for split_name in split_names:
            split_paths = set(root_feat / x for x in
                              paths["subset_list_paths"][split_name].values())
            all_feat_paths[dataset_name].update(split_paths)

    for dataset_name, paths in all_feat_paths.items():
        tar_include_list = Path("misc") / "datasets" / dataset_name / "tar_include.txt"
        tar_include_list.parent.mkdir(exist_ok=True, parents=True)
        if tar_include_list.exists() and not refresh:
            print(f"Found existing tar include list at {tar_include_list}, skipping...")
            continue
        with open(tar_include_list, "w") as f:
            for path in sorted(paths):
                if "aggregated_speech" not in str(path):
                    print(f"Writing {path} to {tar_include_list}")
                    f.write(f"{path}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", default="data/saved", type=Path)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--experiments_path", default="misc/experiments.json")
    parser.add_argument("--target", default="main")
    parser.add_argument("--data_dir", type=Path, default="data")
    parser.add_argument("--challenge_phase", default="public_server_val",
                        choices=["public_server_val", "public_server_test"])
    parser.add_argument("--datasets", nargs="+",
                        default=["activity-net",
                                 "QuerYD", "QuerYDSegments"])
    args = parser.parse_args()

    with open(args.experiments_path, "r") as f:
        experiments = json.load(f)

    generate_tar_lists(
        save_dir=args.save_dir,
        datasets=args.datasets,
        experiments=experiments,
        refresh=args.refresh,
    )


if __name__ == "__main__":
    main()
