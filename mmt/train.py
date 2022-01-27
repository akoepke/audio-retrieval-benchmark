# Copyright 2020 Valentin Gabeur
# Copyright 2020 Samuel Albanie, Yang Liu and Arsha Nagrani
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Cross-modal architecture training.

Code based on the implementation of "Collaborative Experts":
https://github.com/albanie/collaborative-experts
"""

import argparse
import logging
import os
import wget
import random
import time
import json

import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
import numpy as np
from collections import defaultdict
from parse_config import ConfigParser
import torch
from trainer import Trainer
from utils import ranger
from pathlib import Path
from utils.nlp_utils import create_tokenizer
from utils.util import compute_dims
import utils.visualizer as module_vis

logger = logging.getLogger(__name__)


def download_models(experiment, trained_model_path, seed):
    new_folder = str(trained_model_path).split('/trained_model.pth')[0]
    if os.path.exists(trained_model_path) is False:
        print(f"Downloading model for {seed} since it does not exist on the local machine")
        url = f"http://www.robots.ox.ac.uk/~vgg/research/collaborative-experts/data/models/{experiment}/seed-{seed}/trained_model.pth"
        Path(new_folder).mkdir(exist_ok=True, parents=True)
        wget.download(url, out=str(trained_model_path))
    else:
        print(f"Model already downloaded for {experiment} seed {seed}")


def train(config):
  """Cross-modal architecture training."""

  # Get the list of experts and their dimensions
  expert_dims = compute_dims(config)
  raw_input_dims = {}
  for expert, expert_dic in expert_dims.items():
    raw_input_dims[expert] = expert_dic["dim"]

  # Set the random initial seeds

  # Tokenizer to parse sentences into tokens
  tokenizer = create_tokenizer(config["arch"]["args"]["txt_inp"])

  tic = time.time()
  seed = config._args.seed
  cross_seed = config.get("cross_seed", seed)
  logger.debug("Setting experiment random seed to %d", seed)
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)

  # Create the datasets
  logger.info("Preparing the dataloaders ...")
  dataset_types = ["train_sets", "continuous_eval_sets", "final_eval_sets"]
  data_loaders = {}
  loaded_data = {}
  for dataset_type in dataset_types:
    training = dataset_type == "train_sets"
    if not config.get(dataset_type, False):
      continue
    data_loaders[dataset_type] = []
    for _, data_loader in enumerate(config[dataset_type]):
      data_loaders[dataset_type].append(
          getattr(module_data, data_loader["type"])(
              **data_loader["args"],
              raw_input_dims=raw_input_dims,
              training=training,
              tokenizer=tokenizer,
              loaded_data=loaded_data,
              cross_seed=cross_seed,
          ))
  # Setup the cross-modal architecture
  model = config.init(
      name="arch",
      module=module_arch,
      expert_dims=expert_dims,
      tokenizer=tokenizer,
  )

  loss = config.init(name="loss", module=module_loss)
  metrics = [getattr(module_metric, met) for met in config["metrics"]]
  trainable_params = filter(lambda p: p.requires_grad, model.parameters())

  if config["optimizer"]["type"] == "Ranger":
    optimizer = config.init("optimizer", ranger, trainable_params)
  else:
    optimizer = config.init("optimizer", torch.optim, trainable_params)

  lr_scheduler = config.init("lr_scheduler", torch.optim.lr_scheduler,
                             optimizer)

  if "warmup_iterations" in config["optimizer"]:
    warmup_iterations = config["optimizer"]["warmup_iterations"]
  else:
    warmup_iterations = -1

  visualizer = config.init(
      name="visualizer",
      module=module_vis,
      exp_name=config.exper_name,
      web_dirs=config.web_dirs,
  )

  trainer = Trainer(
      model,
      loss,
      metrics,
      optimizer,
      config=config,
      data_loaders=data_loaders,
      lr_scheduler=lr_scheduler,
      visualizer=visualizer,
      skip_first_n_saves=config["trainer"].get("skip_first_n_saves", 0),
      include_optim_in_ckpts=config["trainer"].get("include_optim_in_ckpts",
                                                   False),
      expert_dims=expert_dims,
      tokenizer=tokenizer,
      warmup_iterations=warmup_iterations)

  if not config.only_eval:
    logger.info("Training ...")
    trainer.train()
  logger.info("Final evaluation ...")
  trainer.evaluate()
  duration = time.strftime("%Hh%Mm%Ss", time.gmtime(time.time() - tic))
  logger.info("Script took %s", duration)

  # Report the location of the "best" checkpoint of the final seeded run (here
  # "best" corresponds to the model with the highest geometric mean over the
  # R@1, R@5 and R@10 metrics when a validation set is used, or simply the final
  # epoch of training for fixed-length schedules).
  best_ckpt_path = config.save_dir / "trained_model.pth"
  if os.path.exists(best_ckpt_path):
    logger.info("The best performing ckpt can be found at %s",
                str(best_ckpt_path))


def main_train(raw_args=None):
  parser = argparse.ArgumentParser(description="PyTorch Template")
  parser.add_argument("--config",
                      default=None,
                      type=str,
                      help="config file path (default: None)")
  parser.add_argument(
      "--resume",
      default=None,
      type=str,
      help="path to the experiment dir to resume (default: None)")
  parser.add_argument("--device", type=str, help="indices of GPUs to enable")
  parser.add_argument("--only_eval", action="store_true")
  parser.add_argument("--load_checkpoint",
                      default=None,
                      type=str,
                      help="path to the checkpoint to load (default: None)") 
  
  parser.add_argument("--seeds", default="0,1,2", help="comma separated list of seeds")
  parser.add_argument("--experiment", type=str, default="clotho-train-vggish-vggsound-mmt", help="specify the name of the experiment")
  parser.add_argument("-v",
                      "--verbose",
                      help="increase output verbosity",
                      action="store_true")
  argsin = parser.parse_args(raw_args)
  
  seeds = [int(x) for x in argsin.seeds.split(",")]
  for ii, seed in enumerate(seeds):
    setattr(argsin, 'seed', seed)
    print('Setting experiment random seed to %d'%seed)
    if argsin.only_eval:
        trained_model_path = Path("data/saved/models") / Path(argsin.experiment) 
        trained_model_path = trained_model_path / Path('seed-%d/trained_model.pth'%seed)
        logging.info("Running evaluation on existent seeds")
        download_models(argsin.experiment, trained_model_path, seed)
        argsin.load_checkpoint = trained_model_path
    args = ConfigParser(argsin)
    
    msg = (
        f"Expected the number of training epochs ({args['trainer']['epochs']})"
        f"to exceed the save period ({args['trainer']['save_period']}), otherwise"
        " no checkpoints will be saved.")
    assert args["trainer"]["epochs"] >= args["trainer"]["save_period"], msg
    train(config=args)
  
  savedir = os.path.split(args.save_dir)[0]
  jsonpath = 'exp_results.json'
  exp_files = [os.path.join(savedir,'seed-%d'%seed) for seed in seeds]
  metrics_t2v = defaultdict(list)
  metrics_v2t = defaultdict(list)
  for file_name in exp_files:
    logfile = os.path.join(file_name, jsonpath)
    with open(logfile, 'r') as f:
      json_content = json.load(f)
      for k, v in json_content['perfs'][list(json_content['perfs'].keys())[0]].items():
            metric_name = k.split('/')[1]
            if not 'cols' in k:
                if 't2v' in k:
                    metrics_t2v[metric_name].append(v)
                elif 'v2t' in k:
                    metrics_v2t[metric_name].append(v)
  print('Now averaging results')
  print(argsin.experiment + ':')
  output_string = f"\n"
  keys = list(metrics_t2v.keys())  
  for key in keys:
      output_string += f"{key}_t2v: {np.mean(metrics_t2v[key]):.1f}, {np.std(metrics_t2v[key], ddof=1):.1f}\n"
  for key in keys:
      output_string += f"{key}_v2t: {np.mean(metrics_v2t[key]):.1f}, {np.std(metrics_v2t[key], ddof=1):.1f}\n"
  print(output_string)

if __name__ == "__main__":
  main_train()
