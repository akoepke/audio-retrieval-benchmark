# Audio Retrieval with Natural Language Queries

This repository is the implementation of [Audio Retrieval with Natural Language Queries](https://arxiv.org/pdf/2105.02192.pdf) and it is based on the [Use What You Have: Video retrieval using representations from collaborative experts](https://github.com/albanie/collaborative-experts) repo. Datasets used in this paper are [SoundDescs](https://arxiv.org/pdf/2112.09418.pdf), [AudioCaps](https://www.aclweb.org/anthology/N19-1011.pdf), [CLOTHO](https://arxiv.org/pdf/1910.09387.pdf), [Activity-Net](https://arxiv.org/pdf/1705.00754.pdf) and [QuerYD](https://arxiv.org/pdf/2011.11071.pdf).

### Requirements

The required libraries for running this code can be found in requirements/requirements.txt. The following packages need to be installed:
* ipython
* yaspi
* torch==1.1.0
* zsvision
* numpy
* ipdb
* scikit-learn
* dominate
* mock
* pytorch-swats
* wget
* tqdm

Cuda 10.1 and python 3.7 were used when generating results and models.

```
conda create --name audio-retrieval python=3.7
conda activate audio-retrieval
pip install -r requirements/requirements.txt
```

To be able to run the code below, features extracted from various datasets need to be downloaded. If there is not enough space in your working location to store some of these features (for SoundDescs and AudioCaps the files are larger than 6GB while the others are under 1GB) then you will need to create a folder called `data` inside this repository which should be a symlink to a folder where enogh memory exists. As an example, run the following from the audio-experts code-base.
```
ln -s <path-where-data-can-be-saved> data
```
To download features for each dataset, follow the steps [here](https://github.com/oncescuandreea/audio-experts/blob/audiocaps/misc/exps-names.md)

### Evaluating a pretrained model on multiple seeds and reproducing results
To reproduce results in tables below, multiple models trained with different seeds need to be downloaded and evaluated on the test sets.

The steps needed to reproduce results are:
1. Picking the experiment to be reproduced which is in the form `<dataset-name>-<config-file-name>`. Tables with experiments names and the corresponding form can be found in [`misc/exps-names.md`](https://github.com/oncescuandreea/audio-experts/blob/audiocaps/misc/exps-names.md).
2. Downloading the features and splits corresponding to the dataset for which the experiment is run. For example for AudioCaps run:
```
# fetch the pretrained experts for AudioCaps 
python3 misc/sync_experts.py --dataset AudioCaps
```
Additional examples for the datasets used in this paper can be found in [`misc/exps-names.md`](https://github.com/oncescuandreea/audio-experts/blob/audiocaps/misc/exps-names.md).

3. Running the `eval.py` script.

For example, to reproduce the experiments for audiocaps with complete visual and audio experts, run the following line:
```
python eval.py --experiment audiocaps-train-full-ce-r2p1d-inst-vggish-vggsound
```
If the --experiment flag is not provided, the `eval.py` script will download and evaluate all models on the test set.


### Training a new model

Training a new video-text embedding requires:
1. The pretrained experts for the dataset used for training, which should be located in `<root>/data/<dataset-name>/symlinked-feats` (this will be done automatically by the [utility script](misc/sync_experts.py), or can be done manually). Examples can be found in [`misc/exps-names.md`](https://github.com/oncescuandreea/audio-experts/blob/audiocaps/misc/exps-names.md).
2. A `config.json` file.  You can define your own, or use one of the provided configs in the [configs](configs) directory.

Training is then performed with the following command:
```
python3 train.py --config <path-to-config.json> --device <gpu-id>
```
where `<gpu-id>` is the index of the GPU to train on.  This option can be ommitted to run the training on the CPU.

For example, to train a new embedding for the CLOTHO dataset, run the following sequence of commands:

```
# fetch the pretrained experts for CLOTHO 
python3 misc/sync_experts.py --dataset CLOTHO

# Train the model
python3 train.py --config configs/clotho/train-vggish-vggsound.json --device 0
```


### AudioCaps

#### These are the retrieval results obtained for the AudioCaps dataset when using only audio experts:


| Experts | Task | R@1 | R@5 | R@10 | R@50 | MdR | MnR | Geom | params | Links |
| ----- | ---- | --- | --- | ---- | ---- | --- | --- | ----- | -- | -- |
| CE - VGGish   | t2v  | {{audiocaps-train-full-ce-only-audio.geomt2v}} | {{audiocaps-train-full-ce-only-audio.params}} | [config]({{audiocaps-train-full-ce-only-audio.config}}), [model]({{audiocaps-train-full-ce-only-audio.model}}), [log]({{audiocaps-train-full-ce-only-audio.log}}) |
| CE - VGGish    | v2t  | {{audiocaps-train-full-ce-only-audio.geomv2t}} | {{audiocaps-train-full-ce-only-audio.params}} | [config]({{audiocaps-train-full-ce-only-audio.config}}), [model]({{audiocaps-train-full-ce-only-audio.model}}), [log]({{audiocaps-train-full-ce-only-audio.log}}) |
| CE - VGGSound    | t2v  | {{audiocaps-train-only-vggsound.geomt2v}} | {{audiocaps-train-only-vggsound.params}} | [config]({{audiocaps-train-only-vggsound.config}}), [model]({{audiocaps-train-only-vggsound.model}}), [log]({{audiocaps-train-only-vggsound.log}}) |
| CE - VGGSound   | v2t  | {{audiocaps-train-only-vggsound.geomv2t}} | {{audiocaps-train-only-vggsound.params}} | [config]({{audiocaps-train-only-vggsound.config}}), [model]({{audiocaps-train-only-vggsound.model}}), [log]({{audiocaps-train-only-vggsound.log}}) |
| CE - VGGish + VGGSound    | t2v  | {{audiocaps-train-vggish-vggsound.geomt2v}} | {{audiocaps-train-vggish-vggsound.params}} | [config]({{audiocaps-train-vggish-vggsound.config}}), [model]({{audiocaps-train-vggish-vggsound.model}}), [log]({{audiocaps-train-vggish-vggsound.log}}) |
| CE - VGGish + VGGSound   | v2t  | {{audiocaps-train-vggish-vggsound.geomv2t}} | {{audiocaps-train-vggish-vggsound.params}} | [config]({{audiocaps-train-vggish-vggsound.config}}), [model]({{audiocaps-train-vggish-vggsound.model}}), [log]({{audiocaps-train-vggish-vggsound.log}}) |
| MoEE - VGGish + VGGSound   | t2v  | {{audiocaps-train-vggish-vggsound-moee.geomt2v}} | {{audiocaps-train-vggish-vggsound-moee.params}} | [config]({{audiocaps-train-vggish-vggsound-moee.config}}), [model]({{audiocaps-train-vggish-vggsound-moee.model}}), [log]({{audiocaps-train-vggish-vggsound-moee.log}}) |
| MoEE - VGGish + VGGSound   | v2t  | {{audiocaps-train-vggish-vggsound-moee.geomv2t}} | {{audiocaps-train-vggish-vggsound-moee.params}} | [config]({{audiocaps-train-vggish-vggsound-moee.config}}), [model]({{audiocaps-train-vggish-vggsound-moee.model}}), [log]({{audiocaps-train-vggish-vggsound-moee.log}}) |


#### Using only visual experts for AudioCaps:

| Experts | Task | R@1 | R@5 | R@10 | R@50 | MdR | MnR | Geom | params | Links |
| ----- | ---- | --- | --- | ---- | ---- | --- | --- | ----- | -- | -- |
| CE - Scene   | t2v  | {{audiocaps-train-full-ce-only-scene.geomt2v}} | {{audiocaps-train-full-ce-only-scene.params}} | [config]({{audiocaps-train-full-ce-only-scene.config}}), [model]({{audiocaps-train-full-ce-only-scene.model}}), [log]({{audiocaps-train-full-ce-only-scene.log}}) |
| CE - Scene    | v2t  | {{audiocaps-train-full-ce-only-scene.geomv2t}} | {{audiocaps-train-full-ce-only-scene.params}} | [config]({{audiocaps-train-full-ce-only-scene.config}}), [model]({{audiocaps-train-full-ce-only-scene.model}}), [log]({{audiocaps-train-full-ce-only-scene.log}}) |
| CE - R2P1D   | t2v  | {{audiocaps-train-full-ce-only-r2p1d.geomt2v}} | {{audiocaps-train-full-ce-only-r2p1d.params}} | [config]({{audiocaps-train-full-ce-only-r2p1d.config}}), [model]({{audiocaps-train-full-ce-only-r2p1d.model}}), [log]({{audiocaps-train-full-ce-only-r2p1d.log}}) |
| CE - R2P1D   | v2t  | {{audiocaps-train-full-ce-only-r2p1d.geomv2t}} | {{audiocaps-train-full-ce-only-r2p1d.params}} | [config]({{audiocaps-train-full-ce-only-r2p1d.config}}), [model]({{audiocaps-train-full-ce-only-r2p1d.model}}), [log]({{audiocaps-train-full-ce-only-r2p1d.log}}) |
| CE - Inst    | t2v  | {{audiocaps-train-full-ce-only-inst.geomt2v}} | {{audiocaps-train-full-ce-only-inst.params}} | [config]({{audiocaps-train-full-ce-only-inst.config}}), [model]({{audiocaps-train-full-ce-only-inst.model}}), [log]({{audiocaps-train-full-ce-only-inst.log}}) |
| CE - Inst   | v2t  | {{audiocaps-train-full-ce-only-inst.geomv2t}} | {{audiocaps-train-full-ce-only-inst.params}} | [config]({{audiocaps-train-full-ce-only-inst.config}}), [model]({{audiocaps-train-full-ce-only-inst.model}}), [log]({{audiocaps-train-full-ce-only-inst.log}}) |
| CE - Scene + R2P1D   | t2v  | {{audiocaps-train-full-ce-scene-r2p1d.geomt2v}} | {{audiocaps-train-full-ce-scene-r2p1d.params}} | [config]({{audiocaps-train-full-ce-scene-r2p1d.config}}), [model]({{audiocaps-train-full-ce-scene-r2p1d.model}}), [log]({{audiocaps-train-full-ce-scene-r2p1d.log}}) |
| CE - Scene + R2P1D   | v2t  | {{audiocaps-train-full-ce-scene-r2p1d.geomv2t}} | {{audiocaps-train-full-ce-scene-r2p1d.params}} | [config]({{audiocaps-train-full-ce-scene-r2p1d.config}}), [model]({{audiocaps-train-full-ce-scene-r2p1d.model}}), [log]({{audiocaps-train-full-ce-scene-r2p1d.log}}) |
| CE - Scene + Inst   | t2v  | {{audiocaps-train-full-ce-scene-inst.geomt2v}} | {{audiocaps-train-full-ce-scene-inst.params}} | [config]({{audiocaps-train-full-ce-scene-inst.config}}), [model]({{audiocaps-train-full-ce-scene-inst.model}}), [log]({{audiocaps-train-full-ce-scene-inst.log}}) |
| CE - Scene + Inst   | v2t  | {{audiocaps-train-full-ce-scene-inst.geomv2t}} | {{audiocaps-train-full-ce-scene-inst.params}} | [config]({{audiocaps-train-full-ce-scene-inst.config}}), [model]({{audiocaps-train-full-ce-scene-inst.model}}), [log]({{audiocaps-train-full-ce-scene-inst.log}}) |
| CE - R2P1D + Inst   | t2v  | {{audiocaps-train-full-ce-r2p1d-inst.geomt2v}} | {{audiocaps-train-full-ce-r2p1d-inst.params}} | [config]({{audiocaps-train-full-ce-r2p1d-inst.config}}), [model]({{audiocaps-train-full-ce-r2p1d-inst.model}}), [log]({{audiocaps-train-full-ce-r2p1d-inst.log}}) |
| CE - R2P1D + Inst   | v2t  | {{audiocaps-train-full-ce-r2p1d-inst.geomv2t}} | {{audiocaps-train-full-ce-r2p1d-inst.params}} | [config]({{audiocaps-train-full-ce-r2p1d-inst.config}}), [model]({{audiocaps-train-full-ce-r2p1d-inst.model}}), [log]({{audiocaps-train-full-ce-r2p1d-inst.log}}) |


#### Visual and audio experts for AudioCaps:

| Experts | Task | R@1 | R@5 | R@10 | R@50 | MdR | MnR | Geom | params | Links |
| ----- | ---- | --- | --- | ---- | ---- | --- | --- | ----- | -- | -- |
| CE - R2P1D + Inst + VGGish   | t2v  | {{audiocaps-train-full-ce-r2p1d-inst-vggish.geomt2v}} | {{audiocaps-train-full-ce-r2p1d-inst-vggish.params}} | [config]({{audiocaps-train-full-ce-r2p1d-inst-vggish.config}}), [model]({{audiocaps-train-full-ce-r2p1d-inst-vggish.model}}), [log]({{audiocaps-train-full-ce-r2p1d-inst-vggish.log}}) |
| CE - R2P1D + Inst + VGGish  | v2t  | {{audiocaps-train-full-ce-r2p1d-inst-vggish.geomv2t}} | {{audiocaps-train-full-ce-r2p1d-inst-vggish.params}} | [config]({{audiocaps-train-full-ce-r2p1d-inst-vggish.config}}), [model]({{audiocaps-train-full-ce-r2p1d-inst-vggish.model}}), [log]({{audiocaps-train-full-ce-r2p1d-inst-vggish.log}}) |
| CE - R2P1D + Inst + VGGSound   | t2v  | {{audiocaps-train-full-ce-r2p1d-inst-vggsound.geomt2v}} | {{audiocaps-train-full-ce-r2p1d-inst-vggsound.params}} | [config]({{audiocaps-train-full-ce-r2p1d-inst-vggsound.config}}), [model]({{audiocaps-train-full-ce-r2p1d-inst-vggsound.model}}), [log]({{audiocaps-train-full-ce-r2p1d-inst-vggsound.log}}) |
| CE - R2P1D + Inst + VGGSound  | v2t  | {{audiocaps-train-full-ce-r2p1d-inst-vggsound.geomv2t}} | {{audiocaps-train-full-ce-r2p1d-inst-vggsound.params}} | [config]({{audiocaps-train-full-ce-r2p1d-inst-vggsound.config}}), [model]({{audiocaps-train-full-ce-r2p1d-inst-vggsound.model}}), [log]({{audiocaps-train-full-ce-r2p1d-inst-vggsound.log}}) |
| CE - R2P1D + Inst +VGGish + VGGSound  | t2v  | {{audiocaps-train-full-ce-r2p1d-inst-vggish-vggsound.geomt2v}} | {{audiocaps-train-full-ce-r2p1d-inst-vggish-vggsound.params}} | [config]({{audiocaps-train-full-ce-r2p1d-inst-vggish-vggsound.config}}), [model]({{audiocaps-train-full-ce-r2p1d-inst-vggish-vggsound.model}}), [log]({{audiocaps-train-full-ce-r2p1d-inst-vggish-vggsound.log}}) |
| CE - R2P1D + Inst +VGGish + VGGSound  | v2t  | {{audiocaps-train-full-ce-r2p1d-inst-vggish-vggsound.geomv2t}} | {{audiocaps-train-full-ce-r2p1d-inst-vggish-vggsound.params}} | [config]({{audiocaps-train-full-ce-r2p1d-inst-vggish-vggsound.config}}), [model]({{audiocaps-train-full-ce-r2p1d-inst-vggish-vggsound.model}}), [log]({{audiocaps-train-full-ce-r2p1d-inst-vggish-vggsound.log}}) |

### CLOTHO

| Experts | Task | R@1 | R@5 | R@10 | R@50 | MdR | MnR | Geom | params | Links |
| ----- | ---- | --- | --- | ---- | ---- | --- | --- | ----- | -- | -- |
| CE - VGGish   | t2v  | {{clotho-train-full-ce-only-audio.geomt2v}} | {{clotho-train-full-ce-only-audio.params}} | [config]({{clotho-train-full-ce-only-audio.config}}), [model]({{clotho-train-full-ce-only-audio.model}}), [log]({{clotho-train-full-ce-only-audio.log}}) |
| CE - VGGish   | v2t  | {{clotho-train-full-ce-only-audio.geomv2t}} | {{clotho-train-full-ce-only-audio.params}} | [config]({{clotho-train-full-ce-only-audio.config}}), [model]({{clotho-train-full-ce-only-audio.model}}), [log]({{clotho-train-full-ce-only-audio.log}}) |
| CE - VGGish + VGGSound    | t2v  | {{clotho-train-vggish-vggsound.geomt2v}} | {{clotho-train-vggish-vggsound.params}} | [config]({{clotho-train-vggish-vggsound.config}}), [model]({{clotho-train-vggish-vggsound.model}}), [log]({{clotho-train-vggish-vggsound.log}}) |
| CE - VGGish + VGGSound   | v2t  | {{clotho-train-vggish-vggsound.geomv2t}} | {{clotho-train-vggish-vggsound.params}} | [config]({{clotho-train-vggish-vggsound.config}}), [model]({{clotho-train-vggish-vggsound.model}}), [log]({{clotho-train-vggish-vggsound.log}}) |
| MoEE - VGGish + VGGSound   | t2v  | {{clotho-train-vggish-vggsound-moee.geomt2v}} | {{clotho-train-vggish-vggsound-moee.params}} | [config]({{clotho-train-vggish-vggsound-moee.config}}), [model]({{clotho-train-vggish-vggsound-moee.model}}), [log]({{clotho-train-vggish-vggsound-moee.log}}) |
| MoEE - VGGish + VGGSound   | v2t  | {{clotho-train-vggish-vggsound-moee.geomv2t}} | {{clotho-train-vggish-vggsound-moee.params}} | [config]({{clotho-train-vggish-vggsound-moee.config}}), [model]({{clotho-train-vggish-vggsound-moee.model}}), [log]({{clotho-train-vggish-vggsound-moee.log}}) |

### SoundDescs

| Experts | Task | R@1 | R@5 | R@10 | R@50 | MdR | MnR | Geom | params | Links |
| ----- | ---- | --- | --- | ---- | ---- | --- | --- | ----- | -- | -- |
| CE - VGGish   | t2v  | {{sounddescs-train-full-ce-only-audio.geomt2v}} | {{sounddescs-train-full-ce-only-audio.params}} | [config]({{sounddescs-train-full-ce-only-audio.config}}), [model]({{sounddescs-train-full-ce-only-audio.model}}), [log]({{sounddescs-train-full-ce-only-audio.log}}) |
| CE - VGGish   | v2t  | {{sounddescs-train-full-ce-only-audio.geomv2t}} | {{sounddescs-train-full-ce-only-audio.params}} | [config]({{sounddescs-train-full-ce-only-audio.config}}), [model]({{sounddescs-train-full-ce-only-audio.model}}), [log]({{sounddescs-train-full-ce-only-audio.log}}) |
| CE - VGGish + VGGSound    | t2v  | {{sounddescs-train-vggish-vggsound.geomt2v}} | {{sounddescs-train-vggish-vggsound.params}} | [config]({{sounddescs-train-vggish-vggsound.config}}), [model]({{sounddescs-train-vggish-vggsound.model}}), [log]({{sounddescs-train-vggish-vggsound.log}}) |
| CE - VGGish + VGGSound   | v2t  | {{sounddescs-train-vggish-vggsound.geomv2t}} | {{sounddescs-train-vggish-vggsound.params}} | [config]({{sounddescs-train-vggish-vggsound.config}}), [model]({{sounddescs-train-vggish-vggsound.model}}), [log]({{sounddescs-train-vggish-vggsound.log}}) |
| MoEE - VGGish + VGGSound   | t2v  | {{sounddescs-train-vggish-vggsound-moee.geomt2v}} | {{sounddescs-train-vggish-vggsound-moee.params}} | [config]({{sounddescs-train-vggish-vggsound-moee.config}}), [model]({{sounddescs-train-vggish-vggsound-moee.model}}), [log]({{sounddescs-train-vggish-vggsound-moee.log}}) |
| MoEE - VGGish + VGGSound   | v2t  | {{sounddescs-train-vggish-vggsound-moee.geomv2t}} | {{sounddescs-train-vggish-vggsound-moee.params}} | [config]({{sounddescs-train-vggish-vggsound-moee.config}}), [model]({{sounddescs-train-vggish-vggsound-moee.model}}), [log]({{sounddescs-train-vggish-vggsound-moee.log}}) |

### Pretraining on SoundDescs, finetuning on AudioCaps

| Experts | Task | R@1 | R@5 | R@10 | R@50 | MdR | MnR | Geom | params | Links |
| ----- | ---- | --- | --- | ---- | ---- | --- | --- | ----- | -- | -- |
| CE - VGGish + VGGSound   | t2v  | {{audiocaps-train-vggish-vggsound-finetuned-from-sd.geomt2v}} | {{audiocaps-train-vggish-vggsound-finetuned-from-sd.params}} | [config]({{audiocaps-train-vggish-vggsound-finetuned-from-sd.config}}), [model]({{audiocaps-train-vggish-vggsound-finetuned-from-sd.model}}), [log]({{audiocaps-train-vggish-vggsound-finetuned-from-sd.log}}) |
| CE - VGGish + VGGSound    | v2t  | {{audiocaps-train-vggish-vggsound-finetuned-from-sd.geomv2t}} | {{audiocaps-train-vggish-vggsound-finetuned-from-sd.params}} | [config]({{audiocaps-train-vggish-vggsound-finetuned-from-sd.config}}), [model]({{audiocaps-train-vggish-vggsound-finetuned-from-sd.model}}), [log]({{audiocaps-train-vggish-vggsound-finetuned-from-sd.log}}) |

### Pretraining on AudioCaps, finetuning on CLOTHO

| Experts | Task | R@1 | R@5 | R@10 | R@50 | MdR | MnR | Geom | params | Links |
| ----- | ---- | --- | --- | ---- | ---- | --- | --- | ----- | -- | -- |
| CE - VGGish + VGGSound   | t2v  | {{clotho-train-vggish-vggsound-finetuned-from-ac.geomt2v}} | {{clotho-train-vggish-vggsound-finetuned-from-ac.params}} | [config]({{clotho-train-vggish-vggsound-finetuned-from-ac.config}}), [model]({{clotho-train-vggish-vggsound-finetuned-from-ac.model}}), [log]({{clotho-train-vggish-vggsound-finetuned-from-ac.log}}) |
| CE - VGGish + VGGSound    | v2t  | {{clotho-train-vggish-vggsound-finetuned-from-ac.geomv2t}} | {{clotho-train-vggish-vggsound-finetuned-from-ac.params}} | [config]({{clotho-train-vggish-vggsound-finetuned-from-ac.config}}), [model]({{clotho-train-vggish-vggsound-finetuned-from-ac.model}}), [log]({{clotho-train-vggish-vggsound-finetuned-from-ac.log}}) |

### Pretraining on SoundDescs, finetuning on CLOTHO

| Experts | Task | R@1 | R@5 | R@10 | R@50 | MdR | MnR | Geom | params | Links |
| ----- | ---- | --- | --- | ---- | ---- | --- | --- | ----- | -- | -- |
| CE - VGGish + VGGSound   | t2v  | {{clotho-train-vggish-vggsound-finetuned-from-sd.geomt2v}} | {{clotho-train-vggish-vggsound-finetuned-from-sd.params}} | [config]({{clotho-train-vggish-vggsound-finetuned-from-sd.config}}), [model]({{clotho-train-vggish-vggsound-finetuned-from-sd.model}}), [log]({{clotho-train-vggish-vggsound-finetuned-from-sd.log}}) |
| CE - VGGish + VGGSound    | v2t  | {{clotho-train-vggish-vggsound-finetuned-from-sd.geomv2t}} | {{clotho-train-vggish-vggsound-finetuned-from-sd.params}} | [config]({{clotho-train-vggish-vggsound-finetuned-from-sd.config}}), [model]({{clotho-train-vggish-vggsound-finetuned-from-sd.model}}), [log]({{clotho-train-vggish-vggsound-finetuned-from-sd.log}}) |

### Pretraining on AudioCaps, finetuning on SoundDescs

| Experts | Task | R@1 | R@5 | R@10 | R@50 | MdR | MnR | Geom | params | Links |
| ----- | ---- | --- | --- | ---- | ---- | --- | --- | ----- | -- | -- |
| CE - VGGish + VGGSound   | t2v  | {{sounddescs-train-vggish-vggsound-finetuned-from-ac.geomt2v}} | {{sounddescs-train-vggish-vggsound-finetuned-from-ac.params}} | [config]({{sounddescs-train-vggish-vggsound-finetuned-from-ac.config}}), [model]({{sounddescs-train-vggish-vggsound-finetuned-from-ac.model}}), [log]({{sounddescs-train-vggish-vggsound-finetuned-from-ac.log}}) |
| CE - VGGish + VGGSound    | v2t  | {{sounddescs-train-vggish-vggsound-finetuned-from-ac.geomv2t}} | {{sounddescs-train-vggish-vggsound-finetuned-from-ac.params}} | [config]({{sounddescs-train-vggish-vggsound-finetuned-from-ac.config}}), [model]({{sounddescs-train-vggish-vggsound-finetuned-from-ac.model}}), [log]({{sounddescs-train-vggish-vggsound-finetuned-from-ac.log}}) |


### Visual centric datasets
| Experts | Task | R@1 | R@5 | R@10 | R@50 | MdR | MnR | Geom | params | Links |
| ----- | ---- | --- | --- | ---- | ---- | --- | --- | ----- | -- | -- |
| CE - VGGish QuerYD  | t2v  | {{querydsegments-train-full-ce-only-audio.geomt2v}} | {{querydsegments-train-full-ce-only-audio.params}} | [config]({{querydsegments-train-full-ce-only-audio.config}}), [model]({{querydsegments-train-full-ce-only-audio.model}}), [log]({{querydsegments-train-full-ce-only-audio.log}}) |
| CE - VGGish QuerYD   | v2t  | {{querydsegments-train-full-ce-only-audio.geomv2t}} | {{querydsegments-train-full-ce-only-audio.params}} | [config]({{querydsegments-train-full-ce-only-audio.config}}), [model]({{querydsegments-train-full-ce-only-audio.model}}), [log]({{querydsegments-train-full-ce-only-audio.log}}) |
| CE - VGGish Activity-Net  | t2v  | {{activity-net-train-full-ce-audio-only.geomt2v}} | {{activity-net-train-full-ce-audio-only.params}} | [config]({{activity-net-train-full-ce-audio-only.config}}), [model]({{activity-net-train-full-ce-audio-only.model}}), [log]({{activity-net-train-full-ce-audio-only.log}}) |
| CE - VGGish Activity-Net   | v2t  | {{activity-net-train-full-ce-audio-only.geomv2t}} | {{activity-net-train-full-ce-audio-only.params}} | [config]({{activity-net-train-full-ce-audio-only.config}}), [model]({{activity-net-train-full-ce-audio-only.model}}), [log]({{activity-net-train-full-ce-audio-only.log}}) |


#### More information can be found at our project page: https://www.robots.ox.ac.uk/~vgg/research/audio-retrieval/


### References
[1] If you find this code useful, please consider citing:
```
@misc{koepke2021audio,
      title={Audio Retrieval with Natural Language Queries: A Benchmark Study}, 
      author={A. Sophia Koepke and Andreea-Maria Oncescu and João F. Henriques and Zeynep Akata and Samuel Albanie},
      year={2021},
      eprint={2112.09418},
      archivePrefix={arXiv},
      primaryClass={eess.AS}
}
``` 

[2] If you find this code useful, please consider citing:
```
@inproceedings{Oncescu21a,
               author       = "Oncescu, A.-M. and Koepke, A.S. and Henriques, J. and Akata, Z., Albanie, S.",
               title        = "Audio Retrieval with Natural Language Queries",
               booktitle    = "INTERSPEECH",
               year         = "2021"
             }
``` 

[3] If you find this code useful, please consider citing:
```
@inproceedings{Liu2019a,
  author    = {Liu, Y. and Albanie, S. and Nagrani, A. and Zisserman, A.},
  booktitle = {arXiv preprint arxiv:1907.13487},
  title     = {Use What You Have: Video retrieval using representations from collaborative experts},
  date      = {2019},
}
```