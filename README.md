# Audio Retrieval with Natural Language Queries: A Benchmark Study


[Paper](https://arxiv.org/pdf/2112.09418.pdf) | [Project page](https://www.robots.ox.ac.uk/~vgg/research/audio-retrieval/) | [Text-to-audio search demo](https://meru.robots.ox.ac.uk/audio-retrieval/)

----

This repository is the implementation of [Audio Retrieval with Natural Language Queries: A Benchmark Study](https://arxiv.org/pdf/2112.09418) which builds on the [Audio Retrieval with Natural Language Queries](https://github.com/oncescuandreea/audio-retrieval) repository and provides code for downloading the SoundDescs dataset and for reproducing all result from [Audio Retrieval with Natural Language Queries: A Benchmark Study](https://arxiv.org/pdf/2112.09418). The code is based on the [Use What You Have: Video retrieval using representations from collaborative experts](https://github.com/albanie/collaborative-experts) and [MMT: Multi-modal Transformer for Video Retrieval](https://github.com/gabeur/mmt) repositories.

The datasets used in this paper are [SoundDescs](https://arxiv.org/pdf/2112.09418.pdf), [AudioCaps](https://www.aclweb.org/anthology/N19-1011.pdf), [CLOTHO](https://arxiv.org/pdf/1910.09387.pdf), [Activity-Net](https://arxiv.org/pdf/1705.00754.pdf) and [QuerYD](https://arxiv.org/pdf/2011.11071.pdf).

## Requirements and datasets
The required libraries for running this code can be found in requirements.txt. Cuda 10.1 and Python 3.7 were used.

```
conda create --name audio-retrieval python=3.7
conda activate audio-retrieval
pip install -r requirements.txt
conda install -c conda-forge ffmpeg
```

To be able to run the code below, features extracted from various datasets need to be downloaded. If there is not enough space in your working location to store some of these features (for SoundDescs and AudioCaps the files are larger than 6GB while the others are under 1GB) then you will need to create a folder called `data` inside this repository which should be a symlink to a folder where enogh memory exists. As an example, run the following from the audio-retrieval-benchmark base directory:
```
ln -s <path-where-data-can-be-saved> data
```
To download features for the AudioCaps, Clotho, Activity-Net, and QuerYD  datasets, follow the steps [here](https://github.com/akoepke/audio-retrieval-benchmark/blob/main/misc/exps-names.md). The SoundDescs features can be downloaded analogously:

```
python3 misc/sync_experts.py --dataset SoundDescs
```

In case you want to use the raw audio data for the SoundDescs, we explain how to download the SoundDescs dataset below.


## SoundDescs dataset download and pre-processing

This is a tool to allow for easy download of audio files and text information from the https://sound-effects.bbcrewind.co.uk/search page.

**Downloading audios** \
First download the download_links_renamed.txt or, if needed, the download_links.txt file. Save it in the folder that will be used for downloading audios. To be able to download the files the --download_folder flag must be set when running the commands below.


To only download a few audio files, use the --limit flag with non-zero values.

To download audio files in zip form for the SoundDescs dataset simply run the line below. To download multiple files at the same time use the processes flag. We recommend not using more than two processes to avoid being blocked by the website.
```
python sounddescs_download_audios.py --download_folder {location where to save files} --processes 2
```

To unzip the audio files to a new folder, run the line below. Here a larger number of processes can be used:
```
python sounddescs_download_audios.py --action unzipping --processes 20 --download_folder {location where to save files}
```

To re-sample the audio files at 16kHz and be put in the format needed to run CE, MoEE, and MMT, run the following command:
```
python sounddescs_wavs_transforms.py --exp resample --initial_folder {location where files were saved before} --dest_folder {location where resampled files are stored} --processes 20
```

Other files available that might prove useful are found in the sounddescs_data folder. The files are:
* categories.pkl - this file contains tags for most audio files. These tags can be Nature, Clocks, Sport etc. Some files have more than one tag and some have no tags.
* descriptions.pkl - this file contains the descriptions associated with the audio files. These are used as captions in our CE, MoEE and MMT experiments.
* extra_info.pkl - this file contains information about the audio content such as file type (e.g. MP3) or sample rate (e.g. 44.1KHz)


**Terms and conditions for SoundDescs dataset**

To download and use the SoundDescs dataset, you need to comply with the terms and conditions of the [RemArc Licence](https://sound-effects.bbcrewind.co.uk/licensing).

This is from the official website that hosts the data:

By continuing, you agree to comply with the terms of the RemArc Licence for this and any future downloads.

Commercial use of this content is not allowed under the RemArc license.

For commercial use, buy the sound effect from Pro Sound Effects which can be found in the More Detail section for each sound effect.


## Evaluating pretrained CE, MoEE, and MMT models on multiple seeds and reproducing results
To reproduce results for the CE, MoEE, and MMT models in the tables below, multiple models trained with different seeds need to be downloaded and evaluated on the test sets.

The steps needed to reproduce results are:
1. Picking the experiment to be reproduced which is in the form `<dataset-name>-<config-file-name>`. Tables with experiment names and the corresponding form can be found in [`misc/exps-names.md`](https://github.com/oncescuandreea/audio-experts/blob/audiocaps/misc/exps-names.md).
2. Downloading the features and splits corresponding to the dataset for which the experiment is run. For example for AudioCaps run:
```
# fetch the pretrained experts for AudioCaps
python3 misc/sync_experts.py --dataset AudioCaps
```

Additional examples for the datasets used in this paper can be found in [`misc/exps-names.md`](https://github.com/oncescuandreea/audio-experts/blob/audiocaps/misc/exps-names.md).

3. Running the `eval.py` script.

For example, to reproduce the experiments for AudioCaps with complete visual and audio experts, run the following line:
```
python eval.py --experiment audiocaps-train-full-ce-r2p1d-inst-vggish-vggsound
```

If the --experiment flag is not provided, the `eval.py` script will download and evaluate all CE and MoEE models on the test set.


### Training a new model

Training a new CE audio-text embedding requires:
1. The pretrained experts for the dataset used for training, which should be located in `<root>/data/<dataset-name>/symlinked-feats` (this will be done automatically by the [utility script](misc/sync_experts.py), or can be done manually). Examples can be found in [`misc/exps-names.md`](https://github.com/oncescuandreea/audio-experts/blob/audiocaps/misc/exps-names.md).
2. A `config.json` file.  You can define your own, or use one of the provided configs in the [configs](configs) directory.

Training is then performed with the following command:
```
python3 train.py --config <path-to-config.json> --device <gpu-id>
```
where `<gpu-id>` is the index of the GPU to train on. This option can be ommitted for training on the CPU.

For example, to train a new embedding for the CLOTHO dataset, run the following sequence of commands:

```
# fetch the pretrained experts for CLOTHO
python3 misc/sync_experts.py --dataset CLOTHO

# Train the model
python3 train.py --config configs/clotho/train-vggish-vggsound.json --device 0
```


To train MMT, use the following command:

```
python -m mmt/train.py --config <path-to-config.json>
```

For example, to train MMT on the CLOTHO dataset, run the following sequence of commands:

```
# fetch the pretrained experts for CLOTHO
python3 misc/sync_experts.py --dataset CLOTHO

# Train MMT on CLOTHO
python -m mmt/train --config mmt/configs/clotho/Clotho_mmt.json
```

### AudioCaps

#### These are the retrieval results obtained for the AudioCaps dataset when using only audio experts:


| Experts | Task | R@1 | R@5 | R@10 | R@50 | MdR | MnR | Geom | params | Links |
| ----- | ---- | --- | --- | ---- | ---- | --- | --- | ----- | -- | -- |
| CE - VGGish   | t2v  | <sub><sup>18.5<sub>(0.3)</sub></sup></sub> | <sub><sup>47.4<sub>(0.1)</sub></sup></sub> | <sub><sup>62.0<sub>(0.5)</sub></sup></sub> | <sub><sup>89.3<sub>(0.3)</sub></sup></sub> | <sub><sup>6.0<sub>(0.0)</sub></sup></sub> | <sub><sup>22.7<sub>(0.3)</sub></sup></sub> | <sub><sup>37.9<sub>(0.1)</sub></sup></sub> | 7.39M | [config](http:/www.robots.ox.ac.uk/~vgg/research/collaborative-experts/data/models/audiocaps-train-full-ce-only-audio/fdc0ced2/seed-0/2021-12-02_00-20-36/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/collaborative-experts/data/models/audiocaps-train-full-ce-only-audio/fdc0ced2/seed-0/2021-12-02_00-20-36/trained_model.pth) |
| CE - VGGish    | v2t  | <sub><sup>20.7<sub>(1.8)</sub></sup></sub> | <sub><sup>48.6<sub>(0.7)</sub></sup></sub> | <sub><sup>62.9<sub>(0.4)</sub></sup></sub> | <sub><sup>86.9<sub>(0.2)</sub></sup></sub> | <sub><sup>6.0<sub>(0.0)</sub></sup></sub> | <sub><sup>25.4<sub>(1.3)</sub></sup></sub> | <sub><sup>39.8<sub>(1.3)</sub></sup></sub> | 7.39M | [config](http:/www.robots.ox.ac.uk/~vgg/research/collaborative-experts/data/models/audiocaps-train-full-ce-only-audio/fdc0ced2/seed-0/2021-12-02_00-20-36/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/collaborative-experts/data/models/audiocaps-train-full-ce-only-audio/fdc0ced2/seed-0/2021-12-02_00-20-36/trained_model.pth) |
| CE - VGGSound    | t2v  | <sub><sup>22.4<sub>(0.3)</sub></sup></sub> | <sub><sup>53.9<sub>(1.2)</sub></sup></sub> | <sub><sup>69.2<sub>(0.9)</sub></sup></sub> | <sub><sup>91.4<sub>(1.6)</sub></sup></sub> | <sub><sup>5.0<sub>(0.0)</sub></sup></sub> | <sub><sup>19.9<sub>(3.4)</sub></sup></sub> | <sub><sup>43.7<sub>(0.5)</sub></sup></sub> | 12.12M | [config](http:/www.robots.ox.ac.uk/~vgg/research/collaborative-experts/data/models/audiocaps-train-only-vggsound/1d130268/seed-0/2021-12-01_23-48-05/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/collaborative-experts/data/models/audiocaps-train-only-vggsound/1d130268/seed-0/2021-12-01_23-48-05/trained_model.pth) |
| CE - VGGSound   | v2t  | <sub><sup>27.0<sub>(0.9)</sub></sup></sub> | <sub><sup>57.8<sub>(0.3)</sub></sup></sub> | <sub><sup>72.5<sub>(0.7)</sub></sup></sub> | <sub><sup>92.6<sub>(0.3)</sub></sup></sub> | <sub><sup>4.0<sub>(0.0)</sub></sup></sub> | <sub><sup>17.5<sub>(1.8)</sub></sup></sub> | <sub><sup>48.3<sub>(0.7)</sub></sup></sub> | 12.12M | [config](http:/www.robots.ox.ac.uk/~vgg/research/collaborative-experts/data/models/audiocaps-train-only-vggsound/1d130268/seed-0/2021-12-01_23-48-05/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/collaborative-experts/data/models/audiocaps-train-only-vggsound/1d130268/seed-0/2021-12-01_23-48-05/trained_model.pth) |
| CE - VGGish + VGGSound    | t2v  | <sub><sup>23.6<sub>(0.6)</sub></sup></sub> | <sub><sup>56.2<sub>(0.5)</sub></sup></sub> | <sub><sup>71.4<sub>(0.5)</sub></sup></sub> | <sub><sup>92.3<sub>(1.5)</sub></sup></sub> | <sub><sup>4.0<sub>(0.0)</sub></sup></sub> | <sub><sup>18.3<sub>(3.0)</sub></sup></sub> | <sub><sup>45.6<sub>(0.5)</sub></sup></sub> | 21.86M | [config](http:/www.robots.ox.ac.uk/~vgg/research/collaborative-experts/data/models/audiocaps-train-vggish-vggsound/78bc4be0/seed-0/2021-12-01_23-01-34/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/collaborative-experts/data/models/audiocaps-train-vggish-vggsound/78bc4be0/seed-0/2021-12-01_23-01-34/trained_model.pth) |
| CE - VGGish + VGGSound   | v2t  | <sub><sup>27.6<sub>(1.0)</sub></sup></sub> | <sub><sup>60.5<sub>(0.7)</sub></sup></sub> | <sub><sup>74.7<sub>(0.8)</sub></sup></sub> | <sub><sup>94.2<sub>(0.4)</sub></sup></sub> | <sub><sup>4.0<sub>(0.0)</sub></sup></sub> | <sub><sup>14.7<sub>(1.4)</sub></sup></sub> | <sub><sup>50.0<sub>(0.6)</sub></sup></sub> | 21.86M | [config](http:/www.robots.ox.ac.uk/~vgg/research/collaborative-experts/data/models/audiocaps-train-vggish-vggsound/78bc4be0/seed-0/2021-12-01_23-01-34/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/collaborative-experts/data/models/audiocaps-train-vggish-vggsound/78bc4be0/seed-0/2021-12-01_23-01-34/trained_model.pth) |
| MoEE - VGGish + VGGSound   | t2v  | <sub><sup>23.0<sub>(0.7)</sub></sup></sub> | <sub><sup>55.7<sub>(0.3)</sub></sup></sub> | <sub><sup>71.0<sub>(1.2)</sub></sup></sub> | <sub><sup>93.0<sub>(0.3)</sub></sup></sub> | <sub><sup>4.0<sub>(0.0)</sub></sup></sub> | <sub><sup>16.3<sub>(0.5)</sub></sup></sub> | <sub><sup>45.0<sub>(0.8)</sub></sup></sub> | 8.90M | [config](http:/www.robots.ox.ac.uk/~vgg/research/collaborative-experts/data/models/audiocaps-train-vggish-vggsound-moee/25c49634/seed-0/2021-12-01_22-20-14/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/collaborative-experts/data/models/audiocaps-train-vggish-vggsound-moee/25c49634/seed-0/2021-12-01_22-20-14/trained_model.pth) |
| MoEE - VGGish + VGGSound   | v2t  | <sub><sup>26.6<sub>(0.7)</sub></sup></sub> | <sub><sup>59.3<sub>(1.4)</sub></sup></sub> | <sub><sup>73.5<sub>(1.1)</sub></sup></sub> | <sub><sup>94.0<sub>(0.5)</sub></sup></sub> | <sub><sup>4.0<sub>(0.0)</sub></sup></sub> | <sub><sup>15.6<sub>(0.8)</sub></sup></sub> | <sub><sup>48.8<sub>(0.8)</sub></sup></sub> | 8.90M | [config](http:/www.robots.ox.ac.uk/~vgg/research/collaborative-experts/data/models/audiocaps-train-vggish-vggsound-moee/25c49634/seed-0/2021-12-01_22-20-14/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/collaborative-experts/data/models/audiocaps-train-vggish-vggsound-moee/25c49634/seed-0/2021-12-01_22-20-14/trained_model.pth) |
| MMT - VGGish + VGGSound    | t2v  | <sub><sup>36.1<sub>(3.3)</sub></sup></sub> | <sub><sup>72.0<sub>(2.9)</sub></sup></sub> | <sub><sup>84.5<sub>(2.0)</sub></sup></sub> | <sub><sup>97.6<sub>(0.4)</sub></sup></sub> | <sub><sup>2.3<sub>(0.6)</sub></sup></sub> | <sub><sup>7.5<sub>(1.3)</sub></sup></sub> | <sub><sup>60.3<sub>(2.8)</sub></sup></sub> | 127.08M | [config](./mmt/configs/AudioCaps_mmt.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/collaborative-experts/data/models/audiocaps-train-vggish-vggsound-mmt/seed-0/trained_model.pth) |
| MMT - VGGish + VGGSound   | v2t  | <sub><sup>39.6<sub>(0.2)</sub></sup></sub> | <sub><sup>76.8<sub>(0.9)</sub></sup></sub> | <sub><sup>86.7<sub>(1.8)</sub></sup></sub> | <sub><sup>98.2<sub>(0.4)</sub></sup></sub> | <sub><sup>2.0<sub>(0.0)</sub></sup></sub> | <sub><sup>6.5<sub>(0.5)</sub></sup></sub> |<sub><sup>64.1<sub>(0.5)</sub></sup></sub>  | 127.08M | [config](./mmt/configs/AudioCaps_mmt.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/collaborative-experts/data/models/audiocaps-train-vggish-vggsound-mmt/seed-0/trained_model.pth) |

#### Using only visual experts for AudioCaps:

| Experts | Task | R@1 | R@5 | R@10 | R@50 | MdR | MnR | Geom | params | Links |
| ----- | ---- | --- | --- | ---- | ---- | --- | --- | ----- | -- | -- |
| CE - Scene   | t2v  | <sub><sup>6.0<sub>(0.0)</sub></sup></sub> | <sub><sup>22.9<sub>(0.5)</sub></sup></sub> | <sub><sup>35.6<sub>(0.8)</sub></sup></sub> | <sub><sup>70.4<sub>(0.6)</sub></sup></sub> | <sub><sup>19.0<sub>(0.0)</sub></sup></sub> | <sub><sup>69.1<sub>(4.6)</sub></sup></sub> | <sub><sup>16.9<sub>(0.3)</sub></sup></sub> | 7.51M | [config](http:/www.robots.ox.ac.uk/~vgg/research/collaborative-experts/data/models/audiocaps-train-full-ce-only-scene/a355cc60/seed-0/2021-12-01_21-51-49/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/collaborative-experts/data/models/audiocaps-train-full-ce-only-scene/a355cc60/seed-0/2021-12-01_21-51-49/trained_model.pth) |
| CE - Scene    | v2t  | <sub><sup>6.8<sub>(0.6)</sub></sup></sub> | <sub><sup>22.1<sub>(0.9)</sub></sup></sub> | <sub><sup>31.9<sub>(1.3)</sub></sup></sub> | <sub><sup>62.9<sub>(0.3)</sub></sup></sub> | <sub><sup>26.3<sub>(1.4)</sub></sup></sub> | <sub><sup>121.3<sub>(6.8)</sub></sup></sub> | <sub><sup>16.9<sub>(0.8)</sub></sup></sub> | 7.51M | [config](http:/www.robots.ox.ac.uk/~vgg/research/collaborative-experts/data/models/audiocaps-train-full-ce-only-scene/a355cc60/seed-0/2021-12-01_21-51-49/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/collaborative-experts/data/models/audiocaps-train-full-ce-only-scene/a355cc60/seed-0/2021-12-01_21-51-49/trained_model.pth) |
| CE - R2P1D   | t2v  | <sub><sup>8.1<sub>(0.4)</sub></sup></sub> | <sub><sup>30.0<sub>(0.4)</sub></sup></sub> | <sub><sup>45.8<sub>(0.2)</sub></sup></sub> | <sub><sup>77.2<sub>(0.9)</sub></sup></sub> | <sub><sup>12.5<sub>(0.5)</sub></sup></sub> | <sub><sup>56.6<sub>(4.6)</sub></sup></sub> | <sub><sup>22.3<sub>(0.5)</sub></sup></sub> | 6.21M | [config](http:/www.robots.ox.ac.uk/~vgg/research/collaborative-experts/data/models/audiocaps-train-full-ce-only-r2p1d/4cee61af/seed-0/2021-12-01_22-02-57/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/collaborative-experts/data/models/audiocaps-train-full-ce-only-r2p1d/4cee61af/seed-0/2021-12-01_22-02-57/trained_model.pth) |
| CE - R2P1D   | v2t  | <sub><sup>10.7<sub>(0.1)</sub></sup></sub> | <sub><sup>30.4<sub>(1.5)</sub></sup></sub> | <sub><sup>43.4<sub>(1.9)</sub></sup></sub> | <sub><sup>75.0<sub>(1.0)</sub></sup></sub> | <sub><sup>14.3<sub>(1.2)</sub></sup></sub> | <sub><sup>78.2<sub>(1.6)</sub></sup></sub> | <sub><sup>24.2<sub>(0.7)</sub></sup></sub> | 6.21M | [config](http:/www.robots.ox.ac.uk/~vgg/research/collaborative-experts/data/models/audiocaps-train-full-ce-only-r2p1d/4cee61af/seed-0/2021-12-01_22-02-57/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/collaborative-experts/data/models/audiocaps-train-full-ce-only-r2p1d/4cee61af/seed-0/2021-12-01_22-02-57/trained_model.pth) |
| CE - Inst    | t2v  | <sub><sup>8.2<sub>(0.3)</sub></sup></sub> | <sub><sup>29.7<sub>(0.5)</sub></sup></sub> | <sub><sup>46.2<sub>(0.5)</sub></sup></sub> | <sub><sup>79.2<sub>(1.3)</sub></sup></sub> | <sub><sup>12.0<sub>(0.0)</sub></sup></sub> | <sub><sup>50.4<sub>(7.3)</sub></sup></sub> | <sub><sup>22.4<sub>(0.4)</sub></sup></sub> | 7.38M | [config](http:/www.robots.ox.ac.uk/~vgg/research/collaborative-experts/data/models/audiocaps-train-full-ce-only-inst/dc36aa69/seed-0/2021-12-01_22-13-22/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/collaborative-experts/data/models/audiocaps-train-full-ce-only-inst/dc36aa69/seed-0/2021-12-01_22-13-22/trained_model.pth) |
| CE - Inst   | v2t  | <sub><sup>10.1<sub>(0.8)</sub></sup></sub> | <sub><sup>28.0<sub>(1.4)</sub></sup></sub> | <sub><sup>41.3<sub>(0.6)</sub></sup></sub> | <sub><sup>75.8<sub>(0.7)</sub></sup></sub> | <sub><sup>15.0<sub>(1.0)</sub></sup></sub> | <sub><sup>85.8<sub>(2.4)</sub></sup></sub> | <sub><sup>22.7<sub>(0.9)</sub></sup></sub> | 7.38M | [config](http:/www.robots.ox.ac.uk/~vgg/research/collaborative-experts/data/models/audiocaps-train-full-ce-only-inst/dc36aa69/seed-0/2021-12-01_22-13-22/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/collaborative-experts/data/models/audiocaps-train-full-ce-only-inst/dc36aa69/seed-0/2021-12-01_22-13-22/trained_model.pth) |
| CE - Scene + R2P1D   | t2v  | <sub><sup>8.6<sub>(0.1)</sub></sup></sub> | <sub><sup>30.9<sub>(0.0)</sub></sup></sub> | <sub><sup>47.4<sub>(0.2)</sub></sup></sub> | <sub><sup>79.1<sub>(0.8)</sub></sup></sub> | <sub><sup>11.3<sub>(0.6)</sub></sup></sub> | <sub><sup>51.2<sub>(3.4)</sub></sup></sub> | <sub><sup>23.3<sub>(0.0)</sub></sup></sub> | 16.07M | [config](http:/www.robots.ox.ac.uk/~vgg/research/collaborative-experts/data/models/audiocaps-train-full-ce-scene-r2p1d/b3e32e97/seed-0/2021-12-01_21-53-10/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/collaborative-experts/data/models/audiocaps-train-full-ce-scene-r2p1d/b3e32e97/seed-0/2021-12-01_21-53-10/trained_model.pth) |
| CE - Scene + R2P1D   | v2t  | <sub><sup>11.6<sub>(0.4)</sub></sup></sub> | <sub><sup>31.5<sub>(0.9)</sub></sup></sub> | <sub><sup>43.5<sub>(0.8)</sub></sup></sub> | <sub><sup>75.8<sub>(0.4)</sub></sup></sub> | <sub><sup>14.8<sub>(0.8)</sub></sup></sub> | <sub><sup>69.9<sub>(2.6)</sub></sup></sub> | <sub><sup>25.1<sub>(0.3)</sub></sup></sub> | 16.07M | [config](http:/www.robots.ox.ac.uk/~vgg/research/collaborative-experts/data/models/audiocaps-train-full-ce-scene-r2p1d/b3e32e97/seed-0/2021-12-01_21-53-10/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/collaborative-experts/data/models/audiocaps-train-full-ce-scene-r2p1d/b3e32e97/seed-0/2021-12-01_21-53-10/trained_model.pth) |
| CE - Scene + Inst   | t2v  | <sub><sup>8.2<sub>(0.3)</sub></sup></sub> | <sub><sup>30.4<sub>(0.3)</sub></sup></sub> | <sub><sup>47.1<sub>(0.2)</sub></sup></sub> | <sub><sup>78.9<sub>(1.8)</sub></sup></sub> | <sub><sup>12.0<sub>(0.0)</sub></sup></sub> | <sub><sup>51.7<sub>(8.8)</sub></sup></sub> | <sub><sup>22.7<sub>(0.3)</sub></sup></sub> | 17.25M | [config](http:/www.robots.ox.ac.uk/~vgg/research/collaborative-experts/data/models/audiocaps-train-full-ce-scene-inst/0bc2f4db/seed-0/2021-12-01_22-05-52/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/collaborative-experts/data/models/audiocaps-train-full-ce-scene-inst/0bc2f4db/seed-0/2021-12-01_22-05-52/trained_model.pth) |
| CE - Scene + Inst   | v2t  | <sub><sup>10.2<sub>(1.2)</sub></sup></sub> | <sub><sup>29.0<sub>(1.5)</sub></sup></sub> | <sub><sup>41.5<sub>(1.3)</sub></sup></sub> | <sub><sup>74.5<sub>(0.2)</sub></sup></sub> | <sub><sup>15.7<sub>(0.6)</sub></sup></sub> | <sub><sup>83.8<sub>(2.9)</sub></sup></sub> | <sub><sup>23.0<sub>(0.6)</sub></sup></sub> | 17.25M | [config](http:/www.robots.ox.ac.uk/~vgg/research/collaborative-experts/data/models/audiocaps-train-full-ce-scene-inst/0bc2f4db/seed-0/2021-12-01_22-05-52/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/collaborative-experts/data/models/audiocaps-train-full-ce-scene-inst/0bc2f4db/seed-0/2021-12-01_22-05-52/trained_model.pth) |
| CE - R2P1D + Inst   | t2v  | <sub><sup>9.5<sub>(0.6)</sub></sup></sub> | <sub><sup>33.0<sub>(1.0)</sub></sup></sub> | <sub><sup>50.0<sub>(0.5)</sub></sup></sub> | <sub><sup>81.1<sub>(0.9)</sub></sup></sub> | <sub><sup>10.3<sub>(0.6)</sub></sup></sub> | <sub><sup>45.9<sub>(3.8)</sub></sup></sub> | <sub><sup>25.0<sub>(0.8)</sub></sup></sub> | 15.95M | [config](http:/www.robots.ox.ac.uk/~vgg/research/collaborative-experts/data/models/audiocaps-train-full-ce-r2p1d-inst/af87d681/seed-0/2021-12-01_22-18-55/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/collaborative-experts/data/models/audiocaps-train-full-ce-r2p1d-inst/af87d681/seed-0/2021-12-01_22-18-55/trained_model.pth) |
| CE - R2P1D + Inst   | v2t  | <sub><sup>11.2<sub>(0.1)</sub></sup></sub> | <sub><sup>31.3<sub>(1.5)</sub></sup></sub> | <sub><sup>45.2<sub>(1.9)</sub></sup></sub> | <sub><sup>77.4<sub>(0.7)</sub></sup></sub> | <sub><sup>13.0<sub>(1.0)</sub></sup></sub> | <sub><sup>68.5<sub>(0.7)</sub></sup></sub> | <sub><sup>25.1<sub>(0.8)</sub></sup></sub> | 15.95M | [config](http:/www.robots.ox.ac.uk/~vgg/research/collaborative-experts/data/models/audiocaps-train-full-ce-r2p1d-inst/af87d681/seed-0/2021-12-01_22-18-55/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/collaborative-experts/data/models/audiocaps-train-full-ce-r2p1d-inst/af87d681/seed-0/2021-12-01_22-18-55/trained_model.pth) |

#### Visual and audio experts for AudioCaps:

| Experts | Task | R@1 | R@5 | R@10 | R@50 | MdR | MnR | Geom | params | Links |
| ----- | ---- | --- | --- | ---- | ---- | --- | --- | ----- | -- | -- |
| CE - R2P1D + Inst + VGGish   | t2v  | <sub><sup>24.5<sub>(0.8)</sub></sup></sub> | <sub><sup>59.0<sub>(0.6)</sub></sup></sub> | <sub><sup>74.9<sub>(1.0)</sub></sup></sub> | <sub><sup>94.5<sub>(0.7)</sub></sup></sub> | <sub><sup>4.0<sub>(0.0)</sub></sup></sub> | <sub><sup>14.3<sub>(1.2)</sub></sup></sub> | <sub><sup>47.6<sub>(0.7)</sub></sup></sub> | 23.32M | [config](http:/www.robots.ox.ac.uk/~vgg/research/collaborative-experts/data/models/audiocaps-train-full-ce-r2p1d-inst-vggish/549ac014/seed-0/2021-12-02_00-03-22/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/collaborative-experts/data/models/audiocaps-train-full-ce-r2p1d-inst-vggish/549ac014/seed-0/2021-12-02_00-03-22/trained_model.pth) |
| CE - R2P1D + Inst + VGGish  | v2t  | <sub><sup>31.0<sub>(2.2)</sub></sup></sub> | <sub><sup>64.5<sub>(1.0)</sub></sup></sub> | <sub><sup>78.8<sub>(1.2)</sub></sup></sub> | <sub><sup>95.5<sub>(0.1)</sub></sup></sub> | <sub><sup>3.0<sub>(0.0)</sub></sup></sub> | <sub><sup>11.4<sub>(0.9)</sub></sup></sub> | <sub><sup>54.0<sub>(1.8)</sub></sup></sub> | 23.32M | [config](http:/www.robots.ox.ac.uk/~vgg/research/collaborative-experts/data/models/audiocaps-train-full-ce-r2p1d-inst-vggish/549ac014/seed-0/2021-12-02_00-03-22/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/collaborative-experts/data/models/audiocaps-train-full-ce-r2p1d-inst-vggish/549ac014/seed-0/2021-12-02_00-03-22/trained_model.pth) |
| CE - R2P1D + Inst + VGGSound   | t2v  | <sub><sup>27.6<sub>(0.2)</sub></sup></sub> | <sub><sup>63.8<sub>(0.6)</sub></sup></sub> | <sub><sup>78.0<sub>(0.8)</sub></sup></sub> | <sub><sup>94.7<sub>(0.1)</sub></sup></sub> | <sub><sup>3.0<sub>(0.0)</sub></sup></sub> | <sub><sup>13.4<sub>(0.8)</sub></sup></sub> | <sub><sup>51.6<sub>(0.2)</sub></sup></sub> | 28.05M | [config](http:/www.robots.ox.ac.uk/~vgg/research/collaborative-experts/data/models/audiocaps-train-full-ce-r2p1d-inst-vggsound/57e6009b/seed-0/2021-12-01_23-05-14/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/collaborative-experts/data/models/audiocaps-train-full-ce-r2p1d-inst-vggsound/57e6009b/seed-0/2021-12-01_23-05-14/trained_model.pth) |
| CE - R2P1D + Inst + VGGSound  | v2t  | <sub><sup>32.7<sub>(0.9)</sub></sup></sub> | <sub><sup>69.2<sub>(1.0)</sub></sup></sub> | <sub><sup>82.4<sub>(0.4)</sub></sup></sub> | <sub><sup>96.8<sub>(0.3)</sub></sup></sub> | <sub><sup>2.8<sub>(0.3)</sub></sup></sub> | <sub><sup>9.3<sub>(0.2)</sub></sup></sub> | <sub><sup>57.1<sub>(0.7)</sub></sup></sub> | 28.05M | [config](http:/www.robots.ox.ac.uk/~vgg/research/collaborative-experts/data/models/audiocaps-train-full-ce-r2p1d-inst-vggsound/57e6009b/seed-0/2021-12-01_23-05-14/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/collaborative-experts/data/models/audiocaps-train-full-ce-r2p1d-inst-vggsound/57e6009b/seed-0/2021-12-01_23-05-14/trained_model.pth) |
| CE - R2P1D + Inst +VGGish + VGGSound  | t2v  | <sub><sup>28.0<sub>(0.5)</sub></sup></sub> | <sub><sup>65.3<sub>(0.7)</sub></sup></sub> | <sub><sup>80.4<sub>(0.3)</sub></sup></sub> | <sub><sup>96.0<sub>(0.5)</sub></sup></sub> | <sub><sup>3.0<sub>(0.0)</sub></sup></sub> | <sub><sup>10.8<sub>(0.5)</sub></sup></sub> | <sub><sup>52.8<sub>(0.4)</sub></sup></sub> | 35.43M | [config](http:/www.robots.ox.ac.uk/~vgg/research/collaborative-experts/data/models/audiocaps-train-full-ce-r2p1d-inst-vggish-vggsound/ea6e4cc5/seed-0/2021-12-01_23-31-50/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/collaborative-experts/data/models/audiocaps-train-full-ce-r2p1d-inst-vggish-vggsound/ea6e4cc5/seed-0/2021-12-01_23-31-50/trained_model.pth) |
| CE - R2P1D + Inst +VGGish + VGGSound  | v2t  | <sub><sup>35.8<sub>(0.6)</sub></sup></sub> | <sub><sup>70.2<sub>(1.6)</sub></sup></sub> | <sub><sup>83.3<sub>(0.6)</sub></sup></sub> | <sub><sup>98.3<sub>(0.4)</sub></sup></sub> | <sub><sup>2.0<sub>(0.0)</sub></sup></sub> | <sub><sup>7.8<sub>(0.5)</sub></sup></sub> | <sub><sup>59.4<sub>(0.4)</sub></sup></sub> | 35.43M | [config](http:/www.robots.ox.ac.uk/~vgg/research/collaborative-experts/data/models/audiocaps-train-full-ce-r2p1d-inst-vggish-vggsound/ea6e4cc5/seed-0/2021-12-01_23-31-50/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/collaborative-experts/data/models/audiocaps-train-full-ce-r2p1d-inst-vggish-vggsound/ea6e4cc5/seed-0/2021-12-01_23-31-50/trained_model.pth) |

### CLOTHO

| Experts | Task | R@1 | R@5 | R@10 | R@50 | MdR | MnR | Geom | params | Links |
| ----- | ---- | --- | --- | ---- | ---- | --- | --- | ----- | -- | -- |
| CE - VGGish   | t2v  | <sub><sup>4.0<sub>(0.2)</sub></sup></sub> | <sub><sup>15.0<sub>(0.9)</sub></sup></sub> | <sub><sup>25.4<sub>(0.5)</sub></sup></sub> | <sub><sup>61.4<sub>(1.1)</sub></sup></sub> | <sub><sup>31.7<sub>(1.5)</sub></sup></sub> | <sub><sup>78.2<sub>(2.2)</sub></sup></sub> | <sub><sup>11.5<sub>(0.5)</sub></sup></sub> | 7.39M | [config](http:/www.robots.ox.ac.uk/~vgg/research/collaborative-experts/data/models/clotho-train-full-ce-only-audio/4f58ef05/seed-0/2021-06-10_15-38-28/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/collaborative-experts/data/models/clotho-train-full-ce-only-audio/4f58ef05/seed-0/2021-06-10_15-38-28/trained_model.pth) |
| CE - VGGish   | v2t  | <sub><sup>4.8<sub>(0.4)</sub></sup></sub> | <sub><sup>15.9<sub>(1.8)</sub></sup></sub> | <sub><sup>25.8<sub>(1.7)</sub></sup></sub> | <sub><sup>57.5<sub>(2.5)</sub></sup></sub> | <sub><sup>35.7<sub>(2.5)</sub></sup></sub> | <sub><sup>106.6<sub>(5.7)</sub></sup></sub> | <sub><sup>12.5<sub>(1.0)</sub></sup></sub> | 7.39M | [config](http:/www.robots.ox.ac.uk/~vgg/research/collaborative-experts/data/models/clotho-train-full-ce-only-audio/4f58ef05/seed-0/2021-06-10_15-38-28/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/collaborative-experts/data/models/clotho-train-full-ce-only-audio/4f58ef05/seed-0/2021-06-10_15-38-28/trained_model.pth) |
| CE - VGGish + VGGSound    | t2v  | <sub><sup>6.7<sub>(0.4)</sub></sup></sub> | <sub><sup>21.6<sub>(0.6)</sub></sup></sub> | <sub><sup>33.2<sub>(0.3)</sub></sup></sub> | <sub><sup>69.8<sub>(0.3)</sub></sup></sub> | <sub><sup>22.3<sub>(0.6)</sub></sup></sub> | <sub><sup>58.3<sub>(1.1)</sub></sup></sub> | <sub><sup>16.9<sub>(0.2)</sub></sup></sub> | 21.86M | [config](http:/www.robots.ox.ac.uk/~vgg/research/collaborative-experts/data/models/clotho-train-vggish-vggsound/dec0c820/seed-0/2021-06-10_14-45-51/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/collaborative-experts/data/models/clotho-train-vggish-vggsound/dec0c820/seed-0/2021-06-10_14-45-51/trained_model.pth) |
| CE - VGGish + VGGSound   | v2t  | <sub><sup>7.0<sub>(0.3)</sub></sup></sub> | <sub><sup>22.7<sub>(0.6)</sub></sup></sub> | <sub><sup>34.6<sub>(0.5)</sub></sup></sub> | <sub><sup>67.9<sub>(2.3)</sub></sup></sub> | <sub><sup>21.3<sub>(0.6)</sub></sup></sub> | <sub><sup>72.6<sub>(3.4)</sub></sup></sub> | <sub><sup>17.7<sub>(0.3)</sub></sup></sub> | 21.86M | [config](http:/www.robots.ox.ac.uk/~vgg/research/collaborative-experts/data/models/clotho-train-vggish-vggsound/dec0c820/seed-0/2021-06-10_14-45-51/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/collaborative-experts/data/models/clotho-train-vggish-vggsound/dec0c820/seed-0/2021-06-10_14-45-51/trained_model.pth) |
| MoEE - VGGish + VGGSound   | t2v  | <sub><sup>6.0<sub>(0.1)</sub></sup></sub> | <sub><sup>20.8<sub>(0.7)</sub></sup></sub> | <sub><sup>32.3<sub>(0.3)</sub></sup></sub> | <sub><sup>68.5<sub>(0.5)</sub></sup></sub> | <sub><sup>23.0<sub>(0.0)</sub></sup></sub> | <sub><sup>60.2<sub>(0.8)</sub></sup></sub> | <sub><sup>16.0<sub>(0.3)</sub></sup></sub> | 8.90M | [config](http:/www.robots.ox.ac.uk/~vgg/research/collaborative-experts/data/models/clotho-train-vggish-vggsound-moee/fafa3e91/seed-0/2021-06-10_14-44-51/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/collaborative-experts/data/models/clotho-train-vggish-vggsound-moee/fafa3e91/seed-0/2021-06-10_14-44-51/trained_model.pth) |
| MoEE - VGGish + VGGSound   | v2t  | <sub><sup>7.2<sub>(0.5)</sub></sup></sub> | <sub><sup>22.1<sub>(0.7)</sub></sup></sub> | <sub><sup>33.2<sub>(1.1)</sub></sup></sub> | <sub><sup>67.4<sub>(0.3)</sub></sup></sub> | <sub><sup>22.7<sub>(0.6)</sub></sup></sub> | <sub><sup>71.8<sub>(2.3)</sub></sup></sub> | <sub><sup>17.4<sub>(0.7)</sub></sup></sub> | 8.90M | [config](http:/www.robots.ox.ac.uk/~vgg/research/collaborative-experts/data/models/clotho-train-vggish-vggsound-moee/fafa3e91/seed-0/2021-06-10_14-44-51/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/collaborative-experts/data/models/clotho-train-vggish-vggsound-moee/fafa3e91/seed-0/2021-06-10_14-44-51/trained_model.pth) |
| MMT - VGGish + VGGSound    | t2v  | <sub><sup>6.5<sub>(0.6)</sub></sup></sub> | <sub><sup>21.6<sub>(0.7)</sub></sup></sub> | <sub><sup>32.8<sub>(2.1)</sub></sup></sub> | <sub><sup>66.9<sub>(2.0)</sub></sup></sub> | <sub><sup>23.0<sub>(2.6)</sub></sup></sub> | <sub><sup>67.7<sub>(3.1)</sub></sup></sub> | <sub><sup>16.6<sub>(1.1)</sub></sup></sub> | 127.08M | [config](./mmt/configs/Clotho_mmt.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/collaborative-experts/data/models/clotho-train-vggish-vggsound-mmt/seed-0/trained_model.pth) |
| MMT - VGGish + VGGSound   | v2t  | <sub><sup>6.3<sub>(0.5)</sub></sup></sub> | <sub><sup>22.8<sub>(1.7)</sub></sup></sub> | <sub><sup>33.3<sub>(2.2)</sub></sup></sub> | <sub><sup>67.8<sub>(1.5)</sub></sup></sub> | <sub><sup>22.3<sub>(1.5)</sub></sup></sub> | <sub><sup>67.3<sub>(2.9)</sub></sup></sub> |<sub><sup>16.8<sub>(1.0)</sub></sup></sub>  | 127.08M | [config](./mmt/configs/Clotho_mmt.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/collaborative-experts/data/models/clotho-train-vggish-vggsound-mmt/seed-0/trained_model.pth) |

### SoundDescs

| Experts | Task | R@1 | R@5 | R@10 | R@50 | MdR | MnR | Geom | params | Links |
| ----- | ---- | --- | --- | ---- | ---- | --- | --- | ----- | -- | -- |
| CE - VGGish   | t2v  | <sub><sup>25.4<sub>(0.6)</sub></sup></sub> | <sub><sup>53.3<sub>(0.3)</sub></sup></sub> | <sub><sup>64.1<sub>(0.3)</sub></sup></sub> | <sub><sup>81.7<sub>(0.4)</sub></sup></sub> | <sub><sup>4.7<sub>(0.6)</sub></sup></sub> | <sub><sup>83.7<sub>(1.9)</sub></sup></sub> | <sub><sup>44.3<sub>(0.3)</sub></sup></sub> | 7.39M | [config](http:/www.robots.ox.ac.uk/~vgg/research/collaborative-experts/data/models/sounddescs-train-full-ce-only-audio/ad6e4888/seed-0/2021-10-02_11-36-50/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/collaborative-experts/data/models/sounddescs-train-full-ce-only-audio/ad6e4888/seed-0/2021-10-02_11-36-50/trained_model.pth) |
| CE - VGGish   | v2t  | <sub><sup>24.2<sub>(0.3)</sub></sup></sub> | <sub><sup>52.3<sub>(0.3)</sub></sup></sub> | <sub><sup>62.5<sub>(0.2)</sub></sup></sub> | <sub><sup>80.9<sub>(0.3)</sub></sup></sub> | <sub><sup>5.0<sub>(0.0)</sub></sup></sub> | <sub><sup>83.6<sub>(1.1)</sub></sup></sub> | <sub><sup>42.9<sub>(0.3)</sub></sup></sub> | 7.39M | [config](http:/www.robots.ox.ac.uk/~vgg/research/collaborative-experts/data/models/sounddescs-train-full-ce-only-audio/ad6e4888/seed-0/2021-10-02_11-36-50/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/collaborative-experts/data/models/sounddescs-train-full-ce-only-audio/ad6e4888/seed-0/2021-10-02_11-36-50/trained_model.pth) |
| CE - VGGish + VGGSound    | t2v  | <sub><sup>31.1<sub>(0.2)</sub></sup></sub> | <sub><sup>60.6<sub>(0.7)</sub></sup></sub> | <sub><sup>70.8<sub>(0.5)</sub></sup></sub> | <sub><sup>86.0<sub>(0.2)</sub></sup></sub> | <sub><sup>3.0<sub>(0.0)</sub></sup></sub> | <sub><sup>63.6<sub>(2.2)</sub></sup></sub> | <sub><sup>51.1<sub>(0.4)</sub></sup></sub> | 21.86M | [config](http:/www.robots.ox.ac.uk/~vgg/research/collaborative-experts/data/models/sounddescs-train-vggish-vggsound/11e28f70/seed-0/2021-12-01_17-25-26/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/collaborative-experts/data/models/sounddescs-train-vggish-vggsound/11e28f70/seed-0/2021-12-01_17-25-26/trained_model.pth) |
| CE - VGGish + VGGSound   | v2t  | <sub><sup>30.8<sub>(0.8)</sub></sup></sub> | <sub><sup>60.3<sub>(0.3)</sub></sup></sub> | <sub><sup>69.5<sub>(0.1)</sub></sup></sub> | <sub><sup>85.4<sub>(0.2)</sub></sup></sub> | <sub><sup>3.0<sub>(0.0)</sub></sup></sub> | <sub><sup>63.2<sub>(0.6)</sub></sup></sub> | <sub><sup>50.5<sub>(0.4)</sub></sup></sub> | 21.86M | [config](http:/www.robots.ox.ac.uk/~vgg/research/collaborative-experts/data/models/sounddescs-train-vggish-vggsound/11e28f70/seed-0/2021-12-01_17-25-26/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/collaborative-experts/data/models/sounddescs-train-vggish-vggsound/11e28f70/seed-0/2021-12-01_17-25-26/trained_model.pth) |
| MoEE - VGGish + VGGSound   | t2v  | <sub><sup>30.8<sub>(0.7)</sub></sup></sub> | <sub><sup>60.8<sub>(0.3)</sub></sup></sub> | <sub><sup>70.9<sub>(0.5)</sub></sup></sub> | <sub><sup>85.9<sub>(0.6)</sub></sup></sub> | <sub><sup>3.0<sub>(0.0)</sub></sup></sub> | <sub><sup>62.0<sub>(3.8)</sub></sup></sub> | <sub><sup>51.0<sub>(0.6)</sub></sup></sub> | 8.90M | [config](http:/www.robots.ox.ac.uk/~vgg/research/collaborative-experts/data/models/sounddescs-train-vggish-vggsound-moee/9ec0faac/seed-0/2021-12-17_17-26-15/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/collaborative-experts/data/models/sounddescs-train-vggish-vggsound-moee/9ec0faac/seed-0/2021-12-17_17-26-15/trained_model.pth) |
| MoEE - VGGish + VGGSound   | v2t  | <sub><sup>30.9<sub>(0.3)</sub></sup></sub> | <sub><sup>60.3<sub>(0.4)</sub></sup></sub> | <sub><sup>70.1<sub>(0.3)</sub></sup></sub> | <sub><sup>85.3<sub>(0.6)</sub></sup></sub> | <sub><sup>3.0<sub>(0.0)</sub></sup></sub> | <sub><sup>61.5<sub>(3.2)</sub></sup></sub> | <sub><sup>50.7<sub>(0.3)</sub></sup></sub> | 8.90M | [config](http:/www.robots.ox.ac.uk/~vgg/research/collaborative-experts/data/models/sounddescs-train-vggish-vggsound-moee/9ec0faac/seed-0/2021-12-17_17-26-15/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/collaborative-experts/data/models/sounddescs-train-vggish-vggsound-moee/9ec0faac/seed-0/2021-12-17_17-26-15/trained_model.pth) |
| MMT - VGGish + VGGSound    | t2v  | <sub><sup>30.7<sub>(0.4)</sub></sup></sub> | <sub><sup>61.8<sub>(1.0)</sub></sup></sub> | <sub><sup>72.2<sub>(0.8)</sub></sup></sub> | <sub><sup>88.8<sub>(0.4)</sub></sup></sub> | <sub><sup>3.0<sub>(0.0)</sub></sup></sub> | <sub><sup>34.0<sub>(0.6)</sub></sup></sub> |<sub><sup>51.5<sub>(0.5)</sub></sup></sub>  | 127.08M | [config](./mmt/configs/SoundDescs_mmt.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/collaborative-experts/data/models/sounddescs-train-vggish-vggsound-mmt/seed-0/trained_model.pth) |
| MMT - VGGish + VGGSound   | v2t  | <sub><sup>31.4<sub>(0.8)</sub></sup></sub> | <sub><sup>63.2<sub>(0.7)</sub></sup></sub> | <sub><sup>73.4<sub>(0.5)</sub></sup></sub> | <sub><sup>89.0<sub>(0.3)</sub></sup></sub> | <sub><sup>3.0<sub>(0.0)</sub></sup></sub> | <sub><sup>32.5<sub>(0.4)</sub></sup></sub> | <sub><sup>52.6<sub>(0.7)</sub></sup></sub> | 127.08M | [config](./mmt/configs/SoundDescs_mmt.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/collaborative-experts/data/models/sounddescs-train-vggish-vggsound-mmt/seed-0/trained_model.pth) |

### Pretraining on SoundDescs, finetuning on AudioCaps

| Experts | Task | R@1 | R@5 | R@10 | R@50 | MdR | MnR | Geom | params | Links |
| ----- | ---- | --- | --- | ---- | ---- | --- | --- | ----- | -- | -- |
| CE - VGGish + VGGSound   | t2v  | <sub><sup>23.3<sub>(0.7)</sub></sup></sub> | <sub><sup>52.2<sub>(0.1)</sub></sup></sub> | <sub><sup>63.9<sub>(0.5)</sub></sup></sub> | <sub><sup>84.3<sub>(0.3)</sub></sup></sub> | <sub><sup>5.0<sub>(0.0)</sub></sup></sub> | <sub><sup>59.9<sub>(1.6)</sub></sup></sub> | <sub><sup>42.7<sub>(0.5)</sub></sup></sub> | 21.86M | [config](http:/www.robots.ox.ac.uk/~vgg/research/collaborative-experts/data/models/audiocaps-train-vggish-vggsound-finetuned-from-sd/9ba6b61b/seed-0/2022-01-06_13-50-19/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/collaborative-experts/data/models/audiocaps-train-vggish-vggsound-finetuned-from-sd/9ba6b61b/seed-0/2022-01-06_13-50-19/trained_model.pth) |
| CE - VGGish + VGGSound    | v2t  | <sub><sup>22.2<sub>(0.4)</sub></sup></sub> | <sub><sup>51.7<sub>(0.3)</sub></sup></sub> | <sub><sup>63.3<sub>(0.3)</sub></sup></sub> | <sub><sup>83.8<sub>(0.4)</sub></sup></sub> | <sub><sup>5.0<sub>(0.0)</sub></sup></sub> | <sub><sup>59.2<sub>(0.5)</sub></sup></sub> | <sub><sup>41.7<sub>(0.2)</sub></sup></sub> | 21.86M | [config](http:/www.robots.ox.ac.uk/~vgg/research/collaborative-experts/data/models/audiocaps-train-vggish-vggsound-finetuned-from-sd/9ba6b61b/seed-0/2022-01-06_13-50-19/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/collaborative-experts/data/models/audiocaps-train-vggish-vggsound-finetuned-from-sd/9ba6b61b/seed-0/2022-01-06_13-50-19/trained_model.pth) |

### Pretraining on AudioCaps, finetuning on CLOTHO

| Experts | Task | R@1 | R@5 | R@10 | R@50 | MdR | MnR | Geom | params | Links |
| ----- | ---- | --- | --- | ---- | ---- | --- | --- | ----- | -- | -- |
| CE - VGGish + VGGSound   | t2v  | <sub><sup>9.1<sub>(0.3)</sub></sup></sub> | <sub><sup>27.4<sub>(0.1)</sub></sup></sub> | <sub><sup>39.7<sub>(0.4)</sub></sup></sub> | <sub><sup>75.0<sub>(0.4)</sub></sup></sub> | <sub><sup>17.0<sub>(0.0)</sub></sup></sub> | <sub><sup>48.6<sub>(0.7)</sub></sup></sub> | <sub><sup>21.5<sub>(0.1)</sub></sup></sub> | 21.86M | [config](http:/www.robots.ox.ac.uk/~vgg/research/collaborative-experts/data/models/clotho-train-vggish-vggsound-finetuned-from-ac/f41f2af5/seed-0/2021-12-14_17-45-11/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/collaborative-experts/data/models/clotho-train-vggish-vggsound-finetuned-from-ac/f41f2af5/seed-0/2021-12-14_17-45-11/trained_model.pth) |
| CE - VGGish + VGGSound    | v2t  | <sub><sup>11.1<sub>(1.1)</sub></sup></sub> | <sub><sup>26.9<sub>(0.7)</sub></sup></sub> | <sub><sup>39.6<sub>(1.1)</sub></sup></sub> | <sub><sup>73.7<sub>(0.6)</sub></sup></sub> | <sub><sup>16.3<sub>(0.6)</sub></sup></sub> | <sub><sup>57.4<sub>(1.8)</sub></sup></sub> | <sub><sup>22.8<sub>(1.2)</sub></sup></sub> | 21.86M | [config](http:/www.robots.ox.ac.uk/~vgg/research/collaborative-experts/data/models/clotho-train-vggish-vggsound-finetuned-from-ac/f41f2af5/seed-0/2021-12-14_17-45-11/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/collaborative-experts/data/models/clotho-train-vggish-vggsound-finetuned-from-ac/f41f2af5/seed-0/2021-12-14_17-45-11/trained_model.pth) |

### Pretraining on SoundDescs, finetuning on CLOTHO

| Experts | Task | R@1 | R@5 | R@10 | R@50 | MdR | MnR | Geom | params | Links |
| ----- | ---- | --- | --- | ---- | ---- | --- | --- | ----- | -- | -- |
| CE - VGGish + VGGSound   | t2v  | <sub><sup>6.4<sub>(0.5)</sub></sup></sub> | <sub><sup>21.1<sub>(1.2)</sub></sup></sub> | <sub><sup>32.5<sub>(1.7)</sub></sup></sub> | <sub><sup>69.3<sub>(1.4)</sub></sup></sub> | <sub><sup>22.7<sub>(1.5)</sub></sup></sub> | <sub><sup>57.6<sub>(2.3)</sub></sup></sub> | <sub><sup>16.3<sub>(1.0)</sub></sup></sub> | 21.86M | [config](http:/www.robots.ox.ac.uk/~vgg/research/collaborative-experts/data/models/clotho-train-vggish-vggsound-finetuned-from-sd/94b46215/seed-0/2022-01-06_13-41-01/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/collaborative-experts/data/models/clotho-train-vggish-vggsound-finetuned-from-sd/94b46215/seed-0/2022-01-06_13-41-01/trained_model.pth) |
| CE - VGGish + VGGSound    | v2t  | <sub><sup>6.1<sub>(0.7)</sub></sup></sub> | <sub><sup>20.1<sub>(1.7)</sub></sup></sub> | <sub><sup>31.4<sub>(1.8)</sub></sup></sub> | <sub><sup>65.9<sub>(2.0)</sub></sup></sub> | <sub><sup>24.7<sub>(1.5)</sub></sup></sub> | <sub><sup>78.1<sub>(5.3)</sub></sup></sub> | <sub><sup>15.7<sub>(1.3)</sub></sup></sub> | 21.86M | [config](http:/www.robots.ox.ac.uk/~vgg/research/collaborative-experts/data/models/clotho-train-vggish-vggsound-finetuned-from-sd/94b46215/seed-0/2022-01-06_13-41-01/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/collaborative-experts/data/models/clotho-train-vggish-vggsound-finetuned-from-sd/94b46215/seed-0/2022-01-06_13-41-01/trained_model.pth) |

### Pretraining on AudioCaps, finetuning on SoundDescs

| Experts | Task | R@1 | R@5 | R@10 | R@50 | MdR | MnR | Geom | params | Links |
| ----- | ---- | --- | --- | ---- | ---- | --- | --- | ----- | -- | -- |
| CE - VGGish + VGGSound   | t2v  | <sub><sup>23.3<sub>(0.7)</sub></sup></sub> | <sub><sup>52.2<sub>(0.1)</sub></sup></sub> | <sub><sup>63.9<sub>(0.5)</sub></sup></sub> | <sub><sup>84.3<sub>(0.3)</sub></sup></sub> | <sub><sup>5.0<sub>(0.0)</sub></sup></sub> | <sub><sup>59.9<sub>(1.6)</sub></sup></sub> | <sub><sup>42.7<sub>(0.5)</sub></sup></sub> | 21.86M | [config](http:/www.robots.ox.ac.uk/~vgg/research/collaborative-experts/data/models/sounddescs-train-vggish-vggsound-finetuned-from-ac/eb171dab/seed-0/2021-12-14_22-36-01/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/collaborative-experts/data/models/sounddescs-train-vggish-vggsound-finetuned-from-ac/eb171dab/seed-0/2021-12-14_22-36-01/trained_model.pth) |
| CE - VGGish + VGGSound    | v2t  | <sub><sup>22.2<sub>(0.4)</sub></sup></sub> | <sub><sup>51.7<sub>(0.3)</sub></sup></sub> | <sub><sup>63.3<sub>(0.3)</sub></sup></sub> | <sub><sup>83.8<sub>(0.4)</sub></sup></sub> | <sub><sup>5.0<sub>(0.0)</sub></sup></sub> | <sub><sup>59.2<sub>(1.3)</sub></sup></sub> | <sub><sup>41.7<sub>(0.2)</sub></sup></sub> | 21.86M | [config](http:/www.robots.ox.ac.uk/~vgg/research/collaborative-experts/data/models/sounddescs-train-vggish-vggsound-finetuned-from-ac/eb171dab/seed-0/2021-12-14_22-36-01/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/collaborative-experts/data/models/sounddescs-train-vggish-vggsound-finetuned-from-ac/eb171dab/seed-0/2021-12-14_22-36-01/trained_model.pth) |


### Visual centric datasets
| Experts | Task | R@1 | R@5 | R@10 | R@50 | MdR | MnR | Geom | params | Links |
| ----- | ---- | --- | --- | ---- | ---- | --- | --- | ----- | -- | -- |
| CE - VGGish QuerYD  | t2v  | <sub><sup>3.7<sub>(0.2)</sub></sup></sub> | <sub><sup>11.7<sub>(0.4)</sub></sup></sub> | <sub><sup>17.3<sub>(0.6)</sub></sup></sub> | <sub><sup>36.3<sub>(0.3)</sub></sup></sub> | <sub><sup>115.5<sub>(5.2)</sub></sup></sub> | <sub><sup>273.5<sub>(6.7)</sub></sup></sub> | <sub><sup>9.1<sub>(0.0)</sub></sup></sub> | 7.39M | [config](http:/www.robots.ox.ac.uk/~vgg/research/collaborative-experts/data/models/querydsegments-train-full-ce-only-audio/70111434/seed-0/2021-06-10_14-33-03/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/collaborative-experts/data/models/querydsegments-train-full-ce-only-audio/70111434/seed-0/2021-06-10_14-33-03/trained_model.pth) |
| CE - VGGish QuerYD   | v2t  | <sub><sup>3.8<sub>(0.2)</sub></sup></sub> | <sub><sup>11.5<sub>(0.4)</sub></sup></sub> | <sub><sup>16.8<sub>(0.2)</sub></sup></sub> | <sub><sup>35.2<sub>(0.4)</sub></sup></sub> | <sub><sup>116.3<sub>(2.1)</sub></sup></sub> | <sub><sup>271.9<sub>(5.8)</sub></sup></sub> | <sub><sup>9.0<sub>(0.2)</sub></sup></sub> | 7.39M | [config](http:/www.robots.ox.ac.uk/~vgg/research/collaborative-experts/data/models/querydsegments-train-full-ce-only-audio/70111434/seed-0/2021-06-10_14-33-03/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/collaborative-experts/data/models/querydsegments-train-full-ce-only-audio/70111434/seed-0/2021-06-10_14-33-03/trained_model.pth) |
| CE - VGGish Activity-Net  | t2v  | <sub><sup>1.4<sub>(0.1)</sub></sup></sub> | <sub><sup>5.0<sub>(0.1)</sub></sup></sub> | <sub><sup>8.5<sub>(0.2)</sub></sup></sub> | <sub><sup>22.1<sub>(0.9)</sub></sup></sub> | <sub><sup>312.0<sub>(25.6)</sub></sup></sub> | <sub><sup>765.6<sub>(35.8)</sub></sup></sub> | <sub><sup>3.9<sub>(0.1)</sub></sup></sub> | 7.39M | [config](http:/www.robots.ox.ac.uk/~vgg/research/collaborative-experts/data/models/activity-net-train-full-ce-audio-only/e8639db7/seed-0/2021-06-11_12-23-42/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/collaborative-experts/data/models/activity-net-train-full-ce-audio-only/e8639db7/seed-0/2021-06-11_12-23-42/trained_model.pth) |
| CE - VGGish Activity-Net   | v2t  | <sub><sup>1.1<sub>(0.1)</sub></sup></sub> | <sub><sup>4.5<sub>(0.1)</sub></sup></sub> | <sub><sup>7.9<sub>(0.0)</sub></sup></sub> | <sub><sup>21.6<sub>(0.8)</sub></sup></sub> | <sub><sup>306.3<sub>(27.1)</sub></sup></sub> | <sub><sup>781.7<sub>(30.6)</sub></sup></sub> | <sub><sup>3.4<sub>(0.1)</sub></sup></sub> | 7.39M | [config](http:/www.robots.ox.ac.uk/~vgg/research/collaborative-experts/data/models/activity-net-train-full-ce-audio-only/e8639db7/seed-0/2021-06-11_12-23-42/config.json), [model](http:/www.robots.ox.ac.uk/~vgg/research/collaborative-experts/data/models/activity-net-train-full-ce-audio-only/e8639db7/seed-0/2021-06-11_12-23-42/trained_model.pth) |



#### More information can be found at our project page: https://www.robots.ox.ac.uk/~vgg/research/audio-retrieval/


### References

If you find this code useful, please consider citing [1,2,3,4].

[1]
```
@inproceedings{Koepke2021,
    author    = {Koepke, A.S. and Oncescu, A.-M. and Henriques, J. and Akata, Z. and Albanie, S.},
    title     = {Audio Retrieval with Natural Language Queries: A Benchmark Study},
    booktitle = {arXiv preprint arXiv:2112.09418},
    year      = {2021}
}
```

[2]
```
@inproceedings{Oncescu21a,
    author    = {Oncescu, A.-M. and Koepke, A.S. and Henriques, J. and Akata, Z., Albanie, S.},
    title     = {Audio Retrieval with Natural Language Queries},
    booktitle = {INTERSPEECH},
    year      = {2021}
}
```

[3]
```
@inproceedings{Liu2019a,
    author    = {Liu, Y. and Albanie, S. and Nagrani, A. and Zisserman, A.},
    title     = {Use What You Have: Video retrieval using representations from collaborative experts},
    booktitle = {British Machine Vision Conference (BMVC)},
    year      = {2019},
}
```

[4]
```
@inproceedings{gabeur2020mmt,
    author    = {Gabeur, V. and Sun, C. and Alahari, K. and Schmid, C.},
    title     = {Multi-modal Transformer for Video Retrieval},
    booktitle = {European Conference on Computer Vision (ECCV)},
    year      = {2020}
}
```
