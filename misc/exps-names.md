## This file contains additional instructions for running the commands provided in the main README file

### Downloading required features and splits:
```
python3 misc/sync_experts.py --dataset AudioCaps
python3 misc/sync_experts.py --dataset CLOTHO
python3 misc/sync_experts.py --dataset activity-net
python3 misc/sync_experts.py --dataset QuerYDSegments
python3 misc/sync_experts.py --dataset SoundDescs
```

### Finding the corresponding .json file names for evaluation of pre-trained models

#### AudioCaps:
|Experiment type | Model name|
|---|---|
|CE VGGish only | audiocaps-train-full-ce-only-audio|
|CE VGGSound only | audicaps-train-only-vggsound|
|CE VGGish + VGGSound | audiocaps-train-vggish-vggsound|
|MoEE VGGish + VGGSound | audiocaps-train-vggish-vggsound-moee|
|MMT VGGish + VGGSound | audiocaps-train-vggish-vggsound-mmtVy|

|CE Scene | audiocaps-train-full-ce-only-scene|
|CE R2P1D | audiocaps-train-full-ce-only-r2p1d|
|CE Inst | audiocaps-train-full-ce-only-inst|
|CE Scene + R2P1D | audiocaps-train-full-ce-scene-r2p1d|
|CE Scene + Inst | audiocaps-train-full-ce-scene-inst|
|CE R2P1D + Inst | audiocaps-train-full-ce-r2p1d-inst|
|CE - R2P1D + Inst + VGGish | audiocaps-train-full-ce-r2p1d-inst-vggish |
|CE - R2P1D + Inst + VGGSound | audiocaps-train-full-ce-r2p1d-inst-vggsound |
|CE - R2P1D + Inst + VGGish + VGGSound | audiocaps-train-full-ce-r2p1d-inst-vggish-vggsound |

#### CLOTHO:
|Experiment type | Model name|
|---|---|
|CE VGGish only | clotho-train-full-ce-only-audio|
|CE VGGish + VGGSound | clotho-train-vggish-vggsound|
|MoEE VGGish + VGGSound | clotho-train-vggish-vggsound-moee|
|MMMT VGGish + VGGSound | clotho-train-vggish-vggsound-mmt|

#### Activity-net:
|Experiment type | Model name|
|---|---|
|CE VGGish only | activity-net-train-full-ce-audio-only|

#### QuerYDSegments:
|Experiment type | Model name|
|---|---|
|CE VGGish only | querydsegments-train-full-ce-audio-only|

#### SoundDescs:
|Experiment type | Model name|
|---|---|
|CE VGGish + VGGSound | sounddescs-train-vggish-vggsound|
|MoEE VGGish + VGGSound | sounddescs-train-vggish-vggsound-moee|
|MMMT VGGish + VGGSound | sounddescs-train-vggish-vggsound-mmt|

### Finding the corresponding .json file names for evaluation of pre-trained finetuned models

|Experiment type | Model name|
|---|---|
|CE VGGish + VGGSound trained on SoundDescs finetuned on AudioCaps | audiocaps-train-vggish-vggsound-finetuned-from-sd|
|CE VGGish + VGGSound trained on AudioCaps finetuned on CLOTHO | clotho-train-vggish-vggsound-finetuned-from-ac|
|CE VGGish + VGGSound trained on SoundDescs finetuned on CLOTHO | clotho-train-vggish-vggsound-finetuned-from-sd|
|CE VGGish + VGGSound trained on AudioCaps finetuned on SoundDescs | sounddescs-train-vggish-vggsound-finetuned-from-ac|
