## Pretrained Experts

This folder contains a collection of features, extracted from the QuerYD [2] dataset as part of the paper:
*QuerYD: A video dataset with high-quality textual and audio narrations*.

### Training splits

The training splits are given in the files linked below:

* [train_list.txt](train_list.txt) (9113 videos)
* [val_list.txt](val_list.txt) (1952 videos)
* [test_list.txt](test_list.txt) (1954 videos)


**Tar contents**

The compressed tar file (244MB) can be downloaded from:

```
https://www.robots.ox.ac.uk/~vgg/research/collaborative-experts/data/features-v2/QuerYDSegments-experts.tar.gz
sha1sum: f2be088890294f92355ccfe109f824d814cf2cd5
```
A list of the contents of the tar file are given in [tar_include.txt](tar_include.txt).


### References:

[1] If you use these features, please consider citing:
```
@inproceedings{Liu2019a,
  author    = {Liu, Y. and Albanie, S. and Nagrani, A. and Zisserman, A.},
  booktitle = {British Machine Vision Conference},
  title     = {Use What You Have: Video retrieval using representations from collaborative experts},
  date      = {2019},
}
```

[2] Please also consider citing the original QuerYD dataset, which was described in:

```
@misc{oncescu2020queryd,
  title={QuerYD: A video dataset with high-quality textual and audio narrations}, 
  author={Andreea-Maria Oncescu and Jõao F. Henriques and Yang Liu and Andrew Zisserman and Samuel Albanie},
  year={2020},
}
```