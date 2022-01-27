# AudioCaps dataset statistics
# March 2021, ask


import os
import csv
import json
import numpy as np
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Load AudioCaps test data
# -----------------------------------------------------------------------------

audiocaps_base = '/home/askoepke97/coding/ce/collab-experts/data/AudioCaps/audiocaps/dataset'
audiocaps_test_file = os.path.join(audiocaps_base, 'test.csv')

audiocapid = []
youtubeid = []
yid_dict = dict()

with open(audiocaps_test_file, 'r') as csvfile:
    reader = csv.reader(csvfile)
    i = 0
    for row in reader:
        if i > 0:
            if not int(row[2]) == 0:
                filename = row[1] + '_%d000'%int(row[2])
            else:
                filename = row[1] + '_%d'%int(row[2])
            ytname = row[1]
            yid_dict[filename] = ytname
        i += 1

# -----------------------------------------------------------------------------
# Load audioset ontology and train (and eval) data
# -----------------------------------------------------------------------------

audiosetbase = '/home/askoepke97/coding/ce/collab-experts/data/dataset_statistics'
ontology = os.path.join(audiosetbase, 'ontology.json')

evalcsv = os.path.join(audiosetbase, 'eval_segments.csv')
traincsv = os.path.join(audiosetbase, 'unbalanced_train_segments.csv')

with open(ontology) as json_file:
    ontology_data = json.load(json_file)

classids = dict()

for ind in np.arange(len(ontology_data)):
    classids[ontology_data[ind]['id']] = ontology_data[ind]['name']

evaldict = dict()

with open(evalcsv, 'r') as as_csvfile:
    reader = csv.reader(as_csvfile)
    i = 0
    for row in reader:
        if i > 2:
            ytname = row[0]
            starttime = row[1]
            classes = row[3].split(',')
            newclasses = []
            for classe in classes:
                if classe.strip()[0] == '"' and not classe.strip()[-1] == '"':
                    newclasses.append(classids[row[3].strip()[1:]])
                elif classe.strip()[-1] == '"' and not classe.strip()[0] == '"':
                    newclasses.append(classids[row[3].strip()[:-1]])
                elif classe.strip()[-1] == '"' and classe.strip()[0] == '"':
                    newclasses.append(classids[row[3].strip()[1:-1]])
                else:
                    newclasses.append(classids[row[3].strip()])
            evaldict[ytname] = newclasses
        i += 1

traindict = dict()

with open(traincsv, 'r') as as_csvfile:
    reader = csv.reader(as_csvfile)
    i = 0
    for row in reader:
        if i > 2:
            ytname = row[0]
            starttime = row[1]
            classes = row[3].split(',')
            newclasses = []
            for classe in classes:
                if classe.strip()[0] == '"' and not classe.strip()[-1] == '"':
                    newclasses.append(classids[row[3].strip()[1:]])
                elif classe.strip()[-1] == '"' and not classe.strip()[0] == '"':
                    newclasses.append(classids[row[3].strip()[:-1]])
                elif classe.strip()[-1] == '"' and classe.strip()[0] == '"':
                    newclasses.append(classids[row[3].strip()[1:-1]])
                else:
                    newclasses.append(classids[row[3].strip()])
            traindict[ytname] = newclasses
        i += 1
print(i, 'len train')

# -----------------------------------------------------------------------------
# Load VGGSound training ulrs
# -----------------------------------------------------------------------------

vggsoundpath = '/home/askoepke97/coding/gitrepos/sound_features/VGGSound/data/train.csv'
vggvids = []
with open(vggsoundpath, 'r') as csv_file:
    reader = csv.reader(csv_file)
    for row in tqdm(reader):
        vggvids.append(row[0].split('_')[0])

# -----------------------------------------------------------------------------
# Find overlap between VGGSound training set and AudioCaps test set
# -----------------------------------------------------------------------------

overlap_counter = 0
vggcounter = 0
uniqueclasses = [] #111 unique classes in unfiltered (before removing overlap with VGGSound)  AudioCaps test set, 97 in the val set, 238 in train 
newclassdict = dict()
overlap_test_videos = []
for key, value in tqdm(yid_dict.items()):
    if value in vggvids:
        vggcounter += 1
        overlap_test_videos.append(value)
  # # Check for overlap between AudioCaps test set and AudioSet training data
  #  if value in evaldict.keys():
  #      if not evaldict[value] in newclassdict.values():
  #          uniqueclasses.append(evaldict[value])
  #      newclassdict[key] = evaldict[value]
  #  elif value in traindict.keys():
  #      overlap_counter += 1
  #      if not traindict[value] in newclassdict.values():
  #          uniqueclasses.append(traindict[value])
  #      newclassdict[key] = traindict[value]


# -----------------------------------------------------------------------------
# Filter the test.csv dictionary yid_dict from AudioCaps for overlap with VGGSound
# -----------------------------------------------------------------------------

new_yid_dict = dict()
for key, value in yid_dict.items():
    if not key.split('_')[0] in overlap_test_videos:
        new_yid_dict[key] = value

# -----------------------------------------------------------------------------
# Make dictionaries that contain classes as keys and video names in AudioCaps 
# test as values
# -----------------------------------------------------------------------------

class_video_dict = dict()
for key, value in tqdm(new_yid_dict.items()):
    if value in traindict.keys():
        for vid_class in traindict[value]: #traindict[value] could contain a list with multiple classes
            if not vid_class in class_video_dict.keys():
                class_video_dict[vid_class] = [key]
            elif vid_class in class_video_dict.keys():
                class_video_dict[vid_class].append(key)
    else:
        import pdb; pdb.set_trace()

new_class_video_dict = class_video_dict.copy()
for key, value in class_video_dict.items():
    if len(value) < 10:
        new_class_video_dict.pop(key)

print(len(new_class_video_dict.keys()))

# print count of each class in dictionary, videos belong to single class only
count_no_videos = 0
for key, value in tqdm(new_class_video_dict.items()):
    print(key, len(value))
    count_no_videos += len(value)
print(len(new_yid_dict.keys()), count_no_videos, 'number of videos in test set and number of videos in dictionaries')

# save class dictionary that only contains classes with more than 10 example videos in the test set (34 classes)

with open('test_class_videoid_dict_morethan10.json', 'w') as fp:
    json.dump(new_class_video_dict, fp)

# save class dictionary with all classes in the test set (106 classes)

with open('test_class_videoid_dict_all.json', 'w') as fp:
    json.dump(class_video_dict, fp)

## -----------------------------------------------------------------------------
## Filter the AudioCaps test_list.txt to remove overlap with the VGGSound training data
## -----------------------------------------------------------------------------
#
#audiocaps_testfile = '/home/askoepke97/akata-shared/askoepke97/data/AR/AudioCaps/structured-symlinks/test_list.txt'
#file1 = open(audiocaps_testfile, 'r')
#oldtestfiles = file1.readlines()
#file1.close()
#newtestfiles = []
#for oldtestfile in oldtestfiles:
#    if not oldtestfile.split('_')[0] in overlap_test_videos:
#        newtestfiles.append(oldtestfile)
#file1 = open('/home/askoepke97/akata-shared/askoepke97/data/AR/AudioCaps/structured-symlinks/filtered_test_list.txt', 'w')
#file1.writelines(newtestfiles)
#file1.close()
#
## -----------------------------------------------------------------------------

#print('There are %d videos in the AudioCaps test set that are contained in the AudioSet training set.'%overlap_counter) #975 and there are only 975 videos in the AudioCaps test set, 495 in the val set, 49838 in train
