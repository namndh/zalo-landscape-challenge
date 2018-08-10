import cv2	 
import os 
import sys 
import glob	 
from random import shuffle

import constants	


TRAINVAL_DATA_PATH = os.path.join(constants.DATA_DIR, 'TrainVal')
TEST_DATA_PATH = os.path.join(constants.DATA_DIR, 'Test')

PREPROCESSED_DATA_PATH = os.path.join(constants.DATA_DIR, 'preprocessed_data')

PREPROCESSED_TRAIN_VAL_DATA_PATH = os.path.join(PREPROCESSED_DATA_PATH, 'train_val_data')

PREPOCESSED_TEST_DATA_PATH = os.path.join(PREPROCESSED_DATA_PATH, 'test_data')

trainval_dataset = list()
test_dataset = list()
shuffle_dataset = True


for i in range(constants.NUM_LABELS):
	tmp_path = os.path.join(TRAINVAL_DATA_PATH, str(i) + '/*.jpg')
	print(tmp_path)
	img_adrs = glob.glob(tmp_path, recursive=False)
	labels = [i] * len(img_adrs)
	tmp_sets = list(zip(img_adrs, labels))
	for tmp_set in tmp_sets:
		trainval_dataset.append(tmp_set)

test_path = os.path.join(TEST_DATA_PATH, '*.jpg')
test_data_adrs = glob.glob(test_path, recursive=False)
test_data_adrs = list(test_data_adrs)
for adr in test_data_adrs:
	test_dataset.append(adr)

print(len(trainval_dataset))
print(len(test_dataset))
if shuffle_dataset:
	shuffle(trainval_dataset)
	addrs, labels = zip(*trainval_dataset)

print(len(addrs))
print(len(labels))


train_addrs = addrs[0:int(0.8*len(addrs))]
train_labels = labels[0:int(0.8*len(labels))]

val_addrs = addrs[int(0.8*len(addrs)):int(len(addrs))]
val_labels = labels[int(0.8*len(addrs)):int(len(addrs))]

print('{}.{}.{}.{}'.format(len(train_addrs), len(train_labels), len(val_addrs), len(val_labels)))