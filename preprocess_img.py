import cv2	 
import os 
import sys 
import glob	 
from random import shuffle
import numpy as np 
import tables
import h5py
import pickle

import constants	


TRAINVAL_DATA_PATH = os.path.join(constants.DATA_DIR, 'TrainVal')
TEST_DATA_PATH = os.path.join(constants.DATA_DIR, 'Test')
PREPROCESSED_DATA_PATH = os.path.join(constants.DATA_DIR, 'preprocessed_data')
PREPROCESSED_TRAIN_VAL_DATA_PATH = os.path.join(PREPROCESSED_DATA_PATH, 'train_val_data')
PREPOCESSED_TEST_DATA_PATH = os.path.join(PREPROCESSED_DATA_PATH, 'test_data')
HDF5_PATH = os.path.join(constants.DATA_DIR, 'trainval_data.hdf5')
VAL_ADDRS_PATH = os.path.join(constants.PROJECT_DIR, 'val_addrs.b')

trainval_dataset = list()
test_dataset = list()
shuffle_dataset = True


for i in range(constants.NUM_LABELS):
	tmp_path = os.path.join(TRAINVAL_DATA_PATH, str(i) + '/*.jpg')
	# print(tmp_path)
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

addrs, labels = list(addrs), list(labels)

print(len(addrs))
print(len(labels))

for idx, adr in enumerate(addrs):
	img = cv2.imread(adr)
	if img is None:
		del addrs[idx]
		del labels[idx]

print(len(addrs))
train_addrs = addrs[0:int(0.8*len(addrs))]
train_labels = labels[0:int(0.8*len(labels))]

val_addrs = addrs[int(0.8*len(addrs)):int(len(addrs))]
val_labels = labels[int(0.8*len(addrs)):int(len(addrs))]

train_shape = (len(train_addrs), 1, 256, 256)
val_shape = (len(val_addrs), 1, 256, 256)

hdf5_file = h5py.File(HDF5_PATH, mode='w')
hdf5_file.create_dataset("train_imgs", train_shape, np.uint8)
hdf5_file.create_dataset("val_imgs", val_shape, np.uint8)

hdf5_file.create_dataset("train_labels", (len(train_addrs),), np.int8)
hdf5_file["train_labels"][...] = train_labels
hdf5_file.create_dataset("val_labels", (len(val_addrs),), np.int8)
hdf5_file["val_labels"][...] = val_labels

for i in range(len(train_addrs)):
	if i % 5000 == 0 and i > 1:
		print('Train Data: {}/{}'.format(i, len(train_addrs)))
	addr = train_addrs[i]
	img = cv2.imread(addr)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)

	hdf5_file["train_imgs"][i, ...] = img[None]

for i in range(len(val_addrs)):
	if i % 5000 == 0 and i > 1:
		print('Val Data: {}/{}'.format(i, len(val_addrs)))
	addr  = val_addrs[i]
	img = cv2.imread(addr)
	img= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)

	hdf5_file["val_imgs"][i, ...] = img[None]

hdf5_file.close()