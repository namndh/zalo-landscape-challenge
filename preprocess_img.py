import cv2	 
import os 
import sys 
import glob	 
from random import shuffle
import numpy as np 

import h5py
import pickle

import constants	


TRAINVAL_DATA_PATH = os.path.join(constants.DATA_DIR, 'TrainVal')
TEST_DATA_PATH = os.path.join(constants.DATA_DIR, 'Test')
PREPROCESSED_DATA_PATH = os.path.join(constants.DATA_DIR, 'preprocessed_data')
PREPROCESSED_TRAIN_VAL_DATA_PATH = os.path.join(PREPROCESSED_DATA_PATH, 'train_val_data')
PREPOCESSED_TEST_DATA_PATH = os.path.join(PREPROCESSED_DATA_PATH, 'test_data')
HDF5_PATH = os.path.join(constants.DATA_DIR, 'trainval_data.hdf5')
HDF5_TEST_PATH = os.path.join(constants.DATA_DIR, 'test_data.hdf5')
EMPTY_TEST_ADDRS_FILE = os.path.join(constants.PROJECT_DIR, 'empty_test_addr.b')

trainval_dataset = list()
test_dataset = list()
shuffle_dataset = True

def empty_filter(addrs, labels=None):
	empty_ids = list()
	for idx, img_path in enumerate(addrs):
		if not os.path.isfile(img_path):
			del addrs[idx]
			if labels:
				del labels[idx]
		img = cv2.imread(img_path)
		if img is None:
			del addrs[idx]
			if labels:
				del labels[idx]
			if not labels:
				empty_ids.append(idx)

def get_id(test_data_addrs):
	test_data_addrs = list(test_data_addrs)
	substr1 = 'Test/'
	substr2 = '.jpg'
	interval = len(substr1)
	test_ids = list()
	for addr in test_data_addrs:
		idx1 = addr.find(substr1)
		idx2 = addr.find(substr2)
		test_id = addr[idx1 + interval:idx2]
		test_ids.append(test_id)
	return test_ids


def trainval_paths_list(trainval_sets, shuffle_dataset=True):
	if shuffle_dataset:
		shuffle(trainval_sets)
	
	addrs, labels = zip(*trainval_sets)
	return addrs, labels

def build_train_val_dataset(train_addrs, val_addrs, train_labels, val_labels, HDF5_PATH, size = 256, gray=True):
	if gray:
		channel = 1
	else:
		channel = 3
	train_shape = (len(train_addrs), channel, size, size)
	val_shape = (len(val_addrs), channel, size, size)


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
		if gray:
			img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
		img = img.transpose(2, 0, 1)
		hdf5_file["train_imgs"][i, ...] = img[None]

	for i in range(len(val_addrs)):
		if i % 5000 == 0 and i > 1:
			print('Val Data: {}/{}'.format(i, len(val_addrs)))
		addr  = val_addrs[i]
		img = cv2.imread(addr)
		if gray:
			img= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
		img = img.transpose(2, 0, 1)
		hdf5_file["val_imgs"][i, ...] = img[None]

	hdf5_file.close()

def build_test_dataset(TEST_DATA_PATH,HDF5_TEST_PATH, size=256, gray=True):
	test_path = os.path.join(TEST_DATA_PATH, '*.jpg')
	test_data_addrs = glob.glob(test_path, recursive=False)
	test_data_addrs = list(test_data_addrs)
	tmp_test_addrs = test_data_addrs
	print(len(tmp_test_addrs))
	remove_empty(test_data_addrs)
	empty_addrs = list(set(tmp_test_addrs) - set(test_data_addrs))
	print(len(empty_addrs))
	empty_test_ids = get_id(empty_addrs)
	test_ids = get_id(test_data_addrs)
	with open(EMPTY_TEST_ADDRS_FILE, 'wb') as f:
		pickle.dump(empty_test_ids, f)
		
	if gray:
		channel = 1
	else:
		channel = 3
	# test_shape = (len(test_data_addrs), channel, size, size)
	# test_shape = (len(test_data_addrs), channel, size, size)
	# hdf5_file = h5py.File(HDF5_TEST_PATH, mode='w')
	# hdf5_file.create_dataset("test_imgs", test_shape, np.uint8)
	# dt = h5py.special_dtype(vlen=str)
	# hdf5_file.create_dataset("test_ids", (len(test_ids),), dtype=dt)
	# hdf5_file["test_ids"][...] = test_ids
	# for i in range(len(test_data_addrs)):
	# 	if i % 5000 == 0 and i > 1:
	# 		print('Test data: {}/{}'.format(i, len(test_data_addrs)))
	# 	addr = test_data_addrs[i]
	# 	img = cv2.imread(addr)
	# 	if gray:
	# 		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	# 	img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
	# 	img = img.transpose(2, 0, 1)
	# 	hdf5_file["test_imgs"][i, ...] = img[None]

	# hdf5_file.close()


def main():
	# for i in range(constants.NUM_LABELS):
	# 	tmp_path = os.path.join(TRAINVAL_DATA_PATH, str(i) + '/*.jpg')
	# 	# print(tmp_path)
	# 	img_adrs = glob.glob(tmp_path, recursive=False)
	# 	labels = [int(i)] * len(img_adrs)
	# 	tmp_sets = list(zip(img_adrs, labels))
	# 	for tmp_set in tmp_sets:
	# 		trainval_dataset.append(tmp_set)

	# if shuffle_dataset:
	# 	shuffle(trainval_dataset)
	# 	addrs, labels = zip(*trainval_dataset)

	# addrs, labels = list(addrs), list(labels)
	# print(len(labels))
	# print(type(labels[0]))


	# remove_empty(addrs, labels=labels)

	# print(len(addrs))
	# print(len(labels))
	# train_addrs = addrs[0:int(0.8*len(addrs))]
	# train_labels = labels[0:int(0.8*len(labels))]

	# val_addrs = addrs[int(0.8*len(addrs)):int(len(addrs))]
	# val_labels = labels[int(0.8*len(addrs)):int(len(addrs))]
	# build_train_val_dataset(train_addrs, val_addrs, train_labels, val_labels, HDF5_PATH, size=224, gray=False)
	build_test_dataset(TEST_DATA_PATH, HDF5_TEST_PATH, size=224, gray=False)


if __name__ == '__main__':
	main()
	# test_path = os.path.join(TEST_DATA_PATH, '*.jpg')
	# test_data_addrs = glob.glob(test_path, recursive=False)
	# test_data_addrs = list(test_data_addrs)
	# print(test_data_addrs[0])
	# substr1 = 'Test/'
	# substr2 = '.jpg'
	# idx1 = test_data_addrs[0].find(substr1)
	# idx2 = test_data_addrs[0].find(substr2)
	# print(test_data_addrs[0][idx1+5:idx2])
