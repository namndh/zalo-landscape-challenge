import cv2	 
import os 
import sys 
import glob	 
from random import shuffle
import numpy as np 
import h5py
import tables 

import constants	


TRAINVAL_DATA_PATH = os.path.join(constants.DATA_DIR, 'TrainVal')
TEST_DATA_PATH = os.path.join(constants.DATA_DIR, 'Test')
PREPROCESSED_DATA_PATH = os.path.join(constants.DATA_DIR, 'preprocessed_data')
PREPROCESSED_TRAIN_VAL_DATA_PATH = os.path.join(PREPROCESSED_DATA_PATH, 'train_val_data')
PREPOCESSED_TEST_DATA_PATH = os.path.join(PREPROCESSED_DATA_PATH, 'test_data')
HDF5_PATH = os.path.join(constants.DATA_DIR, 'zalo_landscape_trainval.hdf5')

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

img_test = cv2.imread(train_addrs[0])
print(img_test.shape)

val_addrs = addrs[int(0.8*len(addrs)):int(len(addrs))]
val_labels = labels[int(0.8*len(addrs)):int(len(addrs))]

print('{}.{}.{}.{}'.format(len(train_addrs), len(train_labels), len(val_addrs), len(val_labels)))

data_order = 'torch' # torch or tf for tensorflow

if data_order == 'torch':
	train_shape = (len(train_addrs), 1, 480, 480)
	val_shape = (len(val_addrs), 1, 480, 480)
	test_shape = (len(val_addrs), 1, 480, 480)
	data_shape = (0, 1, 480, 480)

# hdf5_file = h5py.File(HDF5_PATH, mode='w')

# hdf5_file.create_dataset("train_img", train_shape, np.int8)
# hdf5_file.create_dataset("val_img", val_shape, np.int8)

# hdf5_file.create_dataset("train_mean", train_shape[1:], np.float32)

# hdf5_file.create_dataset("train_labels", (len(train_addrs),), np.int8)
# hdf5_file["train_labels"][...] = train_labels

# hdf5_file.create_dataset("val_labels", (len(val_addrs),), np.int8)
# hdf5_file["val_labels"][...] = val_labels


for i in range(2):
	addr = train_addrs[i]
	print(addr)
	img = cv2.imread(addr)
	print(img.shape)
	img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	print(img_gray.shape)
	# cv2.namedWindow('display', cv2.WINDOW_AUTOSIZE)
	# cv2.imshow('img', img_gray)
	# cv2.waitKey(5000)
	# cv2.destroyAllWindows()

img_dtype = tables.Atom(np.uint8, data_shape)
hdf5_file = tables.open_file(HDF5_PATH, mode='w')

train_storage = hdf5_file.create_array(hdf5_file.root, 'train_img', img_dtype, shape=data_shape)
val_storage = hdf5_file.create_array(hdf5_file.root, 'val_img', img_dtype, shape=data_shape)

mean_storage = hdf5_file.create_array(hdf5_file.root, 'train_mean', img_dtype, shape=data_shape)

hdf5_file.create_array(hdf5_file.root, 'train_labels', train_labels)
hdf5_file.create_array(hdf5_file.root, 'val_labels', val_labels)
# data_train = np.empty((len(train_addrs), 1, 480, 480), dtype = np.uint8)
