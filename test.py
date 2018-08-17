import os 
import sys 
import cv2
import numpy as np 
import pickle	
import h5py

import constants

TRAIN_ADDRS_TEXT = os.path.join(constants.PROJECT_DIR, 'train_addrs.b')
DATA_SET_PATH = os.path.join(constants.PROJECT_DIR, 'train_data.b')
HDF5_PATH = os.path.join(constants.DATA_DIR, 'trainval_data.hdf5')

with open(TRAIN_ADDRS_TEXT, 'rb') as f_in:
	train_addrs = pickle.load(f_in)

# data = np.empty((len(train_addrs), 1, 480, 480), dtype=np.uint8)
# for i, addr in enumerate(train_addrs):
# 	if i % 5000 == 0 and i > 1:
# 		print("Train data:{}/{}".format(i, len(train_addrs)))
# 	img = cv2.imread(addr)
# 	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 	img = cv2.resize(img, (480, 480), interpolation=cv2.INTER_CUBIC)
# 	data[i, ...] = img[None]
# with open(DATA_SET_PATH, 'wb+') as f:
# 	pickle.dump(data, f)
# print(data.shape)
# print(data[1, ...])
# cv2.imshow(data[1, ...], 'img')
# cv2.waitKey(2000)
# cv2.destroyAllWindows()
# with open(DATA_SET_PATH, 'wb+') as f:
# 	pickle.dump(data, f)
train_shape = (len(train_addrs), 1, 256, 256)
hdf5_file = h5py.File(HDF5_PATH, mode='w')
hdf5_file.create_dataset("train_img", train_shape, np.int8)
hdf5_file.create_dataset("train_mean", train_shape[1:], np.float32)


# mean = np.zeros(data_shape[1:], np.float32)

for i in range(len(train_addrs)):
	if i % 5000 == 0 and i > 1:
		print('Train data: {}/{}'.format(i, len(train_addrs)))
	addr = train_addrs[i]
	img = cv2.imread(addr)
	img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	hdf5_file["train_img"][i, ...] = img[None]
	# mean += img / float(len(train_labels))

# hdf5_file["train_mean"][...] = mean
hdf5_file.close()