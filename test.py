import os 
import sys
import numpy as np 
import h5py 
import cv2	
import matplotlib.pyplot as plt
import constants
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image


import pickle
# from dataset import ZaloLandscapeDataset


HDF5_PATH = os.path.join(constants.DATA_DIR, 'trainval_data.hdf5')
CIFAR10_PATH = os.path.join(constants.DATA_DIR, 'cifar-10-python/cifar-10-batches-py/data_batch_1')
TEST_IMG_PATH = '/media/t3min4l/Data 2/zalo-landscape-challenge/data/TrainVal/0/1089.jpg'

img = cv2.imread(TEST_IMG_PATH)
print(img.shape)
print(img[:,:,0])
print(img[:,:,1].shape)
print(np.mean(img[:,:,0]))
print(np.std(img[:,:,1]))
print(constants.PROJECT_DIR)

# hdf5_file = h5py.File(HDF5_PATH, "r")

# data_num = hdf5_file["train_imgs"].shape[0]

# print(type(hdf5_file["train_imgs"][0]))
# img = hdf5_file["train_imgs"][0]
# print(img)
# img = img.ravel()
# img = img.reshape(1, -1)
# print(img)
# print(img.shape)
# # img = np.array(img)
# # img = Image.fromarray(img)

# # imgplot = plt.imshow(img, cmap='gray')
# # plt.show()
# # cv2.imshow('IMG', img)
# # cv2.waitKey(2000)
# # cv2.destroyAllWindows()

# print(hdf5_file["train_imgs"][0].shape)

# with open(CIFAR10_PATH, 'rb') as f_in:
# 	tmp = pickle.load(f_in, encoding='bytes')

# for key in tmp.keys():
# 	print(key)


# print(tmp[b'data'][0].shape)

# # img = Image.fromarray(tmp[b'data'][0])

# print(data_num)

# transform_train = transforms.ToTensor()

# zalo_dataset = ZaloLandscapeDataset(hdf5_file=HDF5_PATH, root_dir='/data', train=True, transform=transform_train)

# for i in range(len(zalo_dataset)):
# 	image, label = zalo_dataset[i]

# 	print(type(image))
# 	print(label)

# 	img = image.numpy()
# 	img = np.reshape(img, (256, 256))
# 	imgplot = plt.imshow(img, cmap='gray')
# 	plt.show()
# 	# cv2.imshow('IMG', img)
# 	# cv2.waitKey(2000)
# 	# cv2.destroyAllWindows()

# 	if i > 1:
# 		break	
		
