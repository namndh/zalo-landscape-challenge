import torch 
import torchvision
from torch.autograd import Variable 
import torch.nn.functional as F 
import torch.nn as nn
import torch.optim as optim 
import time 
import os 
import sys 
import argparse
import torch.utils.data as utilsData
import torchvision.models as models

import constants 
from dataset import ZaloLandScapeTestDataset, ZaloLandscapeTrainValDataset

TRAINVAL_PATH = os.path.join(constants.DATA_DIR, 'trainval_data.hdf5')
TEST_PATH = os.path.join(constants.DATA_DIR, 'test_data.hdf5')

parser = argparse.ArgumentParser(description='Zalo Landscape Classification')
parser.add_argument('--lr', default=0.1, type=float, help='Learning Rate')
parser.add_argument('--resume', '-r', action='store_true', help='Resume from checkpoint')
parser.add_argument('--train')
args = parser.parse_args()

device = torch.device('cuda:0', if torch.cuda.is_available() else 'cpu')
EPOCH_NUM = 0
best_acc = 0
print(torch.cuda.current_device())

transform_to_tensor = transforms.ToTensor()

train_set = ZaloLandscapeTrainValDataset(TRAINVAL_PATH, root_dir='./data', train=True, transform=transform_to_tensor)
val_set = ZaloLandscapeTrainValDataset(TRAINVAL_PATH, root_dir='./data', train=False, transform=transform_to_tensor)
test_set = ZaloLandScapeTestDataset(TEST_PATH, root_dir='./data', transform=transform_to_tensor)

train_loader = utilsData.DataLoader(dataset=train_set, batch_size=100, sampler=None, shuffle=True, batch_sampler=None)
val_loader = utilsData.DataLoader(dataset=val_set, batch_size=100, sampler=None, shuffle=True, batch_sampler=None)
test_set = utilsData.DataLoader(dataset=test_set, batch_size=100, sampler=None, shuffle=True, batch_sampler=None)

net = models.resnet34(pretrained=False)
print(net)
net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.1)


def train(epoch):	
	print('\nEpoch: %d' % int(epoch))
	net.train()
	train_loss = 0
	correct = 0 
	total = 0 

	for batch_id, (images, labels) in enumerate(train_loader):
		images, labels = images.to(device), labels.to(device)
		optimizer.zero_grad()
		outputs = net(images)
		loss = criterion(outputs, labels)
		loss.backward()
		optimizer.step()

		train_loss += loss.item()
		_, predicted = outputs.max(1)
		total += labels.size(0)
		correct += predicted.eq(labels).sum().item()
	print('Loss:%.3f | Acc: %.3f%% (%d/%d)' % (train_loss/(batch_id + 1)), 100.*correct/total, correct, total)

def test(epoch):
	print('\nEpoch: %d' % int(epoch))
	net.eval()
	train_loss = 0
	correct = 0
	total = 0

	for batch_id, (images, labels) in enumerate(val_loader):
		images, labels = images.to(device), labels.to(device)
		outputs = net(images)
		loss = criterion(outputs, labels)

		test_loss += loss.item()
		_, predicted = outputs.max(1)
		total += labels.size(0)
		correct += predicted.eq(labels).sum().item()
	print('Loss:%.3f | Acc: %.3f%% (%d/%d)' % (train_loss/(batch_id + 1)), 100.*correct/total, correct, total)

	if acc > best_acc :
		print('Saving ...')
		state = { 
			'net' : net.state_dict(),
			'acc': acc, 
			'epoch': epoch
		}
		if not os.path.isdir('checkpoint'):
			os.mkdir('checkpoint')
		torch.save(state, '/checkpoint/ckpt.t7')
		best_acc = acc

def predict(epoch):
	print('\nEpoch: %d' % epoch)
	net.eval()
	

if device == 'cuda':
	net = torch.nn.DataParallel(net)
	# cudnn.benchmark = True





if args:
	pass