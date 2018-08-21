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
import numpy as np
import csv	
import torchvision.transforms as transforms

from utils import *
import constants 
from dataset import ZaloLandScapeTestDataset, ZaloLandscapeTrainValDataset

TRAINVAL_PATH = os.path.join(constants.DATA_DIR, 'trainval_data.hdf5')
TEST_PATH = os.path.join(constants.DATA_DIR, 'test_data.hdf5')
LOG_FILE = os.path.join(constants.PROJECT_DIR, 'submisson.csv')

parser = argparse.ArgumentParser(description='Zalo Landscape Classification')
parser.add_argument('--lr', default=0.1, type=float, help='Learning Rate')
parser.add_argument('--resume', '-r', action='store_true', help='Resume from checkpoint')
parser.add_argument('--train', '-tr', action='store_true', help='Train the model')
parser.add_argument('--predict', '-pr', action='store_true', help='Predict the data')
args = parser.parse_args()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
start_epoch = 0
best_err = 0
print(torch.cuda.current_device())

transform_to_tensor = transforms.ToTensor()

train_set = ZaloLandscapeTrainValDataset(TRAINVAL_PATH, root_dir='./data', train=True, transform=transform_to_tensor)
val_set = ZaloLandscapeTrainValDataset(TRAINVAL_PATH, root_dir='./data', train=False, transform=transform_to_tensor)
test_set = ZaloLandScapeTestDataset(TEST_PATH, root_dir='./data', transform=transform_to_tensor)

train_loader = utilsData.DataLoader(dataset=train_set, batch_size=200, sampler=None, shuffle=True, batch_sampler=None)
val_loader = utilsData.DataLoader(dataset=val_set, batch_size=200, sampler=None, shuffle=True, batch_sampler=None)
test_loader = utilsData.DataLoader(dataset=test_set, batch_size=200, sampler=None, shuffle=False, batch_sampler=None)

net = models.resnet34(pretrained=False)
print(net)
net.to(device)
if device == 'cuda':
	net = torch.nn.DataParallel(net)
	# cudnn.benchmark = True


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=args.lr)


def train(epoch):	
	print('\nEpoch: %d' % int(epoch))
	net.train()
	train_loss = 0
	train_correct = 0 
	total = 0 
	train_top3_correct = 0

	for batch_id, (images, labels) in enumerate(train_loader):
		images, labels = images.to(device), labels.to(device)
		optimizer.zero_grad()
		outputs = net(images)
		loss = criterion(outputs, labels)
		loss.backward()
		optimizer.step()

		train_loss += loss.item()
		_, predicted = outputs.max(1)
		top3_correct,_ = custom_topK(outputs.data.cpu().numpy(), labels, 3)
		train_top3_correct += top3_correct
		total += labels.size(0)
		train_correct += predicted.eq(labels).sum().item()
	top1_error = 1 - float(train_correct)/total
	top3_error = 1 - float(train_topk_correct)/total
	print('Loss:%.3f | Top 1 Error: %.3f | Top 3 Error : %.3f' % (train_loss/(batch_id + 1)), top1_error, top3_error)

def validate(epoch):
	print('\nEpoch: %d' % int(epoch))
	net.eval()
	train_loss = 0
	train_correct = 0
	total = 0
	train_top3_correct = 0

	for batch_id, (images, labels) in enumerate(val_loader):
		images, labels = images.to(device), labels.to(device)
		outputs = net(images)
		loss = criterion(outputs, labels)

		test_loss += loss.item()
		_, predicted = outputs.max(1)
		top3_correct, _ = custom_topK(outputs.data.cpu().numpy(), labels, 3)
		train_top3_correct += top3_correct
		total += labels.size(0)
		correct += predicted.eq(labels).sum().item()
	top1_error = 1 - float(train_correct)/total 
	top3_error = 1 - float(train_top3_correct)/total
	acc = (1 - top3_error)*100
	print('Loss:%.3f | Top 1 Error: %.3f | Top 3 Error : %.3f' % (train_loss/(batch_id + 1)), top1_error, top3_error)

	if top3_error < best_err :
		print('Saving ...')
		state = { 
			'net' : net.state_dict(),
			'acc': acc, 
			'epoch': epoch,
			'top1_err': top1_error,
			'top3_err': top3_error
		}
		if not os.path.isdir('checkpoint'):
			os.mkdir('checkpoint')
		torch.save(state, '/checkpoint/ckpt.t7')
		best_err = top3_error

def predict():
	assert os.path.isdir('checkpoint'), 'Error: model not available'
	checkpoint = torch.load('./checkpoint/model.t7')
	net.load_state_dict(checkpoint['net'])
	device = checkpoint['device']
	top1_err = checkpoint['top1_err']
	top3_err = checkpoint['top3_err']
	start_epoch = checkpoint['checkpoint']
	f = open(LOG_FILE, 'w+')
	f.write('id,predicted\n')

	print('Model was trained and validated with top-1-err:{} and top-3-err:{}'.format(top1_err, top3_err))
	net.eval()
	for idx, (images, images_ids) in enumerate(test_loader):
		images = images.to(device)
		outputs = net(images)
		outputs = outputs.data.cpu().numpy()
		outputs = np.argsort(outputs, axis=1)[:, -3:][:, ::-1]
	
		for i, image_id in enumerate(images_ids):
			tmp = gen_output_csv(image_id, list(outputs[i]))
			f.write(tmp)

		if idx % 2000 and idx > 1:
			print("Processing {}/{}".format(idx, len(test_loader)))


if args.resume:
	print('===> Resuming from checkpoint ...')
	assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
	checkpoint = torch.load('./checkpoint/ckpt.t7')
	net.load_state_dict(checkpoint['net'])
	# best_acc = checkpoint['acc']
	best_err = checkpoint['top3_error']
	start_epoch = checkpoint['checkpoint']
	for epoch in range(start_epoch, start_epoch + 150):
		train(epoch)
		validate(epoch)

if args.train:
	print('===> Train the model ...')
	for epoch in range(start_epoch, start_epoch + 150):
		train(epoch)
		validate(epoch)

if args.predict:
	predict()

