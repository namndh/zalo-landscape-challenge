import os 
import sys 
import csv
import pickle

import constants 
SUBMISSON_PATH = os.path.join(constants.PROJECT_DIR, 'submission.csv')
EMPTY_IDS_PATH = os.path.join(constants.PROJECT_DIR, 'empty_test_addr.b')

with open(SUBMISSON_PATH, 'r') as submission_file:
	submission = csv.reader(submission_file, delimiter=',')
	lines = sum(1 for row in submission)
	print(lines)

with open(SUBMISSON_PATH, 'a') as submission_file:
	submission = csv.writer(submission_file, delimiter=',')
	with open(EMPTY_IDS_PATH, 'rb') as f:
		empty_addrs = pickle.load(f)
		print(type(empty_addrs))
		for add in empty_addrs:
			row = [add,'0 0 0']
			submission.writerow(row)
		# submission.writerow(['0','0 0 0'])

with open(SUBMISSON_PATH, 'r') as submission_file:
	submission = csv.reader(submission_file, delimiter=',')
	lines = sum(1 for row in submission)
	print(lines)