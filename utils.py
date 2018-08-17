import os
import sys
import numpy as np 
import csv

def custom_topK(pred, gt, k=3): # tham khao cua anh tiepvupsu
	topk = np.argsort(pred, axis=1)[:, -k:][:, ::-1]
	diff = topk - np.array(gt).reshape((-1, 1))
	n_correct  = np.where(diff==0)[0].size
	topk_err = float(n_correct)/pred.shape[0]
	return n_correct, topk_err

def gen_output_csv(idx, outputs):
	idx = str(idx)
	return idx + str(outputs)[1:-1].replace(',', ' ') + '\n'
