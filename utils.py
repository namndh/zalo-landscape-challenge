import os
import sys
import numpy as np 


def custom_topK(pred, gt, k=3): # tham khao cua anh tiepvupsu
	topk = np.argsort(pred, axis=1)[:, -k:][:, ::-1]
	diff = topk - np.array(gt).reshape((-1, 1))
	n_correct  = np.where(diff==0)[0].size
	topk_err = float(n_correct)/pred.shape[0]
	return n_correct, topk_err
