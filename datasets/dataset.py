import torch.utils.data as data
from PIL import Image
import os
import os.path
import sys
import matplotlib.pyplot as plt
import random
import numpy as np
from torchvision import transforms
import torch
from scipy import signal
import bisect
import cv2
import pandas as pd
import utils.videotransforms as videotransforms
import re
import csv


def default_seq_reader(videoslist, win_length, stride):
	shift_length = stride #length-1
	sequences = []
	csv_data_list = os.listdir(videoslist)
	for video in csv_data_list:
		vid_data = pd.read_csv(os.path.join(videoslist,video))
		video_data = vid_data.to_dict("list")
		images = video_data['img']
		labels_V = video_data['V']
		frame_ids = video_data['frame_id']
		label_array = np.asarray(labels_V, dtype=np.float32)
		medfiltered_labels = signal.medfilt(label_array)
		vid = list(zip(images, medfiltered_labels, frame_ids))
		frameid_array = np.asarray(frame_ids, dtype=np.int32)
		length = len(images)
		start = 0
		end = start + win_length

		if end < length:
			while end < length:
				indices = np.where((frameid_array>=start+1) & (frameid_array<=end))[0]
				frame_id = frameid_array[indices]
				norm_frame_id = frame_id - start -1
				if (len(indices) > 0):
					seq = vid[indices[0]:indices[len(indices)-1]+1]
					sequences.append([seq, norm_frame_id])
				start = start + shift_length
				end = start + win_length

			start = length - win_length
			end = length
			indices = np.where((frameid_array>=start+1) & (frameid_array<=end))[0]
			if (len(indices) > 0):
				frame_id = frameid_array[indices]
				norm_frame_id = frame_id - start -1
				seq = vid[indices[0]:indices[len(indices)-1]+1]
				sequences.append([seq, norm_frame_id])

		else:
			end = length
			indices = np.arange(start, end)
			seq = vid[indices[0]:indices[len(indices)-1]+1]
			frame_id = frameid_array[indices]-1
			sequences.append([seq, frame_id])
	return sequences

def default_list_reader(fileList):
	with open(fileList, 'r') as file:
		video_length = 0
		videos = []
		lines = list(file)
		for i in range(9):
			line = lines[video_length]
			imgPath, label = line.strip().split(' ')
			find_str = os.path.dirname(imgPath)
			new_video_length = 0
			for line in lines:
				if find_str in line:
					new_video_length = new_video_length + 1
			videos.append(lines[video_length:video_length + new_video_length])
			video_length = video_length + new_video_length
	return videos

class ImageList(data.Dataset):
	def __init__(self, root, fileList, length, flag, stride, list_reader=default_list_reader, seq_reader=default_seq_reader):
		self.root = root
		self.videoslist = fileList 
		self.win_length = length
		self.stride = stride
		self.sequence_list = seq_reader(self.videoslist, self.win_length, self.stride)
		self.flag = flag

	def __getitem__(self, index):
		seq_path, seq_id = self.sequence_list[index]
		seq, label = self.load_data_label(self.root, seq_path, seq_id, self.flag)
		label_index = torch.DoubleTensor([label])
		return seq, label_index

	def __len__(self):
		return len(self.sequence_list)

	def load_data_label(self, root, SeqPath, seq_id, flag):
		if (flag == 'train'):
			data_transforms = transforms.Compose([videotransforms.RandomCrop(224),
										   videotransforms.RandomHorizontalFlip(),
			])
		else:
			data_transforms=transforms.Compose([videotransforms.CenterCrop(224),
			])
		output = []
		inputs = []
		lab = []
		frame_ids = []
		seq_length = len(SeqPath)
		for image, ids in zip(SeqPath, seq_id):
			imgPath = image[0]
			label = image[1]
			img = cv2.imread(root + imgPath)
			if (img is None):
				img = np.zeros((112, 112, 3), dtype=np.float32)
			w,h,c = img.shape
			if w == 0:
				continue
			else:
				img = cv2.resize(img, (224, 224))[:, :, [2, 1, 0]]

			img = (img/255.)*2 - 1
			inputs.append(img)
			lab.append(float(label))
			frame_ids.append(ids)

		if (len(inputs) <self.win_length):
			imgs = np.zeros((self.win_length, 224, 224, 3), dtype=np.int16)
			lables = np.zeros((self.win_length), dtype=np.int16)
			imgs[frame_ids] = inputs
			lables[frame_ids] = lab
			imgs=np.asarray(imgs, dtype=np.float32)
			targets = np.asarray(lables, dtype=np.float32)
		else:
			imgs=np.asarray(inputs, dtype=np.float32)
			targets = np.asarray(lab, dtype=np.float32)
		imgs = data_transforms(imgs)
		return torch.from_numpy(imgs.transpose([3, 0, 1, 2])), targets
