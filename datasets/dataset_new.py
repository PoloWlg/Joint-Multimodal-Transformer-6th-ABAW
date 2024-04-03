from PIL import Image
import os
import os.path
from os.path import basename, dirname, abspath
import math

import numpy as np
import torchaudio
from torchvision import transforms
import torch
from scipy import signal
from .spec_transform import *
from .clip_transforms import *
import pandas as pd
import utils.videotransforms as videotransforms
import torch.utils.data as data

root_dir = dirname(dirname(abspath(__file__)))
sys.path.append(root_dir)

import dllogger as DLLogger


def get_filename(n):
	filename, ext = os.path.splitext(os.path.basename(n))
	return filename


def sort_files_by_basename(l_files: list) -> list:
	bnames = [basename(p) for p in l_files]

	both = list(zip(bnames, l_files))
	sorted_both = sorted(both, key=lambda x: x[0], reverse=False)
	out_sorted_files = [item[1] for item in sorted_both]

	return out_sorted_files


def default_seq_reader(videoslist, win_length, stride, dilation, wavs_list,
					   realtimestamps_path, take_n_videos: int = -1):
	shift_length = stride #length-1
	sequences = []
	csv_data_list = os.listdir(videoslist)

	skip_vids = ['313.csv', '212.csv', '303.csv', '171.csv',
				 '40-30-1280x720.csv', '286.csv', '270.csv', '234.csv',
				 '239.csv', '266.csv']

	DLLogger.log(f"Number of Sequences before removing skipped ones: "
				 f"{len(csv_data_list)}")

	tmp = []
	for item in csv_data_list:
		if item not in skip_vids:
			tmp.append(item)

	csv_data_list = tmp
	DLLogger.log(f"Number of Sequences after removing skipped ones: "
				 f"{len(csv_data_list)}")

	csv_data_list = sort_files_by_basename(csv_data_list)

	assert (take_n_videos == -1) or (take_n_videos > 0), take_n_videos
	assert isinstance(take_n_videos, int), type(take_n_videos)
	_n = len(csv_data_list)
	if take_n_videos > 0:
		csv_data_list = csv_data_list[:take_n_videos]
		DLLogger.log(f'We took only {len(csv_data_list)}/{_n} for training.'
					 f'take_n_videos={take_n_videos}.')

	for video in csv_data_list:
		if video.startswith('.'):
			continue
		if video in skip_vids:
			continue
		vid_data = pd.read_csv(os.path.join(videoslist,video))
		video_data = vid_data.to_dict("list")
		images = video_data['img']
		labels_V = video_data['V']
		labels_A = video_data['A']
		label_arrayV = np.asarray(labels_V, dtype = np.float32)
		label_arrayA = np.asarray(labels_A, dtype = np.float32)
		frame_ids = video_data['frame_id']
		f_name = get_filename(video)
		if f_name.endswith('_left'):
			wav_file_path = os.path.join(wavs_list, f_name[:-5])
			vidname = f_name[:-5]
		elif f_name.endswith('_right'):
			wav_file_path = os.path.join(wavs_list, f_name[:-6])
			vidname = f_name[:-6]
		else:
			wav_file_path = os.path.join(wavs_list, f_name)
			vidname = f_name
		vid = np.asarray(list(zip(images, label_arrayV, label_arrayA)))
		frameid_array = np.asarray(frame_ids, dtype=np.int32)
		time_filename = os.path.join(realtimestamps_path, vidname) + '_video_ts.txt'
		f = open(os.path.join(time_filename))
		lines = f.readlines()[1:]
		length = len(lines) 
		end = 481
		start = end - win_length
		counter = 0
		cnt = 0
		result = []
		while end < length + 481:
			avail_seq_length = end -start
			count = 15
			num_samples = 0
			vis_subsequnces = []
			aud_subsequnces = []
			for i in range(16):
				sub_indices = np.where((frameid_array>=(start+(i*32))+1) & (frameid_array<=(end -(count*32))))[0]
				wav_file = os.path.join(wav_file_path, str(end -(count*32))) +'.wav'
				if (end -(count*32)) <= length:
					result.append(end -(count*32))
					if len(sub_indices)>=8 and len(sub_indices)<16:
						subseq_indices = sub_indices[-8:]
						vis_subsequnces.append(vid[subseq_indices])
						aud_subsequnces.append(wav_file)
					elif len(sub_indices)>=16 and len(sub_indices)<24:
						subseq_indices = np.flip(np.flip(sub_indices)[::2])
						subseq_indices = subseq_indices[-8:]
						vis_subsequnces.append(vid[subseq_indices])
						aud_subsequnces.append(wav_file)
					elif len(sub_indices)>=24 and len(sub_indices)<32:
						subseq_indices = np.flip(np.flip(sub_indices)[::3])
						subseq_indices = subseq_indices[-8:]
						vis_subsequnces.append(vid[subseq_indices])
						aud_subsequnces.append(wav_file)
					elif len(sub_indices) == 32:
						subseq_indices = np.flip(np.flip(sub_indices)[::4])
						vis_subsequnces.append(vid[subseq_indices])
						aud_subsequnces.append(wav_file)
					elif len(sub_indices) > 0 and len(sub_indices) < 8:
						newList = [sub_indices[-1]]* (8-len(sub_indices))
						sub_indices = np.append(sub_indices, np.array(newList), 0)
						vis_subsequnces.append(vid[sub_indices])
						aud_subsequnces.append(wav_file)
				count = count - 1

			start_frame_id = start +1

			if len(vis_subsequnces) == 16:
				sequences.append([vis_subsequnces, aud_subsequnces])
			if avail_seq_length>512:
				print("Wrong Sequence")
			counter = counter + 1
			if counter > 31:
				end = end + 480 + shift_length
				start = end - win_length
				counter = 0
			else:
				end = end + shift_length
				start = end - win_length

		result.sort()
		if len(set(result)) == length:
			continue
		else:
			print(video)
			print(len(set(result)))
			print(length)
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
	def __init__(self, root, fileList, audList, length, flag, stride, dilation,
				 subseq_length, realtimestamps_path,
				 list_reader=default_list_reader,
				 seq_reader=default_seq_reader,
				 use_more_vision_data_augm: bool = False,
				 use_more_audio_data_augm: bool = False,
				 take_n_videos: int = -1):
		self.root = root
		self.realtimestamps_path = realtimestamps_path
		self.videoslist = fileList 
		self.win_length = length
		self.num_subseqs = int(self.win_length / subseq_length)
		self.wavs_list = audList
		self.stride = stride
		self.dilation = dilation
		self.subseq_length = int(subseq_length / self.dilation)
		self.sequence_list = seq_reader(self.videoslist,
										self.win_length,
										self.stride,
										self.dilation,
										self.wavs_list,
										self.realtimestamps_path,
										take_n_videos=take_n_videos)
		self.sample_rate = 44100
		self.window_size = 20e-3
		self.window_stride = 10e-3
		self.sample_len_secs = 1
		self.sample_len_clipframes = int(
			self.sample_len_secs * self.sample_rate * self.num_subseqs)
		self.sample_len_frames = int(self.sample_len_secs * self.sample_rate)
		self.audio_shift_sec = 1
		self.audio_shift_samples = int(self.audio_shift_sec * self.sample_rate)
		self.flag = flag

		self.use_more_vision_data_augm = use_more_vision_data_augm
		self.use_more_audio_data_augm = use_more_audio_data_augm

	def __getitem__(self, index):
		seq_path, wav_file = self.sequence_list[index]
		seq, label_V, label_A = self.load_vis_data(self.root,
												   seq_path,
												   self.flag,
												   self.subseq_length)

		aud_data = self.load_aud_data(wav_file, self.num_subseqs, self.flag)
		return seq, aud_data, label_V, label_A, wav_file  # _index

	def __len__(self):
		return len(self.sequence_list)

	def load_vis_data(self, root, SeqPath, flag, subseq_len):
		clip_transform = ComposeWithInvert([NumpyToTensor(),
											Normalize(mean=[0.43216, 0.394666, 0.37645],
													  std=[0.22803, 0.22145, 0.216989])])
		if (flag == 'train'):
			data_transforms = transforms.Compose([videotransforms.RandomCrop(224),
										   videotransforms.RandomHorizontalFlip(),
			])
		else:
			data_transforms=transforms.Compose([videotransforms.CenterCrop(224),
			])
		output = []
		subseq_inputs = []
		subseq_labels = []
		labV = []
		labA = []
		frame_ids = []
		seq_length = math.ceil(self.win_length / self.dilation)
		seqs = []
		sz = 112
		for clip in SeqPath:
			images = np.zeros((8, sz, sz, 3), dtype=np.uint8)
			labelV = -5.0
			labelA = -5.0
			for im_index, image in enumerate(clip):
				imgPath = image[0]
				labelV = image[1]
				labelA = image[2]

				try:
					img = np.array(Image.open(os.path.join(root , imgPath)))
					images[im_index, :, :, 0:3] = img
				except:
					pass

			if self.use_more_vision_data_augm:
				imgs = clip_transform(
					more_random_vision_augmentation(images, crop_size=sz))

			else: 
				imgs = clip_transform(RandomColorAugmentation(images))

			seqs.append(imgs)

			labV.append(float(labelV))
			labA.append(float(labelA))


		targetsV = torch.FloatTensor(labV)
		targetsA = torch.FloatTensor(labA)

		vid_seqs = torch.stack(seqs)

		return vid_seqs, targetsV, targetsA 

	def load_aud_data(self, wav_file, num_subseqs, flag):
		transform_spectra = transforms.Compose([
			transforms.ToPILImage(),
			transforms.RandomVerticalFlip(1),
			transforms.ToTensor(),
		])
		audio_spec_transform = ComposeWithInvert([AmpToDB(), Normalize(mean=[-14.8], std=[19.895])])


		spectrograms = []
		max_spec_shape = []
		for wave in wav_file:
			try:
				audio, sr = torchaudio.load(wave) 

			except:
				audio, sr = torchaudio.load(wave) 
			if audio.shape[1] <= 45599:
				_audio = torch.zeros((1, 45599))
				_audio[:, -audio.shape[1]:] = audio
				audio = _audio

			# ================== DATA AUGMENTATION =============================

			if self.use_more_audio_data_augm:
				# https://github.com/pytorch/audio/issues/3520
				n_fft = 1024
				spectros = torchaudio.transforms.Spectrogram(
					n_fft=n_fft,
					win_length=882,
					hop_length=441,
					window_fn=torch.hann_window,
					power=None,  # results is complex.
					return_complex=True
				)(audio)
				# 1, 513, 131: complex.

				# for n_ftt = 1024 --> second dim of spectros is 513 = 512 + 1.
				time_stretch = RandomTimeStretch(n_freq=1 + (n_fft // 2), p=0.6)
				spectros = time_stretch(spectros)
				# 1, 513, 110: complex: (…, freq, time)

				spectros = torch.view_as_real(spectros)
				# 1, 513, 110, 2: real. (…, freq, time, 2)
				power_spectros = torch.norm(spectros, p=2.0,
											dim=-1, keepdim=False)
				# 1, 513, 110: real. (…, freq, time)

				power_spectros = more_random_audio_spectrogram_augmentation(
					power_spectros)

				audio_feature = torchaudio.transforms.MelScale(
					n_mels=64, n_stft=1 + (n_fft // 2))(power_spectros)
				# 1, 64, 110: (…, n_mels, time)
				max_spec_shape.append(audio_feature.shape[2])

			else:
				# original code.
				audiofeatures = torchaudio.transforms.MelSpectrogram(
					sample_rate=sr,
					win_length=882,
					hop_length=441,
					n_mels=64,
					n_fft=1024,
					window_fn=torch.hann_window,
					power=2.0
				)(audio)

				max_spec_shape.append(audiofeatures.shape[2])

				audio_feature = audio_spec_transform(audiofeatures)


			spectrograms.append(audio_feature)

		spec_dim = max(max_spec_shape)

		audio_features = torch.zeros(len(max_spec_shape), 1, 64, spec_dim)
		for batch_idx, spectrogram in enumerate(spectrograms):
			if spectrogram.shape[2] < spec_dim:
				audio_features[batch_idx, :, :, -spectrogram.shape[2]:] = spectrogram
			else:
				audio_features[batch_idx, :,:, :] = spectrogram


		return audio_features 

