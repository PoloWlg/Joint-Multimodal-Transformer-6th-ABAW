from __future__ import print_function
import os
from tqdm import tqdm
import torch
import torch.nn.parallel
import torch.optim
from scipy.ndimage import uniform_filter1d
import pickle as pkl
import numpy as np
import sys
from EvaluationMetrics.cccmetric import ccc

from tools import MyDataParallel


def validate(val_loader,
			 model,
			 fusion_model,
			 args,
			 store_results_pkl: str = '',
			 fc_layer_for_audio_concat=None,
			 transformer_audio_modality_fusion=None,
			 fc_layer_for_video_concat=None,
			 transformer_visio_modality_fusion=None,
			 backbone_pretrainer=None
			 ):
    
	model.eval()

	intra_modal_fusion = args.model_params["intra_modal_fusion"]

	if fusion_model is not None:
		fusion_model.eval()
		assert backbone_pretrainer is None

	if backbone_pretrainer is not None:
		backbone_pretrainer.eval()
		assert fusion_model is None


	if fc_layer_for_audio_concat is not None:
		fc_layer_for_audio_concat.eval()
		assert transformer_audio_modality_fusion is None
		assert intra_modal_fusion in ["feat_concat_fc",
									  "None"], intra_modal_fusion
		if intra_modal_fusion == 'None':
			assert args.goal == 'PRETRAINING', args.goal

	if transformer_audio_modality_fusion is not None:
		transformer_audio_modality_fusion.eval()
		assert fc_layer_for_audio_concat is None
		assert intra_modal_fusion == "encoder_plus_self_attention", intra_modal_fusion


	if fc_layer_for_video_concat is not None:
		fc_layer_for_video_concat.eval()
		assert transformer_visio_modality_fusion is None
		assert intra_modal_fusion in ["feat_concat_fc",
									  "None"], intra_modal_fusion
		if intra_modal_fusion == 'None':
			assert args.goal == 'PRETRAINING', args.goal

	if transformer_visio_modality_fusion is not None:
		transformer_visio_modality_fusion.eval()
		assert fc_layer_for_video_concat is None
		assert intra_modal_fusion == "encoder_plus_self_attention", intra_modal_fusion


	vout = []
	vtar = []
	aout = []
	atar = []
 
	pred_a = dict()
	pred_v = dict()
	label_a = dict()
	label_v = dict()
 
	count = 0
	init_bsz = None
	c_bsz = None

	for batch_idx, (visualdata, audiodata, frame_ids, videos, vid_lengths,
					labelsV, labelsA, wav_file) in tqdm(
		enumerate(val_loader), total=len(val_loader), position=0, leave=True):

		audiodata = audiodata.cuda()
		visualdata = visualdata.cuda()

		if init_bsz is None:
			init_bsz = audiodata.shape[0]

		c_bsz = audiodata.shape[0]

		with torch.no_grad():
			b, seq_t, c, subseq_t, h, w = visualdata.size()

			vision_r2d1_feats = None
			vision_i3d_feats = None
			audio_resnet18_feats = None
			audio_wavlm_feats = None

			visual_feats = None
			aud_feats = None

			if 'R2D1' in args.model_params['l_vision_backbones']:
				vision_r2d1_feats = torch.empty((b, seq_t, 512),
												dtype=visualdata.dtype,
												device=visualdata.device
												)
			if 'I3D' in args.model_params['l_vision_backbones']:
				vision_i3d_feats = torch.empty((b, seq_t, 512),
											   dtype=visualdata.dtype,
											   device=visualdata.device
											   )
			if 'ResNet18' in args.model_params['l_audio_backbones']:
				audio_resnet18_feats = torch.empty((b, seq_t, 512),
												   dtype=visualdata.dtype,
												   device=visualdata.device
												   )

			if 'wavLM' in args.model_params['l_audio_backbones']:
				audio_wavlm_feats = torch.empty((b, seq_t, 768),
												dtype=visualdata.dtype,
												device=visualdata.device
												)

			for i in range(visualdata.shape[0]):

				ft_audio_resnet18, ft_vision_r2d1, ft_vision_i3d = model(
					audiodata[i, :, :, :], visualdata[i, :, :, :, :, :])

				if vision_r2d1_feats is not None:
					vision_r2d1_feats[i, :, :] = ft_vision_r2d1

				if vision_i3d_feats is not None:
					vision_i3d_feats[i, :, :] = ft_vision_i3d

				if audio_resnet18_feats is not None:
					audio_resnet18_feats[i, :, :] = ft_audio_resnet18

				if audio_wavlm_feats is not None:
					wavlm_feat = torch.empty(0, 0)
					for i_clip in range(len(wav_file[i])):
						split_path = wav_file[i][i_clip].split('/')
						create_wav_path = args.wavlm_features + '/' + \
										  split_path[6] + '/' + \
										  split_path[7].split('.')[0] + '.npy'
						if os.path.exists(create_wav_path):
							feat_numpy = np.load(create_wav_path)
						feat_tensor = torch.tensor(feat_numpy)

						if wavlm_feat.numel() == 0:
							wavlm_feat = feat_tensor
						elif wavlm_feat.shape == torch.Size([768]):
							wavlm_feat = torch.cat((wavlm_feat.unsqueeze(0),
													feat_tensor.unsqueeze(0)),
												   dim=0)
						else:
							wavlm_feat = torch.cat(
								(wavlm_feat, feat_tensor.unsqueeze(0)), dim=0)

					audio_wavlm_feats[i, :, :] = wavlm_feat

			if len(args.model_params["l_vision_backbones"]) == 2:

				assert 'R2D1' in args.model_params['l_vision_backbones']
				assert 'I3D' in args.model_params['l_vision_backbones']

				if intra_modal_fusion == 'feat_concat_fc':
					assert fc_layer_for_video_concat is not None
					assert transformer_visio_modality_fusion is None

					concat_video_feat = torch.cat((vision_r2d1_feats,
												   vision_i3d_feats),
												  dim=2).cuda()
					input_size = concat_video_feat.size(2)
					batch_size = vision_r2d1_feats.size(0)
					seq_length = vision_r2d1_feats.size(1)
					input_tensor_reshaped = concat_video_feat.view(
						batch_size, seq_length, input_size
					)
					visual_feats = fc_layer_for_video_concat(
						input_tensor_reshaped)

				elif intra_modal_fusion == "encoder_plus_self_attention":
					assert transformer_visio_modality_fusion is not None
					assert fc_layer_for_video_concat is None

					visual_feats = transformer_visio_modality_fusion(
						vision_r2d1_feats, vision_i3d_feats)

				else:
					raise NotImplementedError(intra_modal_fusion)

			elif (len(args.model_params["l_vision_backbones"]) == 1) and (
					"R2D1" in args.model_params["l_vision_backbones"]
			):
				assert fc_layer_for_video_concat is None
				assert transformer_visio_modality_fusion is None

				visual_feats = vision_r2d1_feats

			elif (len(args.model_params["l_vision_backbones"]) == 1) and (
					"I3D" in args.model_params["l_vision_backbones"]
			):
				assert fc_layer_for_video_concat is None
				assert transformer_visio_modality_fusion is None

				visual_feats = vision_i3d_feats

			elif len(args.model_params["l_vision_backbones"]) == 0:
				assert fc_layer_for_video_concat is None
				assert transformer_visio_modality_fusion is None

				assert args.goal == 'PRETRAINING', args.goal

			else:
				raise NotImplementedError

			if (len(args.model_params['l_audio_backbones']) == 2) and (
					'wavLM' in args.model_params['l_audio_backbones']):

				assert 'wavLM' in args.model_params["l_audio_backbones"]
				assert 'ResNet18' in args.model_params["l_audio_backbones"]

				if intra_modal_fusion == 'feat_concat_fc':
					assert fc_layer_for_audio_concat is not None

					concat_audio_feat = torch.cat((audio_resnet18_feats,
												   audio_wavlm_feats),
												  dim=2).cuda()
					input_size = concat_audio_feat.size(2)
					batch_size = audio_resnet18_feats.size(0)
					seq_length = audio_resnet18_feats.size(1)
					input_tensor_reshaped = concat_audio_feat.view(batch_size,
																   seq_length,
																   input_size)
					aud_feats = fc_layer_for_audio_concat(input_tensor_reshaped)

				elif intra_modal_fusion == 'encoder_plus_self_attention':
					assert transformer_audio_modality_fusion is not None
					assert fc_layer_for_audio_concat is None

					aud_feats = transformer_audio_modality_fusion(
						audio_resnet18_feats, audio_wavlm_feats)

				else:
					raise NotImplementedError(intra_modal_fusion)


			elif (len(args.model_params['l_audio_backbones']) == 1) and (
					'wavLM' in args.model_params['l_audio_backbones']):
				assert fc_layer_for_audio_concat is not None
				assert transformer_audio_modality_fusion is None

				aud_feats = fc_layer_for_audio_concat(audio_wavlm_feats)

			elif (len(args.model_params['l_audio_backbones']) == 1) and (
					'ResNet18' in args.model_params['l_audio_backbones']):
				assert fc_layer_for_audio_concat is None
				assert transformer_audio_modality_fusion is None

				aud_feats = audio_resnet18_feats

			elif len(args.model_params['l_audio_backbones']) == 0:
				assert fc_layer_for_audio_concat is None
				assert transformer_audio_modality_fusion is None

				assert args.goal == 'PRETRAINING', args.goal

			else:
				raise NotImplementedError

			if fusion_model is not None:
				assert aud_feats is not None
				assert visual_feats is not None

				# Some models with data-parallel do not like when bsz is not
				# 'fitting'. an issue occur with validset.
				# for trainset, we use drop_last. we introduce this to the test.
				# the batch size should be 'fitting'. Some models they dont
				# have this issue. Typically, the batchsize shouldnt cause an
				# error with DP. but, it seems to do with some models.
				if (c_bsz != init_bsz) and isinstance(
						fusion_model, MyDataParallel):
					audiovisual_vouts, audiovisual_aouts = fusion_model.module(
						aud_feats, visual_feats)
				else:
					audiovisual_vouts, audiovisual_aouts = fusion_model(
						aud_feats, visual_feats)

			elif backbone_pretrainer is not None:
				_s  = sum([visual_feats is None, aud_feats is None])
				assert _s == 1, _s

				if visual_feats is not None:
					_x = visual_feats
				else:
					_x = aud_feats

				audiovisual_vouts, audiovisual_aouts = backbone_pretrainer(_x)
			else:
				raise NotImplementedError

   
			audiovisual_vouts = audiovisual_vouts.detach().cpu().numpy()
			audiovisual_aouts = audiovisual_aouts.detach().cpu().numpy()

			labelsV = labelsV.cpu().numpy()
			labelsA = labelsA.cpu().numpy()

			for voutputs, aoutputs, labelV, labelA, frameids, video, vid_length in zip(
					audiovisual_vouts, audiovisual_aouts, labelsV, labelsA, frame_ids,
					videos, vid_lengths):

				for voutput, aoutput, labV, labA, frameid, vid, length in zip(
						voutputs, aoutputs, labelV, labelA, frameids, video, vid_length):

					if vid not in pred_a:
						if frameid > 1:
							print(vid, frameid, length)
							# SB:if this occurs, try to set the batch size to 1.
							print("something is wrong")
							sys.exit()
						count = count + 1

						pred_a[vid] = [0]*length
						pred_v[vid] = [0]*length
						label_a[vid] = [0]*length
						label_v[vid] = [0]*length

						# samples to be discarded are set to have a
						# prediction and label as zero.
						if labA == -5.0:
							continue

						if labV == -5.0:
							continue

						pred_a[vid][frameid-1] = aoutput
						pred_v[vid][frameid-1] = voutput
						label_a[vid][frameid-1] = labA
						label_v[vid][frameid-1] = labV
					else:
						if frameid <= length:

							if labA == -5.0:
								continue

							if labV == -5.0:
								continue
    
							pred_a[vid][frameid-1] = aoutput
							pred_v[vid][frameid-1] = voutput
							label_a[vid][frameid-1] = labA
							label_v[vid][frameid-1] = labV

	_smooth_pred_v = {}
	_smooth_pred_a = {}

	for key in pred_a.keys():
		clipped_preds_v = np.clip(pred_v[key], -1.0, 1.0)
		clipped_preds_a = np.clip(pred_a[key], -1.0, 1.0)

		smoothened_preds_v = uniform_filter1d(clipped_preds_v, size=20, mode='constant')
		smoothened_preds_a = uniform_filter1d(clipped_preds_a, size=50, mode='constant')

		_smooth_pred_v[key] = smoothened_preds_v
		_smooth_pred_a[key] = smoothened_preds_a

		tars_v = label_v[key]
		tars_a = label_a[key]

		for i in range(len(smoothened_preds_a)):
			vout.append(smoothened_preds_v[i])
			aout.append(smoothened_preds_a[i])
			vtar.append(tars_v[i])
			atar.append(tars_a[i])

	accV = ccc(np.array(vout), np.array(vtar))
	accA = ccc(np.array(aout), np.array(atar))

	if store_results_pkl != '':
		data = {
			'trg': {
				'vl': label_v,
				'ar': label_a
			},
			'pred': {
				'vl': _smooth_pred_v,
				'ar': _smooth_pred_a
			}
		}

		with open(store_results_pkl, 'wb') as fx:
			pkl.dump(data, fx, protocol=pkl.HIGHEST_PROTOCOL)

	torch.cuda.empty_cache()

	return accV, accA
