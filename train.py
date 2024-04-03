from __future__ import print_function
import os

import torch
import torch.nn.parallel
import torch.optim
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm
from EvaluationMetrics.cccmetric import ccc
import numpy as np


def train(train_loader,
		  model,
		  criterion,
		  optimizer,
		  scheduler,
		  fusion_model,
		  args,
		  fc_layer_for_audio_concat=None,
          transformer_audio_modality_fusion=None,
          fc_layer_for_video_concat=None,
          transformer_visio_modality_fusion=None,
          backbone_pretrainer=None
		  ):

	model.eval()

	if (not args.model_params["freeze_vision_R2D1"]) and "R2D1" in \
			args.model_params['l_vision_backbones']:
		model.vision_r2d1.train(True)

		if model.vision_r2d1_fc is not None:
			model.vision_r2d1_fc.train(True)

	if (not args.model_params["freeze_vision_I3D"]) and "I3D" in \
			args.model_params['l_vision_backbones']:
		model.vision_i3d.train(True)

	if (not args.model_params["freeze_audio_ResNet18"]) and "ResNet18" in \
			args.model_params['l_audio_backbones']:
		model.audio_resnet18.train(True)

	intra_modal_fusion = args.model_params["intra_modal_fusion"]

	if fusion_model is not None:
		fusion_model.train()
		assert backbone_pretrainer is None

	if backbone_pretrainer is not None:
		backbone_pretrainer.train()
		assert fusion_model is None

	if fc_layer_for_audio_concat is not None:
		fc_layer_for_audio_concat.train()
		assert transformer_audio_modality_fusion is None
		assert intra_modal_fusion in ["feat_concat_fc",
									  "None"], intra_modal_fusion
		if intra_modal_fusion == 'None':
			assert args.goal == 'PRETRAINING', args.goal

	if transformer_audio_modality_fusion is not None:
		transformer_audio_modality_fusion.train()
		assert fc_layer_for_audio_concat is None
		assert intra_modal_fusion == "encoder_plus_self_attention", intra_modal_fusion

	if fc_layer_for_video_concat is not None:
		fc_layer_for_video_concat.train()
		assert transformer_visio_modality_fusion is None
		assert intra_modal_fusion in ["feat_concat_fc",
									  "None"], intra_modal_fusion
		if intra_modal_fusion == 'None':
			assert args.goal == 'PRETRAINING', args.goal

	if transformer_visio_modality_fusion is not None:
		transformer_visio_modality_fusion.train()
		assert fc_layer_for_video_concat is None
		assert intra_modal_fusion == "encoder_plus_self_attention", intra_modal_fusion

	epoch_loss = 0
	vout = []
	vtar = []

	aout = []
	atar = []
    
	n = 0

	scaler = torch.cuda.amp.GradScaler()

	for batch_idx, (visualdata, audiodata, labels_V, labels_A, wav_file
					) in tqdm(enumerate(train_loader), total=len(train_loader),
							  position=0, leave=True):


		optimizer.zero_grad(set_to_none=True)

		audiodata = audiodata.cuda()
		visualdata = visualdata.cuda()

		with torch.cuda.amp.autocast():

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
					audiodata[i, :, :, :],  visualdata[i, :, :, :, :, :])

				if vision_r2d1_feats is not None:
					vision_r2d1_feats[i,:,:] = ft_vision_r2d1

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
					visual_feats = fc_layer_for_video_concat(input_tensor_reshaped)

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
				raise NotImplementedError()


			if (len(args.model_params['l_audio_backbones']) == 2) and (
					'wavLM'	in args.model_params['l_audio_backbones']):

				assert 'wavLM' in args.model_params["l_audio_backbones"]
				assert 'ResNet18' in args.model_params["l_audio_backbones"]

				if intra_modal_fusion == 'feat_concat_fc':
					assert fc_layer_for_audio_concat is not None
					assert transformer_audio_modality_fusion is None

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
						audio_resnet18_feats,audio_wavlm_feats)

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
			
			voutputs = audiovisual_vouts.view(-1, audiovisual_vouts.shape[0]*audiovisual_vouts.shape[1])
			aoutputs = audiovisual_aouts.view(-1, audiovisual_aouts.shape[0]*audiovisual_aouts.shape[1])

			vtargets = labels_V.view(-1, labels_V.shape[0]*labels_V.shape[1])
			atargets = labels_A.view(-1, labels_A.shape[0]*labels_A.shape[1])

			v_loss = criterion(voutputs, vtargets.cuda())
			a_loss = criterion(aoutputs, atargets.cuda())
			final_loss = v_loss + a_loss
			epoch_loss = (epoch_loss + final_loss).detach()

		scaler.scale(final_loss).backward()
		scaler.step(optimizer)
		scaler.update()
		n = n + 1

		vout.extend(voutputs.squeeze(0).detach().cpu().tolist())
		vtar.extend(vtargets.squeeze(0).detach().tolist())

		aout.extend(aoutputs.squeeze(0).detach().cpu().tolist())
		atar.extend(atargets.squeeze(0).detach().tolist())

	if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
		scheduler.step(epoch_loss.cpu().data.numpy() / n)

	else:
		scheduler.step()


	train_vacc = ccc(vout, vtar)
	train_aacc = ccc(aout, atar)
	

	torch.cuda.empty_cache()

	return train_vacc, train_aacc
