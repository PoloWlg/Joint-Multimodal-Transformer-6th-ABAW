from models.pytorch_i3d_new import Unit3D
import torch.nn as nn
from models.temporal_convolutional_model import TemporalConvNet

class I3D_WSDDA(nn.Module):
	def __init__(self, model):
		super(I3D_WSDDA, self).__init__()
		self.i3d_WSDDA = model
		self.predictions = nn.Sequential(
					Unit3D(in_channels=384+384+128+128, output_channels=512,
							 kernel_shape=[1, 1, 1],
							 padding=0,
							 activation_fn=None,
							 use_batch_norm=False,
							 use_bias=True,
							 name='logits'),
					Unit3D(in_channels=512, output_channels=1,
							 kernel_shape=[1, 1, 1],
							 padding=0,
							 activation_fn=None,
							 use_batch_norm=False,
							 use_bias=True,
							 name='logits')
					)

		self.temporal = TemporalConvNet(
			num_inputs=1024, num_channels=[512, 512, 512, 512], kernel_size=5, attention=0,
			dropout=0.1)
		self.vregressor = nn.Sequential(nn.Linear(512, 128),
							#nn.Dropout(0.5),
							nn.BatchNorm1d(128),
							nn.Linear(128, 1))
       						#nn.Tanh())
		self.aregressor = nn.Sequential(nn.Linear(512, 128),
							nn.BatchNorm1d(128),
							#nn.Dropout(0.5),
							nn.Linear(128, 1))
		self.dropout = nn.Dropout(0.5)

	def forward(self, x):
		batch_size, C, timesteps, H, W = x.size()
		feature = self.i3d_WSDDA.extract_features(x)
		features = feature.squeeze(3).squeeze(3)
		temp_features = self.temporal(features).transpose(1, 2).contiguous()
		return temp_features
