import torch.nn as nn


__all__ = ['FcLayer']

class FcLayer(nn.Module):
	def __init__(self, input_dim, output_dim):
		super(FcLayer, self).__init__()
		self.fc_layer = nn.Linear(input_dim, output_dim)

	def forward(self, x):
		return self.fc_layer(x)
