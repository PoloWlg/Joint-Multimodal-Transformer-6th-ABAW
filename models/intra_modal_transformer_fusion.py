import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, input_dim):
        super(Attention, self).__init__()
        self.W = nn.Linear(input_dim, input_dim)
        self.V = nn.Linear(input_dim, input_dim, bias=False)
        self.tanh = nn.Tanh()
        self.fc = nn.Linear(input_dim, 2)

        self.out_layer1 = nn.Linear(512, 256)
        self.out_layer2 = nn.Linear(256, 64)
        self.out_layer3 = nn.Linear(64, 2)

    def forward(self, x):
        q = self.W(x)
        attn_weights = torch.softmax(self.V(self.tanh(q)), dim=1)
        attended_x = attn_weights * x
        out = self.out_layer1(attended_x)
        out = self.out_layer2(out)
        out = self.out_layer3(out)
        return out


class SequentialEncoder(nn.Sequential):
    def forward(self, x):
        for module in self._modules.values():
            x = module(x)
        return x


class TransformerEncoderBlock(nn.Module):
    def __init__(self, input_dim, num_heads, hidden_dim, num_layers):
        super(TransformerEncoderBlock, self).__init__()
        self.layers = SequentialEncoder(
            *[
                TransformerEncoderLayer(input_dim, num_heads, hidden_dim)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x):
        x = self.layers(x)
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, input_dim, num_heads, hidden_dim):
        super(TransformerEncoderLayer, self).__init__()
        self.attention = nn.MultiheadAttention(input_dim, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.layer_norm2 = nn.LayerNorm(input_dim)

    def forward(self, x):
        # Apply self-attention
        attn_output, _ = self.attention(x, x, x)
        x = x + attn_output
        x = self.layer_norm1(x)

        # Apply feed forward network
        ff_output = self.feed_forward(x)
        x = x + ff_output
        x = self.layer_norm2(x)
        return x


class Intra_modal_transformer_fusion(nn.Module):
    def __init__(self, feat_dim, num_heads, hidden_dim, num_layers,
                 reduce_dim_for_audio=False):
        super(Intra_modal_transformer_fusion, self).__init__()
        self.final_visual_encoder = TransformerEncoderBlock(
            feat_dim, num_heads, hidden_dim, num_layers
        )
        self.final_self_attention = nn.MultiheadAttention(512, num_heads)
        self.fc = nn.Linear(768, 512)

    def forward(self, features_a, features_b):
        # todo: make it support a list of features not just 2. the list could
        #  hold an unknown number of features.
        # reduce dim for wavLM features of 768 to 512
        if features_a.shape[-1] == 768:
            features_a = self.fc(features_a)
        if features_b.shape[-1] == 768:
            features_b = self.fc(features_b)

        stack_attention = torch.stack((features_a, features_b), dim=2)
        # stack_attention = stack_attention.permute(1, 0, 2, 3)
        b_size = stack_attention.shape[0]
        seq_size = stack_attention.shape[1]
        stack_attention_flatten = stack_attention.flatten(0, 1).permute(1, 0, 2)
        stack_attention_flatten = stack_attention_flatten
        final_encoded = self.final_visual_encoder(stack_attention_flatten)

        final_attention, _ = self.final_self_attention(final_encoded,
                                                       final_encoded,
                                                       final_encoded)
        final_attention = final_attention.permute(1, 0, 2)
        final_attention_unflatten = final_attention.unflatten(0, (
        b_size, seq_size))

        final_attention_unflatten = final_attention_unflatten[:, :, -1, :]
        # final_attention_unflatten = final_attention_unflatten.permute(1, 0, 2)

        return final_attention_unflatten
