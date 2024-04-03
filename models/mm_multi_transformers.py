import sys

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
            *[TransformerEncoderLayer(input_dim, num_heads, hidden_dim)
              for _ in range(num_layers)])

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
            nn.Linear(hidden_dim, input_dim)
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


class MultimodalTransformer_w_JR(nn.Module):
    def __init__(self, visual_dim, audio_dim, num_heads, hidden_dim,
                 num_layers, output_format: str):
        super(MultimodalTransformer_w_JR, self).__init__()

        assert output_format in ['FC', 'SELF_ATTEN'], output_format
        self.output_format = output_format

        # Encoder blocks
        self.visual_encoder = TransformerEncoderBlock(visual_dim, num_heads,
                                                      hidden_dim, num_layers)
        self.physiological_encoder = TransformerEncoderBlock(audio_dim,
                                                             num_heads,
                                                             hidden_dim,
                                                             num_layers)
        self.joint_representation_encoder = TransformerEncoderBlock(audio_dim,
                                                                    num_heads,
                                                                    hidden_dim,
                                                                    num_layers)
        self.final_encoder = TransformerEncoderBlock(3072, num_heads,
                                                     hidden_dim, num_layers)

        # Cross attention
        self.cross_attention_v = nn.MultiheadAttention(visual_dim, num_heads)
        self.cross_attention_p = nn.MultiheadAttention(audio_dim, num_heads)
        self.cross_attention_pv = nn.MultiheadAttention(512, num_heads)

        # Fully connected layer for joint representation
        self.out_layer_pv = nn.Linear(1024, 512)

        if output_format == 'FC':
            # Fully connected layer for the final output
            self.out_layer1 = nn.Linear(3072, 1024)

        elif output_format == 'SELF_ATTEN':
            # Final attention module
            self.final_visual_encoder = TransformerEncoderBlock(visual_dim,
                                                                num_heads,
                                                                hidden_dim,
                                                                num_layers)
            self.final_self_attention = nn.MultiheadAttention(512, num_heads)

        else:
            raise NotImplementedError(output_format)

    def forward(self, visual_features, physiological_features):
        # Concatenate the visual and physiological features
        joint_representation = torch.cat(
            (visual_features, physiological_features), dim=2)

        # Decrease the dimensionality of the joint representation
        joint_representation = self.out_layer_pv(joint_representation)

        # Permute dimension from (batch, seq, feature) to (seq, batch, feature)
        visual_features = visual_features.permute(1, 0, 2)
        physiological_features = physiological_features.permute(1, 0, 2)
        joint_representation = joint_representation.permute(1, 0, 2)

        # Pass the visual, physiological and joint representation features through their respective encoders
        visual_encoded = self.visual_encoder(visual_features)
        physiological_encoded = self.physiological_encoder(
            physiological_features)
        joint_representation_encoded = self.joint_representation_encoder(
            joint_representation)
        # visual_encoded = visual_encoded.permute(1, 0, 2)
        # physiological_encoded = physiological_encoded.permute(1, 0, 2)
        # joint_representation_encoded = joint_representation_encoded.permute(1, 0, 2)

        # Do all the cross-attention between the visual encoded and physio encoded features
        cross_attention_output_v_p, _ = self.cross_attention_v(visual_encoded,
                                                               physiological_encoded,
                                                               physiological_encoded)

        # Do all the cross-attention between the physio encoded and visio encoded features
        cross_attention_output_p_v, _ = self.cross_attention_p(
            physiological_encoded, visual_encoded, visual_encoded)

        # Do all the cross-attention between the joint representation encoded and visio encoded features
        cross_attention_output_pv_v, _ = self.cross_attention_pv(
            joint_representation_encoded, visual_encoded, visual_encoded)

        # Do all the cross-attention between the visio encoded and joint representation encoded features
        cross_attention_output_v_pv, _ = self.cross_attention_v(visual_encoded,
                                                                joint_representation_encoded,
                                                                joint_representation_encoded)

        # Do all the cross-attention between the joint representation encoded and physio encoded features
        cross_attention_output_pv_p, _ = self.cross_attention_pv(
            joint_representation_encoded, physiological_encoded,
            physiological_encoded)

        # Do all the cross-attention between the physio encoded and joint representation encoded features
        cross_attention_output_p_pv, _ = self.cross_attention_p(
            physiological_encoded, joint_representation_encoded,
            joint_representation_encoded)

        if self.output_format == "SELF_ATTEN":
            '''
             --- [Start] Final Attention module ---
            '''
            stack_attention = torch.stack((cross_attention_output_v_p,
                                           cross_attention_output_p_v,
                                           cross_attention_output_pv_v,
                                           cross_attention_output_v_pv,
                                           cross_attention_output_pv_p,
                                           cross_attention_output_p_pv), dim=2)
            stack_attention = stack_attention.permute(1, 0, 2, 3)
            stack_attention_flatten = stack_attention.flatten(0, 1).permute(1, 0, 2)
            stack_attention_flatten = stack_attention_flatten
            b_size = stack_attention.shape[0]
            seq_size = stack_attention.shape[1]
            final_encoded = self.final_visual_encoder(stack_attention_flatten)

            final_attention, _ = self.final_self_attention(final_encoded,
                                                           final_encoded,
                                                           final_encoded)
            final_attention = final_attention.permute(1, 0, 2)
            final_attention_unflatten = final_attention.unflatten(0, (
            b_size, seq_size))

            final_attention_unflatten = final_attention_unflatten[:, :, -1, :]
            # bsz, seq, 512.
            '''
             --- [End] Final Attention module ---
            '''

            return final_attention_unflatten

        elif self.output_format == 'FC':
            # Concatenate Cross-attention outputs
            concat_attention = torch.cat((cross_attention_output_v_p,
                                          cross_attention_output_p_v,
                                          cross_attention_output_pv_v,
                                          cross_attention_output_v_pv,
                                          cross_attention_output_pv_p,
                                          cross_attention_output_p_pv), dim=2)
            out = self.out_layer1(concat_attention)  # bsz, seq, 1024

            return out

        else:
            raise NotImplementedError(self.output_format)


class FeatureConcatFC(nn.Module):
    def __init__(self, visual_dim, audio_dim,):
        super(FeatureConcatFC, self).__init__()
        self.fc = nn.Linear(visual_dim + audio_dim, 512)

    def forward(self, visual_features, audio_features):
        out = torch.cat((visual_features, audio_features), dim=2)
        out = self.fc(out)
        return out