from email.mime import image
from math import comb
from matplotlib.transforms import Transform
from sympy import im
import torch
import torch.nn as nn
import clip
import torch.nn.functional as F
import numpy as np

class LambdaRepformer(nn.Module):
    def __init__(self):
        super(LambdaRepformer, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._init_transformer()
        self._init_cross_attention()
        self._init_layers()
        self.attention_aggregator = AttentionAggregator(398, 512)

    def _init_layers(self):
        self.bert_scene_narrative = nn.Linear(768, 512)
        self.ada_scene_narrative = nn.Linear(1536, 512)
        self.clip_image_linear = nn.Linear(512, 512)
        self.bert_inst = nn.Linear(768, 512)
        self.vit_linear = nn.Linear(768, 512)
        self.dinov2_linear = nn.Linear(1024, 512)
        self.ada_linear = nn.Linear(1536, 512)
        self.clip_inst = nn.Linear(512, 512)
        self.text_linear = nn.Linear(768+512, 512)
        self.fc1 = nn.Linear(512, 128)
        self.batch_norm = nn.BatchNorm1d(128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 2)
        self.conv = nn.Conv2d(1024, 512, kernel_size=1)

    def _init_transformer(self):
        self.transformer = nn.Transformer(
            d_model=512, # 512 (need to change other sections of the code to change value)
            nhead=8, # 8 (embed_dim must be divisible by num_heads)
            num_encoder_layers=4, # 4
            num_decoder_layers=4, # 4
            dim_feedforward=2048, # 2048
            dropout=0.4, # 0.4
            activation='gelu', # 'gelu'
            batch_first=True
        )

    def _init_cross_attention(self):
        self.cross_attention = nn.Transformer(
            d_model=512, # 512 (need to change other sections of the code to change value)
            nhead=8, # 8 (embed_dim must be divisible by num_heads)
            num_encoder_layers=4, # 4
            num_decoder_layers=4, # 4
            dim_feedforward=2048, # 2048
            dropout=0.4, # 0.4
            activation='gelu', # 'gelu'
            batch_first=True
        )

    def forward(self, images, texts):
        inst_bert, clip_inst, ada_inst = self._embed_instructions(texts)
        bert_scene, ada_scene, clip2d_image, clip_image, vit_image, dinov2_image = self._embed_images(images)

        bert_scene_0 = bert_scene[:, 0, :].unsqueeze(1)
        bert_scene_1 = bert_scene[:, 1, :].unsqueeze(1)
        ada_scene_0 = ada_scene[:, 0, :].unsqueeze(1)
        ada_scene_1 = ada_scene[:, 1, :].unsqueeze(1)
        clip2d_image_0 = clip2d_image[:, :196, :]
        clip2d_image_1 = clip2d_image[:, 196:, :]
        clip_image_0 = clip_image[:, 0, :].unsqueeze(1)
        clip_image_1 = clip_image[:, 1, :].unsqueeze(1)
        vit_image_0 = vit_image[:, 0, :].unsqueeze(1)
        vit_image_1 = vit_image[:, 1, :].unsqueeze(1)
        dinov2_image_0 = dinov2_image[:, 0, :].unsqueeze(1)
        dinov2_image_1 = dinov2_image[:, 1, :].unsqueeze(1)

        text_features = torch.cat([clip_inst, ada_inst, inst_bert], dim=1)
        lambda_features_0 = torch.cat([vit_image_0, dinov2_image_0, clip2d_image_0, clip_image_0, ada_scene_0, bert_scene_0], dim=1)
        lambda_features_1 = torch.cat([vit_image_1, dinov2_image_1, clip2d_image_1, clip_image_1, ada_scene_1, bert_scene_1], dim=1)
        lambda_features = self.cross_attention(lambda_features_0, lambda_features_1)
        combined_features = self.transformer(text_features, lambda_features)

        x = self._process_combined_features(combined_features)
        return x

    def _embed_instructions(self, texts):
        inst_bert = self._embed_single(texts["bert"], self.bert_inst, unsqueeze_dim=1)
        clip_inst = self._embed_single(texts["clip"], self.clip_inst, unsqueeze_dim=1)
        ada_inst = self._embed_single(texts["ada"], self.ada_linear, unsqueeze_dim=1)
        return inst_bert, clip_inst, ada_inst

    def _embed_images(self, images):
        bert_scene = self._embed_per_image(images["bert_scene_narratives"], self.bert_scene_narrative)
        ada_scene = self._embed_per_image(images["ada_scene_narratives"], self.ada_scene_narrative)
        clip_image = self._embed_per_image(images["clip_images"].to(self.device), self.clip_image_linear)
        clip2d_image = self._process_clip2d_images(images["clip2d_images"])
        vit_image = self._embed_per_image(images["vit_images"].to(self.device), self.vit_linear)
        dinov2_image = self._embed_per_image(images["dinov2_images"].to(self.device), self.dinov2_linear)
        return bert_scene, ada_scene, clip2d_image, clip_image, vit_image, dinov2_image

    def _embed_single(self, tensor, layer, unsqueeze_dim=None):
        tensor = tensor.to(self.device)
        if unsqueeze_dim is not None:
            tensor = tensor.unsqueeze(unsqueeze_dim)
        return layer(tensor.float()) if layer else tensor

    def _embed_per_image(self, tensor, layer):
        tensor = tensor.to(self.device).to(layer.weight.dtype)
        # print(f"Tensor dtype: {tensor.dtype}")
        # print(f"Layer weight dtype: {layer.weight.dtype}")
        return layer(tensor)

    def _process_clip2d_images(self, tensor):
        tensor = tensor.to(self.device).view(-1, 1024, 14, 14)
        tensor = self.conv(tensor).flatten(2).permute(0, 2, 1)
        return tensor.reshape(-1, 2*196, 512)

    def _process_combined_features(self, features):
        x = self.attention_aggregator(features).squeeze(1)
        x = self.fc1(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class AttentionAggregator(nn.Module):
    def __init__(self, seq_len, d_model):
        super(AttentionAggregator, self).__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.attention_weights = nn.Linear(d_model, 1)

    def forward(self, x):
        scores = self.attention_weights(x).squeeze(-1)  # shape: (B, seq_len)
        attention_weights = F.softmax(scores, dim=1)  # shape: (B, seq_len)
        weighted_average = torch.bmm(attention_weights.unsqueeze(1), x) # shape: (B, 1, d_model)
        return weighted_average
