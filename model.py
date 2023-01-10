#!/usr/bin/env python
# coding: utf-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoModelForImageClassification

# #### TextEncoder

class TextEncoder(nn.Module):
    def __init__(self, ptm_name, device, pretrained, freeze=False):
        super().__init__()
        if pretrained:
            self.model = AutoModel.from_pretrained(ptm_name)
        else:
            self.model = AutoModel.from_config(AutoConfig.from_pretrained(ptm_name))
        self.feat_dim = AutoConfig.from_pretrained(ptm_name).hidden_size

        self.freeze = freeze
        if freeze:
            for name ,param in self.model.named_parameters():
                param.requires_grad = False
        
    def forward(self, inputs):
        if self.freeze:
            self.model.eval()
        last_hidden_state = self.model(**inputs).last_hidden_state # [batch_size, seq_len, hidden_size]
        feature = torch.mean(last_hidden_state, axis=1) # [batch_size, hidden_size]

        feature = F.normalize(feature, dim=-1)
        return feature

# #### ImageEncoder

class ImageEncoder(nn.Module):
    def __init__(self, ptm_name, pretrained):
        super().__init__()
        if pretrained:
            self.model = AutoModelForImageClassification.from_pretrained(ptm_name)
        else:
            self.model = AutoModelForImageClassification.from_config(AutoConfig.from_pretrained(ptm_name))
        self.feat_dim = 1000

    def forward(self, inputs):
        feature = self.model(**inputs).logits # [batch_size, 1000]
        feature = F.normalize(feature, dim=-1)
        return feature



# #### SimpleCLIP

class SimpleCLIP(nn.Module):
    def __init__(self, dim, text_ptm_name, img_ptm_name, device, pretrained, freeze=False):
        super().__init__()
        self.device = device
        self.textencoder = TextEncoder(text_ptm_name, device, pretrained=pretrained, freeze=freeze)
        self.imgencoder = ImageEncoder(img_ptm_name, pretrained=pretrained)

        self.text_projection = nn.Parameter(torch.empty(self.textencoder.feat_dim, dim))
        self.img_projection = nn.Parameter(torch.empty(self.imgencoder.feat_dim, dim))
        self.logit_scale = nn.Parameter(torch.ones([]))
        self.init_parameters()
    
    def init_parameters(self):
        nn.init.constant_(self.logit_scale, np.log(1 / 0.07))
        nn.init.normal_(self.text_projection, std=0.02)
        nn.init.normal_(self.img_projection, std=0.02)

    def loss(self, text_feat, img_feat, logit_scale):
        labels = torch.arange(text_feat.shape[0], device=self.device, dtype=torch.long)

        logits_per_image = logit_scale * img_feat @ text_feat.T   # [batch_size, batch_size]
        logits_per_text = logit_scale * text_feat @ img_feat.T   # [batch_size, batch_size]
        
        total_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
            ) / 2
        return total_loss

    def forward(self, text_inputs, img_inputs, outputLoss=False):
        text_feat = self.textencoder(text_inputs) @ self.text_projection # [batch_size, dim]
        img_feat = self.imgencoder(img_inputs) @ self.img_projection # [batch_size, dim]
        logit_scale = self.logit_scale.exp()
        if outputLoss:
            loss = self.loss(text_feat, img_feat, logit_scale)
            return loss, text_feat, img_feat, logit_scale
        else:
            return text_feat, img_feat, logit_scale