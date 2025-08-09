"""
Model definitions for the vision-text integration pipeline.
Includes MedViT for vision processing and VisionTextModel for combining vision and text embeddings.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
from functools import partial
import math
import os
from einops import rearrange
from timm.layers import DropPath, trunc_normal_
import logging
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["NUMEXPR_MAX_THREADS"] = "8"

NORM_EPS = 1e-5

# Helper classes for MedViT
class ConvBNReLU(nn.Module):
    """Convolution-BatchNorm-ReLU block for efficient feature extraction."""
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                              padding=1, groups=groups, bias=False)
        self.norm = nn.BatchNorm2d(out_channels, eps=NORM_EPS)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x

def _make_divisible(v, divisor, min_value=None):
    """Ensure channel counts are divisible by a specified divisor."""
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class PatchEmbed(nn.Module):
    """Patch embedding layer with optional downsampling."""
    def __init__(self, in_channels, out_channels, stride=1):
        super(PatchEmbed, self).__init__()
        norm_layer = partial(nn.BatchNorm2d, eps=NORM_EPS)
        if stride == 2:
            self.avgpool = nn.AvgPool2d((2, 2), stride=2, ceil_mode=True, count_include_pad=False)
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
            self.norm = norm_layer(out_channels)
        elif in_channels != out_channels:
            self.avgpool = nn.Identity()
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
            self.norm = norm_layer(out_channels)
        else:
            self.avgpool = nn.Identity()
            self.conv = nn.Identity()
            self.norm = nn.Identity()

    def forward(self, x):
        return self.norm(self.conv(self.avgpool(x)))

class MHCA(nn.Module):
    """Multi-Head Channel Attention for local feature aggregation."""
    def __init__(self, out_channels, head_dim):
        super(MHCA, self).__init__()
        norm_layer = partial(nn.BatchNorm2d, eps=NORM_EPS)
        self.group_conv3x3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1,
                                       padding=1, groups=out_channels // head_dim, bias=False)
        self.norm = norm_layer(out_channels)
        self.act = nn.ReLU(inplace=True)
        self.projection = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.group_conv3x3(x)
        out = self.norm(out)
        out = self.act(out)
        out = self.projection(out)
        return out

class h_sigmoid(nn.Module):
    """Hard sigmoid activation function."""
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    """Hard swish activation function."""
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class ECALayer(nn.Module):
    """Efficient Channel Attention layer."""
    def __init__(self, channel, gamma=2, b=1, sigmoid=True):
        super(ECALayer, self).__init__()
        t = int(abs((math.log(channel, 2) + b) / gamma))
        k = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.sigmoid = nn.Sigmoid() if sigmoid else h_sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        y = y.transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)

class SELayer(nn.Module):
    """Squeeze-and-Excitation layer."""
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            h_sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class LocalityFeedForward(nn.Module):
    """Locality-aware feed-forward network with depth-wise convolutions."""
    def __init__(self, in_dim, out_dim, stride, expand_ratio=4., act='hs+se', reduction=4,
                 wo_dp_conv=False, dp_first=False):
        super(LocalityFeedForward, self).__init__()
        hidden_dim = int(in_dim * expand_ratio)
        kernel_size = 3
        layers = [
            nn.Conv2d(in_dim, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            h_swish() if act.find('hs') >= 0 else nn.ReLU6(inplace=True)
        ]
        if not wo_dp_conv:
            dp = [
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, kernel_size // 2, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                h_swish() if act.find('hs') >= 0 else nn.ReLU6(inplace=True)
            ]
            if dp_first:
                layers = dp + layers
            else:
                layers.extend(dp)
        if act.find('+') >= 0:
            attn = act.split('+')[1]
            if attn == 'se':
                layers.append(SELayer(hidden_dim, reduction=reduction))
            elif attn.find('eca') >= 0:
                layers.append(ECALayer(hidden_dim, sigmoid=attn == 'eca'))
            else:
                raise NotImplementedError(f'Activation type {act} is not implemented')
        layers.extend([
            nn.Conv2d(hidden_dim, out_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_dim)
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        x = x + self.conv(x)
        return x

class ECB(nn.Module):
    """Efficient Convolution Block for MedViT."""
    def __init__(self, in_channels, out_channels, stride=1, path_dropout=0, drop=0, head_dim=32, mlp_ratio=3):
        super(ECB, self).__init__()
        norm_layer = partial(nn.BatchNorm2d, eps=NORM_EPS)
        self.patch_embed = PatchEmbed(in_channels, out_channels, stride)
        self.mhca = MHCA(out_channels, head_dim)
        self.attention_path_dropout = DropPath(path_dropout)
        self.conv = LocalityFeedForward(out_channels, out_channels, 1, mlp_ratio, reduction=out_channels)
        self.norm = norm_layer(out_channels)

    def forward(self, x):
        x = self.patch_embed(x)
        x = x + self.attention_path_dropout(self.mhca(x))
        x = self.norm(x)
        x = x + self.conv(x)
        return x

class E_MHSA(nn.Module):
    """Efficient Multi-Head Self-Attention for MedViT."""
    def __init__(self, dim, out_dim=None, head_dim=32, qkv_bias=True, qk_scale=None,
                 attn_drop=0, proj_drop=0., sr_ratio=1):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim if out_dim is not None else dim
        self.num_heads = self.dim // head_dim
        self.scale = qk_scale or head_dim ** -0.5
        self.q = nn.Linear(dim, self.dim, bias=qkv_bias)
        self.k = nn.Linear(dim, self.dim, bias=qkv_bias)
        self.v = nn.Linear(dim, self.dim, bias=qkv_bias)
        self.proj = nn.Linear(self.dim, self.out_dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.AvgPool1d(kernel_size=sr_ratio ** 2, stride=sr_ratio ** 2)
            self.norm = nn.BatchNorm1d(dim, eps=NORM_EPS)

    def forward(self, x):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        if self.sr_ratio > 1:
            x_ = x.transpose(1, 2)
            x_ = self.sr(x_)
            x_ = self.norm(x_)
            x_ = x_.transpose(1, 2)
            k = self.k(x_).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 3, 1)
            v = self.v(x_).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        else:
            k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 3, 1)
            v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        attn = (q @ k) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class LTB(nn.Module):
    """Local-Transformer Block for MedViT."""
    def __init__(self, in_channels, out_channels, path_dropout, stride=1, sr_ratio=1,
                 mlp_ratio=2, head_dim=32, mix_block_ratio=0.75, attn_drop=0, drop=0):
        super(LTB, self).__init__()
        norm_func = partial(nn.BatchNorm2d, eps=NORM_EPS)
        self.mhsa_out_channels = _make_divisible(int(out_channels * mix_block_ratio), 32)
        self.mhca_out_channels = out_channels - self.mhsa_out_channels
        self.patch_embed = PatchEmbed(in_channels, self.mhsa_out_channels, stride)
        self.norm1 = norm_func(self.mhsa_out_channels)
        self.e_mhsa = E_MHSA(self.mhsa_out_channels, head_dim=head_dim, sr_ratio=sr_ratio,
                             attn_drop=attn_drop, proj_drop=drop)
        self.mhsa_path_dropout = DropPath(path_dropout)
        self.projection = PatchEmbed(self.mhsa_out_channels, self.mhca_out_channels, stride=1)
        self.mhca = MHCA(self.mhca_out_channels, head_dim=head_dim)
        self.mhca_path_dropout = DropPath(path_dropout * (1 - mix_block_ratio))
        self.norm2 = norm_func(out_channels)
        self.conv = LocalityFeedForward(out_channels, out_channels, 1, mlp_ratio, reduction=out_channels)

    def forward(self, x):
        x = self.patch_embed(x)
        B, C, H, W = x.shape
        out = self.norm1(x)
        out = rearrange(out, "b c h w -> b (h w) c")
        out = self.mhsa_path_dropout(self.e_mhsa(out))
        x = x + rearrange(out, "b (h w) c -> b c h w", h=H)
        out = self.projection(x)
        out = out + self.mhca_path_dropout(self.mhca(out))
        x = torch.cat([x, out], dim=1)
        out = self.norm2(x)
        x = x + self.conv(out)
        return x

class MedViT(nn.Module):
    """
    MedViT: A vision transformer model designed for medical image processing.
    Reference: https://github.com/Omid-Nejati/MedViT
    """
    def __init__(self, stem_chs, depths, path_dropout=0.2, num_classes=None,
                 strides=[1, 2, 2, 2], sr_ratios=[8, 4, 2, 1], head_dim=32, mix_block_ratio=0.75):
        super(MedViT, self).__init__()
        self.stage_out_channels = [[96] * depths[0],
                                   [192] * (depths[1] - 1) + [256],
                                   [384, 384, 384, 384, 512] * (depths[2] // 5),
                                   [768] * (depths[3] - 1) + [1024]]
        self.stage_block_types = [[ECB] * depths[0],
                                  [ECB] * (depths[1] - 1) + [LTB],
                                  [ECB, ECB, ECB, ECB, LTB] * (depths[2] // 5),
                                  [ECB] * (depths[3] - 1) + [LTB]]
        
        # Initialize stem
        self.stem = nn.Sequential(
            ConvBNReLU(3, stem_chs[0], kernel_size=3, stride=2),
            ConvBNReLU(stem_chs[0], stem_chs[1], kernel_size=3, stride=1),
            ConvBNReLU(stem_chs[1], stem_chs[2], kernel_size=3, stride=1),
            ConvBNReLU(stem_chs[2], stem_chs[2], kernel_size=3, stride=2),
        )
        
        # Initialize stages
        input_channel = stem_chs[-1]
        features = []
        idx = 0
        dpr = [x.item() for x in torch.linspace(0, path_dropout, sum(depths))]
        for stage_id in range(len(depths)):
            numrepeat = depths[stage_id]
            output_channels = self.stage_out_channels[stage_id]
            block_types = self.stage_block_types[stage_id]
            for block_id in range(numrepeat):
                stride = strides[stage_id] if block_id == 0 else 1
                output_channel = output_channels[block_id]
                block_type = block_types[block_id]
                if block_type is ECB:
                    layer = ECB(input_channel, output_channel, stride=stride, path_dropout=dpr[idx + block_id])
                elif block_type is LTB:
                    layer = LTB(input_channel, output_channel, path_dropout=dpr[idx + block_id], stride=stride,
                                sr_ratio=sr_ratios[stage_id], head_dim=head_dim, mix_block_ratio=mix_block_ratio)
                features.append(layer)
                input_channel = output_channel
            idx += numrepeat
        self.features = nn.Sequential(*features)
        
        # Final norm and avgpool
        self.norm = nn.BatchNorm2d(output_channel, eps=NORM_EPS)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Linear projection head if num_classes is provided
        if num_classes is not None:
            self.proj_head = nn.Linear(output_channel, num_classes)
        else:
            self.proj_head = None
        
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights using Kaiming and truncated normal initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Forward pass through MedViT, producing a flattened feature vector."""
        x = self.stem(x)
        for layer in self.features:
            x = layer(x)
        x = self.norm(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        if self.proj_head is not None:
            x = self.proj_head(x)
        return x

    def load_pretrained_weights(self, weight_path):
        """Load pretrained weights for MedViT, ignoring classification head if num_classes=None."""
        state_dict = torch.load(weight_path, map_location='cpu')
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        if self.proj_head is None:
            state_dict = {k: v for k, v in state_dict.items() if not k.startswith('proj_head')}
        self.load_state_dict(state_dict, strict=False)

class TopicAttention(nn.Module):
    """Visual and Semantic Topic Attention to emphasize key tokens in text embeddings."""
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super(TopicAttention, self).__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x, attention_mask=None):
        B, N, C = x.shape  # [batch, seq_len, dim]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [batch, heads, seq_len, dim/head]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [batch, heads, seq_len, seq_len]
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)  # [batch, 1, 1, seq_len]
            attn = attn.masked_fill(attention_mask == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class VisionTextModel(nn.Module):
    """
    VisionTextModel: Integrates MedViT vision features with BioGpt text embeddings for multimodal captioning.
    Supports contrastive loss and visual/semantic topic attention.
    """
    def __init__(self, vision_model, text_model, vision_dim, text_dim, config):
        super(VisionTextModel, self).__init__()
        self.vision_model = vision_model
        self.text_model = text_model
        self.config = config
        self.vision_dim = vision_dim
        self.text_dim = text_dim
        
        # Always create vision projection to match text dimension
        self.vision_projection = nn.Linear(vision_dim, text_dim)
        
        # Vision token embedding for learnable special tokens
        self.vision_token_embed = nn.Parameter(torch.randn(1, 1, text_dim))
            
        # Projection heads for contrastive loss (if enabled)
        if config.CONS_LOSS:
            self.vision_proj = nn.Linear(vision_dim, 512)
            self.text_proj = nn.Linear(text_dim, 512)
            self.temperature = nn.Parameter(torch.tensor(0.07))
        
        # Topic attention module (if enabled)
        if config.TOPIC_ATTENTION:
            self.topic_attention = TopicAttention(text_dim, num_heads=8, dropout=0.1)
        
        logger.info(f"VisionTextModel initialized with vision_dim={vision_dim}, text_dim={text_dim}, "
                    f"cons_loss={config.CONS_LOSS}, topic_attention={config.TOPIC_ATTENTION}")

    def contrastive_loss(self, vision_features, text_features):
        """Compute InfoNCE contrastive loss between vision and text embeddings."""
        vision_embed = self.vision_proj(vision_features)  # [batch, 512]
        text_embed = self.text_proj(text_features)       # [batch, 512]
        
        # Normalize embeddings
        vision_embed = F.normalize(vision_embed, dim=-1)
        text_embed = F.normalize(text_embed, dim=-1)
        
        # Compute similarity matrix
        logits = torch.matmul(vision_embed, text_embed.T) / self.temperature  # [batch, batch]
        
        # Labels: diagonal elements are positive pairs
        batch_size = vision_embed.size(0)
        labels = torch.arange(batch_size, device=vision_embed.device)
        
        # InfoNCE loss (symmetric)
        loss_v2t = F.cross_entropy(logits, labels)
        loss_t2v = F.cross_entropy(logits.T, labels)
        return (loss_v2t + loss_t2v) / 2

    def forward(self, pixel_values, input_ids, attention_mask):
        """Forward pass combining vision and text embeddings."""
        try:
            vision_features = self.vision_model(pixel_values)
            batch_size = pixel_values.size(0)
            
            text_embed = self.text_model.get_input_embeddings()(input_ids)
            
            if self.config.TOPIC_ATTENTION:
                text_embed = self.topic_attention(text_embed, attention_mask=attention_mask)

            vision_projected = self.vision_projection(vision_features)
            vision_embed = vision_projected.unsqueeze(1)
            vision_embed = vision_embed + self.vision_token_embed.expand(batch_size, -1, -1)
            
            combined_embed = torch.cat([vision_embed, text_embed], dim=1)
            
            vision_mask = torch.ones(batch_size, 1, device=attention_mask.device)
            combined_mask = torch.cat([vision_mask, attention_mask], dim=1)
            
            labels = input_ids.clone()
            labels[labels == self.text_model.config.pad_token_id] = -100
            vision_label = torch.full((batch_size, 1), -100, dtype=labels.dtype, device=labels.device)
            adjusted_labels = torch.cat([vision_label, labels], dim=1)
            
            outputs = self.text_model(
                inputs_embeds=combined_embed,
                attention_mask=combined_mask,
                labels=adjusted_labels
            )
            lm_loss = outputs.loss

            cont_loss = torch.tensor(0.0, device=lm_loss.device)
            if self.config.CONS_LOSS:
                text_features = torch.mean(text_embed, dim=1)
                cont_loss = self.contrastive_loss(vision_features, text_features)

            total_loss = lm_loss
            if self.config.CONS_LOSS:
                total_loss = total_loss + self.config.CONT_LOSS_WEIGHT * cont_loss

            return {
                'lm_loss': lm_loss,
                'cont_loss': cont_loss,
                'total_loss': total_loss
            }

        except Exception as e:
            logger.error(f"Error in VisionTextModel.forward: {str(e)}")
            raise
        
        # ADD THIS NEW METHOD RIGHT AFTER THE FORWARD METHOD
    def generate_caption(self, pixel_values, tokenizer, max_length=128, num_beams=1, early_stopping=False):
        """Generate caption for given images during inference/validation."""
        with torch.no_grad():
            vision_features = self.vision_model(pixel_values)  # [batch, vision_dim]
            
            # Apply vision projection to match text dimension
            if hasattr(self, 'vision_projection'):
                vision_projected = self.vision_projection(vision_features)
            else:
                # Fallback if projection doesn't exist (shouldn't happen with our fix #2)
                vision_projected = vision_features
                
            vision_embed = vision_projected.unsqueeze(1)  # [batch, 1, text_dim]
            
            # Add learnable vision token if it exists
            if hasattr(self, 'vision_token_embed'):
                batch_size = pixel_values.size(0)
                vision_embed = vision_embed + self.vision_token_embed.expand(batch_size, -1, -1)
            
            # Create attention mask for vision tokens
            attention_mask = torch.ones(vision_embed.size(0), vision_embed.size(1), 
                                      device=vision_embed.device)
            
            # Generate with proper tokenizer settings
            generated_ids = self.text_model.generate(
                inputs_embeds=vision_embed,
                attention_mask=attention_mask,
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=early_stopping,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=False  # Ensure deterministic generation for baseline
            )
            return generated_ids