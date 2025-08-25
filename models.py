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
from utils import merge_pre_bn
import torch.utils.checkpoint as checkpoint

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


#---------------------MEDVIT---------------------

NORM_EPS = 1e-5 # to avoid divison by zero

# Stem stage (Stage 0)
class ConvBNReLU(nn.Module):
    """Convolution-BatchNorm-ReLU block for efficient feature extraction."""
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                              padding=1, groups=groups, bias=False) # bias is not needed because of batch normalization
        self.norm = nn.BatchNorm2d(out_channels, eps=NORM_EPS)
        self.act = nn.ReLU(inplace=True) # inplace means the activation overwrites its input tensor memory for memory-saving

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x

# For Patch Embedding (Stage 1 to 4)
class PatchEmbed(nn.Module):
    """Patch embedding layer with optional downsampling."""
    def __init__(self, in_channels, out_channels, stride=1):
        super(PatchEmbed, self).__init__()
        norm_layer = partial(nn.BatchNorm2d, eps=NORM_EPS)
        if stride == 2:
            self.avgpool = nn.AvgPool2d((2, 2), stride=2, ceil_mode=True, count_include_pad=False) #ceil_mode is True to ensure the pooling covers all pixels, and count include pad is false to avoid artificially lowering averages near edges when padding is present
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
    """
    Multi-Head Convolutional Attention | Group convolution
    """
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
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6
    
class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class SELayer(nn.Module): # Squeeze and excitation layer
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1) # Takes average of entire spatial dimension (B, C, H, W) → (B, C, 1, 1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction), #Compress
                nn.ReLU(inplace=True), #Activate
                nn.Linear(channel // reduction, channel), #Expand
                h_sigmoid() #Normalize
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()  # (Batch, Channels, Height, Width)    
        # Step 1: SQUEEZE - Global average pooling
        y = self.avg_pool(x).view(b, c)  # (B, C, H, W) → (B, C, 1, 1) → (B, C)
        # Step 2: EXCITATION - Learn channel importance
        y = self.fc(y).view(b, c, 1, 1)  # (B, C) → (B, C/r) → (B, C) → (B, C, 1, 1)
        # Step 3: SCALE - Apply attention weights
        return x * y  # Element-wise multiplication
    
    
class ECALayer(nn.Module): #Efficient Channel Attention
    def __init__(self, channel, gamma=2, b=1, sigmoid=True):
        super(ECALayer, self).__init__()
        t = int(abs((math.log(channel, 2) + b) / gamma))
        k = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        if sigmoid:
            self.sigmoid = nn.Sigmoid()
        else:
            self.sigmoid = h_sigmoid()

    def forward(self, x):
            # Step 1: Global average pooling (same as SE)
            y = self.avg_pool(x)  # (B, C, H, W) → (B, C, 1, 1)    
            # Step 2: Reshape for 1D convolution  
            y = self.conv(y.squeeze(-1).transpose(-1, -2))  # Magic happens here!
            # Step 3: Reshape back and apply attention
            y = y.transpose(-1, -2).unsqueeze(-1)
            y = self.sigmoid(y)
            return x * y.expand_as(x)
        
class LocalityFeedForward(nn.Module):
    def __init__(self, in_dim, out_dim, stride, expand_ratio=4., act='hs+se', reduction=4,
                 wo_dp_conv=False, dp_first=False):
        """
        expand_ratio: expansion ratio of the hidden dimension.
        act: the activation function.
                    relu: ReLU
                    hs: h_swish
                    hs+se: h_swish and SE module
                    hs+eca: h_swish and ECA module
                    hs+ecah: h_swish and ECA module. Compared with eca, h_sigmoid is used.
        reduction: reduction rate in SE module.
        wo_dp_conv: without depth-wise convolution.
        dp_first: place depth-wise convolution as the first layer.
        """
        super(LocalityFeedForward, self).__init__()
        hidden_dim = int(in_dim * expand_ratio)
        kernel_size = 3

        layers = []
        # the first linear layer is replaced by 1x1 convolution.
        # Layer sequence
        layers.extend([
            # EXPANSION PHASE
            nn.Conv2d(in_dim, hidden_dim, 1, 1, 0, bias=False), # 1*1 Conv
            nn.BatchNorm2d(hidden_dim), # Normalizze
            h_swish() if act.find('hs') >= 0 else nn.ReLU6(inplace=True)]) # Activate

        # the depth-wise convolution between the two linear layers
        if not wo_dp_conv:
            # SPATIAL PROCESSING PHASE
            dp = [
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, kernel_size // 2, groups=hidden_dim, bias=False), # Depthwise conv 
                nn.BatchNorm2d(hidden_dim), #Normalize
                h_swish() if act.find('hs') >= 0 else nn.ReLU6(inplace=True) #Activate
            ]
            if dp_first:
                layers = dp + layers
            else:
                layers.extend(dp)

        if act.find('+') >= 0:
            attn = act.split('+')[1]
            if attn == 'se':
                # ATTENTION PHASE
                layers.append(SELayer(hidden_dim, reduction=reduction))
            elif attn.find('eca') >= 0:
                layers.append(ECALayer(hidden_dim, sigmoid=attn == 'eca'))
            else:
                raise NotImplementedError('Activation type {} is not implemented'.format(act))

        # the second linear layer is replaced by 1x1 convolution.
        # COMPRESSION PHASE
        layers.extend([
            nn.Conv2d(hidden_dim, out_dim, 1, 1, 0, bias=False), #1*1 Conv
            nn.BatchNorm2d(out_dim) # Normalize
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        x = x + self.conv(x) # Residual | preserves original information
        return x

# ECB = Complete Medical Examination
# ├── MHCA = Multiple specialists examine image simultaneously
# ├── LocalityFeedForward = Detailed analysis of specific regions
# ├── SE/ECA Layer = Prioritize most important findings
# └── Residual Connections = Don't forget original observations

class ECB(nn.Module):
    """
    Efficient Convolution Block
    """
    def __init__(self, in_channels, out_channels, stride=1, path_dropout=0,
                 drop=0, head_dim=32, mlp_ratio=3):
        super(ECB, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        norm_layer = partial(nn.BatchNorm2d, eps=NORM_EPS)
        assert out_channels % head_dim == 0
        self.patch_embed = PatchEmbed(in_channels, out_channels, stride) # Changes channel dimensions if needed
        self.mhca = MHCA(out_channels, head_dim) # Captures important spatial relationships
        self.attention_path_dropout = DropPath(path_dropout)
        self.conv = LocalityFeedForward(out_channels, out_channels, 1, mlp_ratio, reduction=out_channels) # Refines local features | Local Processing
        self.norm = norm_layer(out_channels)
        #self.mlp = Mlp(out_channels, mlp_ratio=mlp_ratio, drop=drop, bias=True)
        #self.mlp_path_dropout = DropPath(path_dropout)
        self.is_bn_merged = False

    def merge_bn(self): # Batch Normalization merging | Deployment optimization
        # During training: Keep BatchNorm separate for proper statistics 
        # During inference: Merge BatchNorm into convolution weights for faster computation
        if not self.is_bn_merged:
            self.mlp.merge_bn(self.norm)
            self.is_bn_merged = True

    def forward(self, x):
        x = self.patch_embed(x) # Change channels if needed
        x = x + self.attention_path_dropout(self.mhca(x)) # Residual Connection | attention + residual
        if not torch.onnx.is_in_onnx_export() and not self.is_bn_merged: # Normalize for training
            out = self.norm(x) 
        else:
            out = x # Use raw for interface
        #x = x + self.mlp_path_dropout(self.mlp(out))
        x = x + self.conv(out) # (B, dim, 14, 14) # Maintain information flow | Local processing + residual
        return x
    
def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class E_MHSA(nn.Module):
    """
    Efficient Multi-Head Self Attention
    """
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
        self.N_ratio = sr_ratio ** 2
        if sr_ratio > 1:
            self.sr = nn.AvgPool1d(kernel_size=self.N_ratio, stride=self.N_ratio)
            self.norm = nn.BatchNorm1d(dim, eps=NORM_EPS)
        self.is_bn_merged = False

    def merge_bn(self, pre_bn):
        merge_pre_bn(self.q, pre_bn)
        if self.sr_ratio > 1:
            merge_pre_bn(self.k, pre_bn, self.norm)
            merge_pre_bn(self.v, pre_bn, self.norm)
        else:
            merge_pre_bn(self.k, pre_bn)
            merge_pre_bn(self.v, pre_bn)
        self.is_bn_merged = True

    def forward(self, x):
        B, N, C = x.shape
        q = self.q(x)
        q = q.reshape(B, N, self.num_heads, int(C // self.num_heads)).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.transpose(1, 2)
            x_ = self.sr(x_)
            if not torch.onnx.is_in_onnx_export() and not self.is_bn_merged:
                x_ = self.norm(x_)
            x_ = x_.transpose(1, 2)
            k = self.k(x_)
            k = k.reshape(B, -1, self.num_heads, int(C // self.num_heads)).permute(0, 2, 3, 1)
            v = self.v(x_)
            v = v.reshape(B, -1, self.num_heads, int(C // self.num_heads)).permute(0, 2, 1, 3)
        else:
            k = self.k(x)
            k = k.reshape(B, -1, self.num_heads, int(C // self.num_heads)).permute(0, 2, 3, 1)
            v = self.v(x)
            v = v.reshape(B, -1, self.num_heads, int(C // self.num_heads)).permute(0, 2, 1, 3)
        attn = (q @ k) * self.scale

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
class LTB(nn.Module):
    """
    Local Transformer Block
    """
    def __init__(
            self, in_channels, out_channels, path_dropout, stride=1, sr_ratio=1,
            mlp_ratio=2, head_dim=32, mix_block_ratio=0.75, attn_drop=0, drop=0,
    ):
        super(LTB, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mix_block_ratio = mix_block_ratio
        norm_func = partial(nn.BatchNorm2d, eps=NORM_EPS)

        self.mhsa_out_channels = _make_divisible(int(out_channels * mix_block_ratio), 32)
        self.mhca_out_channels = out_channels - self.mhsa_out_channels

        self.patch_embed = PatchEmbed(in_channels, self.mhsa_out_channels, stride)
        self.norm1 = norm_func(self.mhsa_out_channels)
        self.e_mhsa = E_MHSA(self.mhsa_out_channels, head_dim=head_dim, sr_ratio=sr_ratio,
                             attn_drop=attn_drop, proj_drop=drop)
        self.mhsa_path_dropout = DropPath(path_dropout * mix_block_ratio)

        self.projection = PatchEmbed(self.mhsa_out_channels, self.mhca_out_channels, stride=1)
        self.mhca = MHCA(self.mhca_out_channels, head_dim=head_dim)
        self.mhca_path_dropout = DropPath(path_dropout * (1 - mix_block_ratio))

        self.norm2 = norm_func(out_channels)
        self.conv = LocalityFeedForward(out_channels, out_channels, 1, mlp_ratio, reduction=out_channels)

        #self.mlp = Mlp(out_channels, mlp_ratio=mlp_ratio, drop=drop)
        #self.mlp_path_dropout = DropPath(path_dropout)

        self.is_bn_merged = False

    def merge_bn(self):
        if not self.is_bn_merged:
            self.e_mhsa.merge_bn(self.norm1)
            self.mlp.merge_bn(self.norm2)
            self.is_bn_merged = True

    def forward(self, x):
        x = self.patch_embed(x)
        B, C, H, W = x.shape
        if not torch.onnx.is_in_onnx_export() and not self.is_bn_merged:
            out = self.norm1(x)
        else:
            out = x
        out = rearrange(out, "b c h w -> b (h w) c")  # b n c
        out = self.mhsa_path_dropout(self.e_mhsa(out))
        x = x + rearrange(out, "b (h w) c -> b c h w", h=H)

        out = self.projection(x)
        out = out + self.mhca_path_dropout(self.mhca(out))
        x = torch.cat([x, out], dim=1)

        if not torch.onnx.is_in_onnx_export() and not self.is_bn_merged:
            out = self.norm2(x)
        else:
            out = x
        x = x + self.conv(out)
        #x = x + self.mlp_path_dropout(self.mlp(out))
        return x

class MedViT(nn.Module):
    def __init__(self, stem_chs, depths, path_dropout, attn_drop=0, drop=0, num_classes=1000,
                 strides=[1, 2, 2, 2], sr_ratios=[8, 4, 2, 1], head_dim=32, mix_block_ratio=0.75,
                 use_checkpoint=False):
        super(MedViT, self).__init__()
        self.use_checkpoint = use_checkpoint

        self.stage_out_channels = [[96] * (depths[0]),
                                   [192] * (depths[1] - 1) + [256],
                                   [384, 384, 384, 384, 512] * (depths[2] // 5),
                                   [768] * (depths[3] - 1) + [1024]]

        # Next Hybrid Strategy
        self.stage_block_types = [[ECB] * depths[0],
                                  [ECB] * (depths[1] - 1) + [LTB],
                                  [ECB, ECB, ECB, ECB, LTB] * (depths[2] // 5),
                                  [ECB] * (depths[3] - 1) + [LTB]]

        self.stem = nn.Sequential(
            ConvBNReLU(3, stem_chs[0], kernel_size=3, stride=2),
            ConvBNReLU(stem_chs[0], stem_chs[1], kernel_size=3, stride=1),
            ConvBNReLU(stem_chs[1], stem_chs[2], kernel_size=3, stride=1),
            ConvBNReLU(stem_chs[2], stem_chs[2], kernel_size=3, stride=2),
        )
        
        input_channel = stem_chs[-1]
        features = []
        idx = 0
        dpr = [x.item() for x in torch.linspace(0, path_dropout, sum(depths))]  # stochastic depth decay rule
        for stage_id in range(len(depths)):
            numrepeat = depths[stage_id]
            output_channels = self.stage_out_channels[stage_id]
            block_types = self.stage_block_types[stage_id]
            for block_id in range(numrepeat):
                if strides[stage_id] == 2 and block_id == 0:
                    stride = 2
                else:
                    stride = 1
                output_channel = output_channels[block_id]
                block_type = block_types[block_id]
                if block_type is ECB:
                    layer = ECB(input_channel, output_channel, stride=stride, path_dropout=dpr[idx + block_id],
                                drop=drop, head_dim=head_dim)
                    features.append(layer)
                elif block_type is LTB:
                    layer = LTB(input_channel, output_channel, path_dropout=dpr[idx + block_id], stride=stride,
                                sr_ratio=sr_ratios[stage_id], head_dim=head_dim, mix_block_ratio=mix_block_ratio,
                                attn_drop=attn_drop, drop=drop)
                    features.append(layer)
                input_channel = output_channel
            idx += numrepeat
        self.features = nn.Sequential(*features)

        self.norm = nn.BatchNorm2d(output_channel, eps=NORM_EPS)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # Remove Classification Layer
        if num_classes is not None:
            self.proj_head = nn.Sequential(nn.Linear(output_channel, num_classes),)
            self.stage_out_idx = [sum(depths[:idx + 1]) - 1 for idx in range(len(depths))]
        else:
            self.proj_head = None
        
        print('initialize_weights...')
        self._initialize_weights()

    def merge_bn(self):
        self.eval()
        for idx, module in self.named_modules():
            if isinstance(module, ECB) or isinstance(module, LTB):
                module.merge_bn()

    def _initialize_weights(self):
        for n, m in self.named_modules():
            if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                trunc_normal_(m.weight, std=.02)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def load_pretrained_weights(self, weight_path):
        try:
            checkpoint = torch.load(weight_path, map_location='cpu')
            print(f"✓ Checkpoint loaded from {weight_path}")
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                elif 'model' in checkpoint:
                    state_dict = checkpoint['model']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
                
            # Remove proj_head if present (since we might set num_classes=None)
            if self.proj_head is None:
                state_dict = {k: v for k, v in state_dict.items() if not k.startswith('proj_head')}
            
            # Load with better error handling
            missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
            
            if missing_keys:
                print(f"Missing keys: {missing_keys}")
            if unexpected_keys:
                print(f"Unexpected keys: {unexpected_keys}")
                
            print("✓ Pretrained weights loaded successfully")
            
        except Exception as e:
            print(f"❌ Error loading weights: {e}")
            raise

    def forward(self, x):
        x = self.stem(x)
        for idx, layer in enumerate(self.features):
            if self.use_checkpoint:
                x = checkpoint.checkpoint(layer, x)
            else:
                x = layer(x)
        x = self.norm(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        if self.proj_head is not None: 
            x = self.proj_head(x)
        return x

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
            # Fixed attention mask handling
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)  # [batch, 1, 1, seq_len]
            # Create a mask for both dimentions of attention matrix
            mask_expanded = attention_mask.expand(-1, self.num_heads, N, -1)
            attn = attn.masked_fill(mask_expanded == 0, float('-inf'))
            
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
        self.vision_token_embed = nn.Parameter(torch.randn(1, 1, text_dim) * 0.02)
        
        # Projection heads for contrastive loss (if enabled)    
        if getattr(config, 'CONS_LOSS', False):
            self.vision_proj = nn.Linear(vision_dim, 512)
            self.text_proj = nn.Linear(text_dim, 512)
            self.temperature = nn.Parameter(torch.tensor(0.07))
             
        # Topic attention module (if enabled)
        if getattr(config, 'TOPIC_ATTENTION', False):
            self.topic_attention = TopicAttention(text_dim, num_heads=8, dropout=0.1)
        
        # Topic attention module (if enabled)
        if config.TOPIC_ATTENTION:
            self.topic_attention = TopicAttention(text_dim, num_heads=8, dropout=0.1)
        
        # Loss tracking for balancing
        self.loss_history = {
            'lm_loss': [],
            'cont_loss': []
        }
        logger.info(f"VisionTextModel initialized with vision_dim={vision_dim}, text_dim={text_dim}. ")

    def get_loss_weights(self):
        """
        Dynamically adjust loss weights based on loss history.
        This helps balance the different loss components.
        """
        if len(self.loss_history['lm_loss']) < 10:
            # Use default weights for first few iterations
            return 1.0, getattr(self.config, 'CONT_LOSS_WEIGHT', 0.1)
        
        # Calculate recent average losses
        recent_lm = sum(self.loss_history['lm_loss'][-10:]) / 10
        recent_cont = sum(self.loss_history['cont_loss'][-10:]) / 10
        
        # Avoid division by zero
        if recent_cont == 0:
            return 1.0, 0.0
        
        # Balance losses to be of similar magnitude
        lm_weight = 1.0
        cont_weight = recent_lm / recent_cont * getattr(self.config, 'CONT_LOSS_WEIGHT', 0.1)
        
        # Clamp weights to reasonable ranges
        cont_weight = max(0.01, min(1.0, cont_weight))
        
        return lm_weight, cont_weight
    
    
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
        """Forward pass with improved error handling and loss balancing."""
        try:
            
            batch_size = pixel_values.size(0)
            
            # Get vision features
            vision_features = self.vision_model(pixel_values)  # [batch, vision_dim]
            
            # Verify vision output shape
            if vision_features.dim() != 2 or vision_features.size(1) != self.vision_dim:
                raise ValueError(f"Vision model output shape mismatch: expected [batch, {self.vision_dim}], "
                               f"got {vision_features.shape}")

            # Get text embeddings
            text_embed = self.text_model.get_input_embeddings()(input_ids)  # [batch, seq_len, text_dim]
            
            # Apply topic attention if enabled
            if hasattr(self, 'topic_attention'):
                text_embed = self.topic_attention(text_embed, attention_mask=attention_mask)

            # Project vision features to text dimension
            vision_projected = self.vision_projection(vision_features)  # [batch, text_dim]
            vision_embed = vision_projected.unsqueeze(1)  # [batch, 1, text_dim]
            
            # Add learnable vision token embedding
            vision_embed = vision_embed + self.vision_token_embed.expand(batch_size, -1, -1)
            
            # Combine vision and text embeddings
            combined_embed = torch.cat([vision_embed, text_embed], dim=1)  # [batch, 1+seq_len, text_dim]
            
            # Create combined attention mask
            vision_mask = torch.ones(batch_size, 1, device=attention_mask.device, dtype=attention_mask.dtype)
            combined_mask = torch.cat([vision_mask, attention_mask], dim=1)
            
            # Create labels for language modeling (ignore vision token)
            labels = input_ids.clone()
            labels[labels == self.text_model.config.pad_token_id] = -100
            batch_size = labels.shape[0]
            vision_label = torch.full((batch_size, 1), -100, dtype=labels.dtype, device=labels.device)
            adjusted_labels = torch.cat([vision_label, labels], dim=1)
            
            # Forward through text model
            outputs = self.text_model(
                inputs_embeds=combined_embed,
                attention_mask=combined_mask,
                labels=adjusted_labels
            )
            
            lm_loss = outputs.loss
            
            # Contrastive loss if enabled
            cont_loss = torch.tensor(0.0, device=lm_loss.device)
            if getattr(self.config, 'CONS_LOSS', False):
                # Use mean pooling of text embeddings for contrastive learning
                text_lengths = attention_mask.sum(dim=1, keepdim=True).float()
                text_features = (text_embed * attention_mask.unsqueeze(-1)).sum(dim=1) / text_lengths
                cont_loss = self.contrastive_loss(vision_features, text_features)

            # Update loss history for balancing
            if self.training:
                self.loss_history['lm_loss'].append(lm_loss.item())
                self.loss_history['cont_loss'].append(cont_loss.item())
                
                # Keep only recent history
                if len(self.loss_history['lm_loss']) > 100:
                    self.loss_history['lm_loss'] = self.loss_history['lm_loss'][-50:]
                    self.loss_history['cont_loss'] = self.loss_history['cont_loss'][-50:]

            # Get balanced weights
            lm_weight, cont_weight = self.get_loss_weights()
            
            # Compute total loss
            total_loss = lm_weight * lm_loss
            if getattr(self.config, 'CONS_LOSS', False):
                total_loss = total_loss + cont_weight * cont_loss

            return {
                'lm_loss': lm_loss,
                'cont_loss': cont_loss,
                'total_loss': total_loss,
                'lm_weight': lm_weight,
                'cont_weight': cont_weight
            }
            

        except Exception as e:
            logger.error(f"Error in VisionTextModel.forward: {str(e)}")
            logger.error(f"Input shapes - pixel_values: {pixel_values.shape}, "
                        f"input_ids: {input_ids.shape}, attention_mask: {attention_mask.shape}")
            raise
            
    def generate_caption(self, pixel_values, tokenizer, max_length=128, num_beams=1, early_stopping=False):
        try:
            # Set models to eval mode
            self.vision_model.eval()
            self.text_model.eval()
        
            with torch.no_grad():
                vision_features = self.vision_model(pixel_values)
                batch_size = pixel_values.size(0)
            
                # Project vision features
                vision_projected = self.vision_projection(vision_features)
                vision_embed = vision_projected.unsqueeze(1)
            
            # Add learnable vision token embedding
            vision_embed = vision_embed + self.vision_token_embed.expand(batch_size, -1, -1)
            
            # Create attention mask for vision tokens
            attention_mask = torch.ones(batch_size, 1, device=vision_embed.device)

            # Generate with improved parameters
            generated_ids = self.text_model.generate(
                inputs_embeds=vision_embed,
                attention_mask=attention_mask,
                max_length=max_length,
                min_length=8,  # Ensure minimum length
                num_beams=max(num_beams, 1),
                early_stopping=early_stopping,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                bos_token_id=tokenizer.bos_token_id,
                do_sample=False,  # Deterministic for testing
                repetition_penalty=1.3,  # Prevent repetition
                # length_penalty=1.1,  # Encourage longer sequences
                no_repeat_ngram_size=3,  # Prevent repetition
                bad_words_ids=[[tokenizer.unk_token_id]] if hasattr(tokenizer, 'unk_token_id') else None
            )
            
            return generated_ids
            
        except Exception as e:
            logger.error(f"Error in generate_caption: {str(e)}")
            raise

    def print_loss_info(self):
        """Print current loss balancing information."""
        if len(self.loss_history['lm_loss']) > 0:
            lm_weight, cont_weight = self.get_loss_weights()
            recent_lm = sum(self.loss_history['lm_loss'][-10:]) / min(10, len(self.loss_history['lm_loss']))
            recent_cont = sum(self.loss_history['cont_loss'][-10:]) / min(10, len(self.loss_history['cont_loss']))
            
            print(f"Loss Balancing Info:")
            print(f"  Recent LM Loss: {recent_lm:.4f} (weight: {lm_weight:.4f})")
            print(f"  Recent Cont Loss: {recent_cont:.4f} (weight: {cont_weight:.4f})")
            print(f"  Total iterations: {len(self.loss_history['lm_loss'])}")
