import math
from functools import partial

import torch
import torch.nn as nn

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1, 1, 0, bias=True),
            nn.GELU(),
            nn.BatchNorm2d(hidden_features, eps=1e-5),
        )
        self.proj = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, groups=hidden_features)
        self.proj_act = nn.GELU()
        self.proj_bn = nn.BatchNorm2d(hidden_features, eps=1e-5)
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_features, out_features, 1, 1, 0, bias=True),
            nn.BatchNorm2d(out_features, eps=1e-5),
        )
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        x = self.conv1(x)
        x = self.drop(x)
        x = self.proj(x) + x
        x = self.proj_act(x)
        x = self.proj_bn(x)
        x = self.conv2(x)
        x = x.flatten(2).permute(0, 2, 1)
        x = self.drop(x)
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)

        H, W = H // self.patch_size[0], W // self.patch_size[1]
        return x, H, W


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, layer=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.layer = layer
        self.proj = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 1, 1, 0, groups=dim)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, _, C = x.shape

        x_LFE = x.permute(0, 2, 1).reshape(B, C, H, W)
        x = self.proj(x_LFE) + x_LFE
        x = self.proj2(x)
        x = x.flatten(2).permute(0, 2, 1)

        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x


from wtconv import WTConv2d


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.origin_dim = dim
        self.sr_ratio = sr_ratio

        self.dim = self.origin_dim // 4 * 3
        self.conv_part_dim = self.origin_dim // 4

        self.wtconv = WTConv2d(self.conv_part_dim, self.conv_part_dim, kernel_size=3, wt_levels=5)

        if self.sr_ratio == 8 or self.sr_ratio == 4:
            self.conv_part_2d = nn.Conv2d(in_channels=self.conv_part_dim,
                                          out_channels=self.conv_part_dim,
                                          kernel_size=3,
                                          padding=1)

        elif sr_ratio == 2 or sr_ratio == 1:
            self.conv_part_2d = nn.Conv2d(in_channels=self.conv_part_dim,
                                          out_channels=self.conv_part_dim,
                                          kernel_size=5,
                                          padding=2)
        self.convGELU = nn.GELU()
        self.convNorm = nn.LayerNorm(self.conv_part_dim)

        self.proj = nn.Linear(self.origin_dim, self.origin_dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.num_heads = num_heads
        head_dim = self.dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(self.dim, self.dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        if sr_ratio > 1:
            self.act = nn.GELU()
            if sr_ratio == 8:
                self.sr1 = nn.Conv2d(self.dim, self.dim, kernel_size=8, stride=8, groups=self.dim)
                self.norm1 = nn.LayerNorm(self.dim)
                self.sr2 = nn.Conv2d(self.dim, self.dim, kernel_size=4, stride=4, groups=self.dim)
                self.norm2 = nn.LayerNorm(self.dim)
                self.kv1 = nn.Linear(self.dim, self.dim, bias=qkv_bias)
                self.kv2 = nn.Linear(self.dim, self.dim, bias=qkv_bias)

            if sr_ratio == 4:
                self.sr1 = nn.Conv2d(self.dim, self.dim, kernel_size=4, stride=4, groups=self.dim)
                self.norm1 = nn.LayerNorm(self.dim)
                self.sr2 = nn.Conv2d(self.dim, self.dim, kernel_size=2, stride=2, groups=self.dim)
                self.norm2 = nn.LayerNorm(self.dim)
                self.sr3 = nn.Conv2d(self.dim, self.dim, kernel_size=1, stride=1, groups=self.dim)
                self.norm3 = nn.LayerNorm(self.dim)
                self.sr4 = nn.Conv2d(self.dim, self.dim // 4, kernel_size=3, stride=1, padding=1)
                self.norm4 = nn.LayerNorm(self.dim // 4)
                self.kv1 = nn.Linear(self.dim, self.dim // 2, bias=qkv_bias)
                self.kv2 = nn.Linear(self.dim, self.dim // 2, bias=qkv_bias)
                self.kv3 = nn.Linear(self.dim, self.dim // 2, bias=qkv_bias)

            if sr_ratio == 2:
                self.sr1 = nn.Conv2d(self.dim, self.dim, kernel_size=2, stride=2, groups=self.dim)
                self.norm1 = nn.LayerNorm(self.dim)
                self.sr2 = nn.Conv2d(self.dim, self.dim, kernel_size=1, stride=1, groups=self.dim)
                self.norm2 = nn.LayerNorm(self.dim)
                self.sr3 = nn.Conv2d(self.dim, self.dim // 4, kernel_size=3, stride=1, padding=1)
                self.norm3 = nn.LayerNorm(self.dim // 4)
                self.sr4 = nn.Conv2d(self.dim, self.dim // 4, kernel_size=3, stride=1, padding=1)
                self.norm4 = nn.LayerNorm(self.dim // 4)
                self.kv1 = nn.Linear(self.dim, self.dim // 2, bias=qkv_bias)
                self.kv2 = nn.Linear(self.dim, self.dim // 2, bias=qkv_bias)
        else:
            self.kv = nn.Linear(self.dim, self.dim * 2, bias=qkv_bias)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H=56, W=56):
        B, N, C = x.shape
        x_conv_part, x_attn_part = torch.split(x, [self.conv_part_dim, self.dim], dim=2)
        x_conv_part = x_conv_part.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x_conv_part = self.wtconv(x_conv_part)
        x_conv_part = self.convGELU(
            self.convNorm(self.conv_part_2d(x_conv_part).reshape(B, self.conv_part_dim, -1).permute(0, 2, 1)))
        _, _, C = x_attn_part.shape
        q = self.q(x_attn_part).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        if self.sr_ratio > 1:
            x_ = x_attn_part.permute(0, 2, 1).reshape(B, C, H, W)
            if self.sr_ratio == 8:
                x_1 = self.act(self.norm1(self.sr1(x_).reshape(B, C, -1).permute(0, 2, 1)))
                x_2 = self.act(self.norm2(self.sr2(x_).reshape(B, C, -1).permute(0, 2, 1)))
                kv1 = self.kv1(x_1).reshape(B, -1, 2, self.num_heads // 2, C // self.num_heads).permute(2, 0, 3, 1, 4)
                kv2 = self.kv2(x_2).reshape(B, -1, 2, self.num_heads // 2, C // self.num_heads).permute(2, 0, 3, 1, 4)
                k1, v1 = kv1[0], kv1[1]
                k2, v2 = kv2[0], kv2[1]
                attn1 = (q[:, :self.num_heads // 2] @ k1.transpose(-2, -1)) * self.scale
                attn1 = attn1.softmax(dim=-1)
                attn1 = self.attn_drop(attn1)
                x1 = (attn1 @ v1).transpose(1, 2).reshape(B, N, C // 2)
                attn2 = (q[:, self.num_heads // 2:] @ k2.transpose(-2, -1)) * self.scale
                attn2 = attn2.softmax(dim=-1)
                attn2 = self.attn_drop(attn2)
                x2 = (attn2 @ v2).transpose(1, 2).reshape(B, N, C // 2)
                x = torch.cat([x1, x2], dim=-1)


            elif self.sr_ratio == 4:
                x_1 = self.act(self.norm1(self.sr1(x_).reshape(B, C, -1).permute(0, 2, 1)))
                x_2 = self.act(self.norm2(self.sr2(x_).reshape(B, C, -1).permute(0, 2, 1)))
                x_3 = self.act(self.norm3(self.sr3(x_).reshape(B, C, -1).permute(0, 2, 1)))
                x_4 = self.act(self.norm4(self.sr4(x_).reshape(B, C // 4, -1).permute(0, 2, 1)))
                kv1 = self.kv1(x_1).reshape(B, -1, 2, self.num_heads // 4, C // self.num_heads).permute(2, 0, 3, 1, 4)
                kv2 = self.kv2(x_2).reshape(B, -1, 2, self.num_heads // 4, C // self.num_heads).permute(2, 0, 3, 1, 4)
                kv3 = self.kv3(x_3).reshape(B, -1, 2, self.num_heads // 4, C // self.num_heads).permute(2, 0, 3, 1, 4)
                k1, v1 = kv1[0], kv1[1]
                k2, v2 = kv2[0], kv2[1]
                k3, v3 = kv3[0], kv3[1]

                attn1 = (q[:, :self.num_heads // 4] @ k1.transpose(-2, -1)) * self.scale
                attn1 = attn1.softmax(dim=-1)
                attn1 = self.attn_drop(attn1)
                x1 = (attn1 @ v1).transpose(1, 2).reshape(B, N, C // 4)

                attn2 = (q[:, self.num_heads // 4: self.num_heads // 2] @ k2.transpose(-2, -1)) * self.scale
                attn2 = attn2.softmax(dim=-1)
                attn2 = self.attn_drop(attn2)
                x2 = (attn2 @ v2).transpose(1, 2).reshape(B, N, C // 4)

                attn3 = (q[:, self.num_heads // 2: self.num_heads // 4 * 3] @ k3.transpose(-2, -1)) * self.scale
                attn3 = attn3.softmax(dim=-1)
                attn3 = self.attn_drop(attn3)
                x3 = (attn3 @ v3).transpose(1, 2).reshape(B, N, C // 4)

                x = torch.cat([x1, x2, x3, x_4], dim=-1)


            elif self.sr_ratio == 2:
                x_1 = self.act(self.norm1(self.sr1(x_).reshape(B, C, -1).permute(0, 2, 1)))
                x_2 = self.act(self.norm2(self.sr2(x_).reshape(B, C, -1).permute(0, 2, 1)))
                x_3 = self.act(self.norm3(self.sr3(x_).reshape(B, C // 4, -1).permute(0, 2, 1)))
                x_4 = self.act(self.norm4(self.sr4(x_).reshape(B, C // 4, -1).permute(0, 2, 1)))

                kv1 = self.kv1(x_1).reshape(B, -1, 2, self.num_heads // 4, C // self.num_heads).permute(2, 0, 3, 1, 4)
                kv2 = self.kv2(x_2).reshape(B, -1, 2, self.num_heads // 4, C // self.num_heads).permute(2, 0, 3, 1, 4)

                k1, v1 = kv1[0], kv1[1]
                k2, v2 = kv2[0], kv2[1]

                attn1 = (q[:, :self.num_heads // 4] @ k1.transpose(-2, -1)) * self.scale
                attn1 = attn1.softmax(dim=-1)
                attn1 = self.attn_drop(attn1)
                x1 = (attn1 @ v1).transpose(1, 2).reshape(B, N, C // 4)

                attn2 = (q[:, self.num_heads // 4: self.num_heads // 2] @ k2.transpose(-2, -1)) * self.scale
                attn2 = attn2.softmax(dim=-1)
                attn2 = self.attn_drop(attn2)
                x2 = (attn2 @ v2).transpose(1, 2).reshape(B, N, C // 4)

                x = torch.cat([x1, x2, x_3, x_4], dim=-1)

            else:
                x_1 = self.act(self.norm1(self.sr1(x_).reshape(B, C, -1).permute(0, 2, 1)))
                x_2 = self.act(self.norm2(self.sr2(x_).reshape(B, C, -1).permute(0, 2, 1)))
                kv1 = self.kv1(x_1).reshape(B, -1, 2, self.num_heads // 2, C // self.num_heads).permute(2, 0, 3, 1, 4)
                kv2 = self.kv2(x_2).reshape(B, -1, 2, self.num_heads // 2, C // self.num_heads).permute(2, 0, 3, 1, 4)
                k1, v1 = kv1[0], kv1[1]
                k2, v2 = kv2[0], kv2[1]

                attn1 = (q[:, :self.num_heads // 2] @ k1.transpose(-2, -1)) * self.scale
                attn1 = attn1.softmax(dim=-1)
                attn1 = self.attn_drop(attn1)
                x1 = (attn1 @ v1).transpose(1, 2).reshape(B, N, C // 2)

                attn2 = (q[:, self.num_heads // 2:] @ k2.transpose(-2, -1)) * self.scale
                attn2 = attn2.softmax(dim=-1)
                attn2 = self.attn_drop(attn2)
                x2 = (attn2 @ v2).transpose(1, 2).reshape(B, N, C // 2)

                x = torch.cat([x1, x2], dim=-1)
        else:
            kv = self.kv(x_attn_part).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            k, v = kv[0], kv[1]

            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        x = torch.cat([x, x_conv_part], dim=-1)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)


class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_tensors[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class REPVTModel(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None,
                 depths=[3, 4, 6, 3], num_stages=4, drop_path_rate=0., drop_rate=0., attn_drop_rate=0.,
                 norm_layer=nn.LayerNorm, sr_ratios=[8, 4, 2, 1], stem_channel=16, fc_dim=1280):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.num_stages = num_stages

        self.dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.cur = 0

        self.conv_stem = nn.Sequential(

            nn.Conv2d(3, stem_channel, kernel_size=3, stride=2, padding=1, bias=True),
            nn.GELU(),
            nn.BatchNorm2d(stem_channel),

            nn.Conv2d(stem_channel, stem_channel, kernel_size=3, stride=1, padding=1, bias=True),
            nn.GELU(),
            nn.BatchNorm2d(stem_channel),

            nn.Conv2d(stem_channel, stem_channel, kernel_size=3, stride=1, padding=1, bias=True),
            nn.GELU(),
            nn.BatchNorm2d(stem_channel),
        )

        self.patch_embed_1 = PatchEmbed(img_size=img_size // 2,
                                        patch_size=2,
                                        in_chans=stem_channel,
                                        embed_dim=embed_dims[0])

        self.block_1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[self.cur + j],
            norm_layer=norm_layer, sr_ratio=sr_ratios[0], layer=1)
            for j in range(depths[0])])

        self.cur += depths[0]

        self.patch_embed_2 = PatchEmbed(img_size=img_size // 4,
                                        patch_size=2,
                                        in_chans=embed_dims[0],
                                        embed_dim=embed_dims[1])

        self.block_2 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[self.cur + j],
            norm_layer=norm_layer, sr_ratio=sr_ratios[1], layer=2)
            for j in range(depths[1])])

        self.cur += depths[1]

        self.patch_embed_3 = PatchEmbed(img_size=img_size // 8,
                                        patch_size=2,
                                        in_chans=embed_dims[1],
                                        embed_dim=embed_dims[2])

        self.block_3 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[self.cur + j],
            norm_layer=norm_layer, sr_ratio=sr_ratios[2], layer=3)
            for j in range(depths[2])])

        self.cur += depths[2]

        self.patch_embed_4 = PatchEmbed(img_size=img_size // 16,
                                        patch_size=2,
                                        in_chans=embed_dims[2],
                                        embed_dim=embed_dims[3])

        self.block_4 = nn.ModuleList([Block(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[self.cur + j],
            norm_layer=norm_layer, sr_ratio=sr_ratios[3], layer=4)
            for j in range(depths[3])])

        self.cur += depths[3]

        self._fc = nn.Conv2d(embed_dims[-1], fc_dim, kernel_size=1)
        self._bn = nn.BatchNorm2d(fc_dim, eps=1e-5)
        self._swish = MemoryEfficientSwish()
        self._avg_pooling = nn.AdaptiveAvgPool2d(1)

        self.head = nn.Linear(fc_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if isinstance(m, nn.Conv2d) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        B = x.shape[0]
        x = self.conv_stem(x)

        x, H, W = self.patch_embed_1(x)

        for blk in self.block_1:
            x = blk(x, H, W)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        x, H, W = self.patch_embed_2(x)

        for blk in self.block_2:
            x = blk(x, H, W)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        x, H, W = self.patch_embed_3(x)

        for blk in self.block_3:
            x = blk(x, H, W)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        x, H, W = self.patch_embed_4(x)

        for blk in self.block_4:
            x = blk(x, H, W)

        B, N, C = x.shape
        x = self._fc(x.permute(0, 2, 1).reshape(B, C, H, W))
        x = self._bn(x)
        x = self._swish(x)
        x = self._avg_pooling(x).flatten(start_dim=1)

        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


@register_model
def mscvit_t(pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs):
    model = REPVTModel(
        patch_size=4,
        embed_dims=[32, 64, 128, 256],
        num_heads=[2, 4, 8, 16],
        mlp_ratios=[8, 8, 4, 4],
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=[1, 2, 4, 1],
        sr_ratios=[8, 4, 2, 1],
        stem_channel=16,
        **kwargs)
    model.default_cfg = _cfg()

    return model


@register_model
def mscvit_xs(pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs):
    model = REPVTModel(
        patch_size=4,
        embed_dims=[48, 96, 192, 384],
        num_heads=[2, 4, 8, 16],
        mlp_ratios=[8, 8, 4, 4],
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=[1, 1, 3, 2],
        sr_ratios=[8, 4, 2, 1],
        stem_channel=24,
        **kwargs)
    model.default_cfg = _cfg()

    return model


@register_model
def mscvit_s(pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs):
    model = REPVTModel(
        patch_size=4,
        embed_dims=[64, 128, 256, 512],
        num_heads=[2, 4, 8, 16],
        mlp_ratios=[8, 8, 4, 4],
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=[2, 2, 4, 2],
        sr_ratios=[8, 4, 2, 1],
        stem_channel=32,
        **kwargs)
    model.default_cfg = _cfg()

    return model
