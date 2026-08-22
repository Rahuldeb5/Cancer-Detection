import torch
import torch.nn as nn
import torch.nn.functional as F
from .conv_layers import BasicBlock, Bottleneck, SingleConv
from .trans_layers import LayerNorm

def get_block(name):
    block_map = {
        'SingleConv': SingleConv,
        'BasicBlock': BasicBlock,
        'Bottleneck': Bottleneck,
    }
    return block_map[name]

def get_norm(name):
    norm_map = {'bn': nn.BatchNorm3d,
                'in': nn.InstanceNorm3d,
                'ln': LayerNorm
                }

    return norm_map[name]

def get_act(name):
    act_map = {
        'relu': nn.ReLU,
        'lrelu': nn.LeakyReLU,
        'gelu': nn.GELU,
        'swish': nn.SiLU
    }
    return act_map[name]

def get_model(args, pretrain=False):
    from .medformer import MedFormer  # deferred: medformer imports get_block/get_norm/get_act from this module

    if args.model != 'medformer':
        raise ValueError(f"Unknown model: {args.model}")

    net = MedFormer(
        in_chan=getattr(args, 'in_chan', 1),
        num_classes=args.classes,
        base_chan=getattr(args, 'base_chan', 32),
        map_size=getattr(args, 'map_size', [4, 8, 8]),
        conv_block=getattr(args, 'conv_block', 'BasicBlock'),
        conv_num=getattr(args, 'conv_num', [2, 1, 0, 0, 0, 1, 2, 2]),
        trans_num=getattr(args, 'trans_num', [0, 1, 2, 2, 2, 1, 0, 0]),
        chan_num=getattr(args, 'chan_num', [64, 128, 256, 320, 256, 128, 64, 32]),
        num_heads=getattr(args, 'num_heads', [1, 4, 8, 16, 8, 4, 1, 1]),
        fusion_depth=getattr(args, 'fusion_depth', 2),
        fusion_dim=getattr(args, 'fusion_dim', 320),
        fusion_heads=getattr(args, 'fusion_heads', 4),
        expansion=getattr(args, 'expansion', 4),
        attn_drop=getattr(args, 'attn_drop', 0.),
        proj_drop=getattr(args, 'proj_drop', 0.),
        proj_type=getattr(args, 'proj_type', 'depthwise'),
        norm=getattr(args, 'norm', 'in'),
        act=getattr(args, 'act', 'gelu'),
        # MedFormer's own default kernel_size=[3,3,3,3] is broken two ways: BasicBlock/SingleConv
        # expect each per-stage entry to be a 3-element per-axis list like [3,3,3] (not a bare int),
        # and __init__ indexes up to kernel_size[4] (inc + 4 down-stages), so 4 entries isn't enough
        kernel_size=getattr(args, 'kernel_size', [[3, 3, 3]] * 5),
        # same issue as kernel_size above -- down_scale needs a per-axis list, not a bare int
        scale=getattr(args, 'scale', [[2, 2, 2]] * 4),
        aux_loss=getattr(args, 'aux_loss', False),
    )

    if pretrain:
        state_dict = torch.load(args.load, map_location='cpu', weights_only=True)
        net.load_state_dict(state_dict['model_state_dict'])

    return net