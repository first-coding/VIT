from functools import partial
import torch.nn as nn

config = {
    'img_size': 32,
    'patch_size': 4,
    'in_c': 3,
    'num_classes': 10,
    'embed_dim': 768,
    'depth': 12,
    'num_heads': 12,
    'mlp_ratio': 4.0,
    'qkv_bias': True,
    'qk_scale': None,
    'representation_size': None,
    'drop_ratio': 0.0,
    'attn_drop_ratio': 0.0,
    'drop_path_ratio': 0.0,
    'embed_layer': None,
    'norm_layer': partial(nn.LayerNorm, eps=1e-6),
    'act_layer': nn.GELU,
    'batch_size': 32,
    'epochs': 10,
    'lr': 0.001,
}

def init_vit_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)
