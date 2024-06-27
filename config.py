from functools import partial
import torch.nn as nn

# config = {
#     'img_size': 32,
#     'patch_size': 4,
#     'in_c': 3,
#     'num_classes': 10,
#     'embed_dim': 384,
#     'depth': 6,
#     'num_heads': 8,
#     'mlp_ratio': 3.0,
#     'qkv_bias': True,
#     'qk_scale': None,
#     'representation_size': 64,
#     'drop_ratio': 0.2,
#     'attn_drop_ratio': 0.1,
#     'drop_path_ratio': 0.2,
#     'embed_layer': None,
#     'norm_layer': partial(nn.LayerNorm, eps=1e-6),
#     'act_layer': nn.GELU,
#     'batch_size': 32,
#     'epochs': 30,
#     'lr': 0.0001,
# }

config = {
'img_size': 48,                  # 输入图像的尺寸
'patch_size': 4,                 # 图像分块的大小
'in_c': 3,                       # 输入通道数 (RGB 图像)
'num_classes': 10,               # 类别数量 (CIFAR-10 是 10 类)
'embed_dim':768 ,                 # 每个图像块的嵌入维度
'depth': 12,                      # Transformer 中编码器层数
'num_heads': 12,                  # 多头自注意力机制的头数
'mlp_ratio': 4.0,                # MLP 层的宽度和嵌入维度的比值
'qkv_bias': True,                # 是否使用偏置
'qk_scale': None,                # Query 和 Key 的缩放因子
'representation_size': 128,       # 输出表示的尺寸
'drop_ratio': 0.1,               # Dropout 比例
'attn_drop_ratio': 0.,          # 自注意力机制的 Dropout 比例
'drop_path_ratio': 0.1,          # DropPath（随机深度）的丢弃比例
'embed_layer': None,             # 嵌入层，使用默认
'norm_layer': partial(nn.LayerNorm, eps=1e-6),  # 归一化层
'act_layer': nn.GELU,            # 激活函数
'batch_size': 32,                # 批次大小
'epochs': 30,                    # 训练周期数
'lr': 0.0001,                    # 学习率
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
