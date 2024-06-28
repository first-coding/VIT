from functools import partial
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.optim as optim
config = {
    'img_size': 32,                  # 输入图像的尺寸
    'patch_size': 8,                 # 图像分块的大小
    'in_c': 3,                       # 输入通道数 (RGB 图像)
    'num_classes': 10,               # 类别数量 (CIFAR-10 是 10 类)
    'embed_dim':384,                 # 每个图像块的嵌入维度
    'depth': 16,                      # Transformer 中编码器层数
    'num_heads': 8,                  # 多头自注意力机制的头数
    'mlp_ratio': 4.0,                # MLP 层的宽度和嵌入维度的比值
    'qkv_bias': True,                # 是否使用偏置
    'qk_scale': None,                # Query 和 Key 的缩放因子
    'representation_size': 128,       # 输出表示的尺寸
    'drop_ratio': 0.2,               # Dropout 比例
    'attn_drop_ratio': 0.1,          # 自注意力机制的 Dropout 比例
    'drop_path_ratio': 0.2,          # DropPath（随机深度）的丢弃比例
    'embed_layer': None,             # 嵌入层，使用默认
    'Attention_proj_drop_ratio':0.1,
    'MLP_drop':0.1,
    'MLP_hidden_features':None,      #隐藏层特征维度，在models embded_dim * mlp_ratio
    'MLP_out_features':None,         #输出层特征维度，None即自适应
    'norm_layer': partial(nn.LayerNorm, eps=1e-6),  # 归一化层
    'act_layer': nn.GELU,            # 激活函数
    'batch_size': 16,                # 批次大小
    'epochs': 15,                    # 训练周期数
    'optimizer': optim.AdamW,  # 使用 AdamW 优化器
    'scheduler': CosineAnnealingLR,  # 使用余弦退火学习率调度器
    'lr':0.0001 ,                    # 学习率
    'weight_decay': 0.00001,            # 权重衰减，用于 L2 正则化
    'label_smoothing': 0.1,          # 标签平滑系数
    'use_fp16': True,                # 启用混合精度训练
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
