import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Combiner(nn.Module):
    def __init__(self, clip_feature_dim: int, projection_dim: int, hidden_dim: int):
        super(Combiner, self).__init__()
        # 投影层
        self.text_projection_layer = nn.Linear(clip_feature_dim, projection_dim)
        self.image_projection_layer = nn.Linear(clip_feature_dim, projection_dim)
        
        # 多头注意力机制用于提取公共特征
        self.num_heads = 4
        self.attention = nn.MultiheadAttention(
            embed_dim=projection_dim,
            num_heads=self.num_heads,
            dropout=0.1
        )
        
        # 模态对齐投影
        self.modal_alignment = nn.Sequential(
            nn.Linear(projection_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, projection_dim)
        )
        
        # 动态加权
        self.dynamic_scalar = nn.Sequential(
            nn.Linear(projection_dim * 2, hidden_dim), 
            nn.ReLU(), 
            nn.Dropout(0.1), 
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        # 融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(projection_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, clip_feature_dim)
        )

    def forward(self, image_features: torch.tensor, text_features: torch.tensor) -> torch.tensor:
        # 特征投影
        text_projected = F.relu(self.text_projection_layer(text_features))
        image_projected = F.relu(self.image_projection_layer(image_features))
        
        # 模态对齐
        aligned_text = self.modal_alignment(text_projected)
        aligned_image = self.modal_alignment(image_projected)
        # aligned_text = text_projected
        # aligned_image = image_projected
        
        # 准备注意力输入
        # 将特征转换为注意力机制所需的形状 (seq_len, batch_size, embed_dim)
        img_attn = aligned_image.unsqueeze(0)
        txt_attn = aligned_text.unsqueeze(0)
        
        # 使用多头注意力提取公共特征
        common_features, _ = self.attention(
            query=img_attn,
            key=txt_attn,
            value=txt_attn
        )
        common_features = common_features.squeeze(0)
        
        # 提取模态特定特征
        image_specific = aligned_image - common_features
        text_specific = aligned_text - common_features
        
        # 动态加权融合
        dynamic_weight = self.dynamic_scalar(
            torch.cat([image_specific, text_specific], dim=-1)
        )
        
        # 融合所有特征
        combined_features = torch.cat([
            image_specific,
            text_specific,
            common_features
        ], dim=-1)
        
        output = self.fusion_layer(combined_features)
        
        # 残差连接
        output = output + dynamic_weight * text_features + (1 - dynamic_weight) * image_features
        
        return F.normalize(output, dim=-1)

# import os
# import numpy as np
# import scipy.sparse as sp
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import math



# class Combiner(nn.Module):
#     def __init__(self, clip_feature_dim: int, projection_dim: int, hidden_dim: int):
#         super(Combiner, self).__init__()
#         self.text_projection_layer = nn.Linear(clip_feature_dim, projection_dim)
#         self.image_projection_layer = nn.Linear(clip_feature_dim, projection_dim)

#         # 公共特征提取模块
#         self.common_gate = nn.Sequential(
#             nn.Linear(clip_feature_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, clip_feature_dim),
#             nn.Sigmoid()
#         )

#         # 模态对齐投影
#         self.align_proj = nn.Sequential(
#             nn.Linear(clip_feature_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, clip_feature_dim)
#         )
#         self.dynamic_scalar = nn.Sequential(
#             nn.Linear(projection_dim * 2, hidden_dim), nn.ReLU(), nn.Dropout(0.9), nn.Linear(hidden_dim, 1),
#             nn.Sigmoid()
#         )

#         # 融合层
#         self.combiner_layer = nn.Linear(projection_dim * 2, hidden_dim)
#         self.output_layer = nn.Linear(hidden_dim, clip_feature_dim)

#         # Dropout
#         self.dropout = nn.Dropout(0.9)
#         # ... rest of the existing initialization code ...

#     def forward(self, image_features: torch.tensor, text_features: torch.tensor) -> torch.tensor:
#         # 特征对齐
#         aligned_image = self.align_proj(image_features)
#         aligned_text = self.align_proj(text_features)

#         # 计算门控权重
#         image_gate = self.common_gate(aligned_image)
#         text_gate = self.common_gate(aligned_text)

#         # 提取公共特征
#         common_embeds = (image_gate * aligned_image + text_gate * aligned_text) / 2

#         # 计算特征的独立部分
#         image_specific = image_features - common_embeds
#         text_specific = text_features - common_embeds
#         text_projected = self.dropout(F.relu(self.text_projection_layer(text_specific)))
#         image_projected = self.dropout(F.relu(self.image_projection_layer(image_specific)))
#         # 融合独立特征和公共特征
#         combined_features = torch.cat((text_projected, image_projected), dim=-1)  # 公共特征与独立特征拼接
#         combined_hidden = self.dropout(F.relu(self.combiner_layer(combined_features)))

#         # 动态加权融合
#         dynamic_scalar = self.dynamic_scalar(torch.cat((image_projected, text_projected), dim=-1))  # 计算动态标量
#         combiner = self.output_layer(combined_hidden) + dynamic_scalar * text_specific + (
#                 1 - dynamic_scalar) * image_specific
#         output = (combiner + common_embeds) / 2

#         # 归一化
#         return F.normalize(output, dim=-1)