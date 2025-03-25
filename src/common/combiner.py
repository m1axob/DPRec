import torch
from torch import nn
import torch.nn.functional as F


class CombinedFusion(nn.Module):
    def __init__(self, clip_feature_dim: int, projection_dim: int, hidden_dim: int):
        super(CombinedFusion, self).__init__()
        # 投影层
        self.text_projection_layer = nn.Linear(clip_feature_dim, projection_dim)
        self.image_projection_layer = nn.Linear(clip_feature_dim, projection_dim)

        # 门控机制
        self.gate_image_prefer = nn.Sequential(nn.Linear(projection_dim, projection_dim), nn.Sigmoid())
        self.gate_text_prefer = nn.Sequential(nn.Linear(projection_dim, projection_dim), nn.Sigmoid())

        # 注意力机制
        self.query_common = nn.Sequential(
            nn.Linear(projection_dim, projection_dim), nn.Tanh(), nn.Linear(projection_dim, 1, bias=False)
        )

        # 动态加权
        self.dynamic_scalar = nn.Sequential(
            nn.Linear(projection_dim * 2, hidden_dim), nn.ReLU(), nn.Dropout(0.5), nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        # 融合层
        self.combiner_layer = nn.Linear(projection_dim * 3, hidden_dim)  # 改为三倍投影维度，用于拼接公共与独立特征
        self.output_layer = nn.Linear(hidden_dim, clip_feature_dim)

        # Dropout
        self.dropout = nn.Dropout(0.5)

    def forward(self, image_features: torch.tensor, text_features: torch.tensor) -> torch.tensor:
        # 特征投影
        text_projected = self.dropout(F.relu(self.text_projection_layer(text_features)))
        image_projected = self.dropout(F.relu(self.image_projection_layer(image_features)))

        # 门控机制调整偏好
        image_prefer = self.gate_image_prefer(image_projected)
        text_prefer = self.gate_text_prefer(text_projected)
        image_specific = torch.multiply(image_prefer, image_projected)
        text_specific = torch.multiply(text_prefer, text_projected)

        # 注意力机制提取公共特征
        att_common = torch.cat([self.query_common(image_projected), self.query_common(text_projected)], dim=-1)
        weight_common = F.softmax(att_common, dim=-1)
        common_embeds = weight_common[:, 0].unsqueeze(1) * image_projected + weight_common[:, 1].unsqueeze(
            1) * text_projected

        # 融合独立特征和公共特征
        combined_features = torch.cat((image_specific, text_specific, common_embeds), dim=-1)  # 公共特征与独立特征拼接
        combined_hidden = self.dropout(F.relu(self.combiner_layer(combined_features)))

        # 动态加权融合
        dynamic_scalar = self.dynamic_scalar(torch.cat((image_projected, text_projected), dim=-1))  # 计算动态标量
        output = self.output_layer(combined_hidden) + dynamic_scalar * text_features + (
                    1 - dynamic_scalar) * image_features

        # 归一化
        return F.normalize(output, dim=-1)

