import torch
import torch.nn as nn
import math

class AttentionFusion(nn.Module):
    def __init__(self, embed_dim):
        super(AttentionFusion, self).__init__()
        self.embed_dim = embed_dim
        # 可以根据需要添加更多的线性层或其他组件
        self.query_projection = nn.Linear(embed_dim, embed_dim)
        self.key_projection = nn.Linear(embed_dim, embed_dim)
        self.value_projection = nn.Linear(embed_dim, embed_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, image_embeds, text_embeds):
        # 将文本和图像嵌入转换成查询、键、值的形式
        query = self.query_projection(text_embeds)
        key = self.key_projection(image_embeds)
        value = self.value_projection(image_embeds)

        # 计算注意力权重
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.embed_dim)
        attention_weights = self.softmax(attention_scores)

        # 应用注意力权重并融合信息
        attended_image = torch.matmul(attention_weights, value)
        fused_embeds = attended_image + text_embeds

        return fused_embeds