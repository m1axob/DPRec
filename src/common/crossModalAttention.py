import torch
import torch.nn as nn
import torch.nn.functional as F
import math
class MultiHeadAttention(nn.Module):
   def __init__(self, dim, num_heads=4):
       super().__init__()
       self.num_heads = num_heads
       self.head_dim = dim // num_heads
       self.query = nn.Linear(dim, dim)
       self.key = nn.Linear(dim, dim)
       self.value = nn.Linear(dim, dim)
       
   def forward(self, x):
       B, N, D = x.shape
       
       # 多头投影
       q = self.query(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
       k = self.key(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
       v = self.value(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
       
       # 注意力计算
       scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
       attn = F.softmax(scores, dim=-1)
       out = torch.matmul(attn, v)
       
       # 拼接多头结果
       out = out.transpose(1, 2).reshape(B, N, D)
       return out
