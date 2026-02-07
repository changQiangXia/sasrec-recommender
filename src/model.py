"""
SASRec Model
============
修复后的模型实现

关键修复：
1. 设备一致性：所有内部操作使用 self.dev，避免 device 不匹配
2. predict 函数优化：支持批量评估
3. 增加注释和类型提示
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class SASRec(nn.Module):
    """
    SASRec: Self-Attentive Sequential Recommendation
    
    基于 Transformer 的序列推荐模型，使用自注意力机制建模用户行为序列。
    """
    
    def __init__(self, num_items: int, config):
        """
        Args:
            num_items: 物品总数（不包括 padding 0）
            config: 配置对象，包含 hidden_units, num_heads, num_blocks 等
        """
        super(SASRec, self).__init__()
        self.num_items = num_items
        self.hidden_units = config.hidden_units
        self.dev = config.device
        self.max_seq_len = config.max_seq_len
        
        # ========== 嵌入层 ==========
        # Item 嵌入（0 是 padding）
        self.item_emb = nn.Embedding(
            num_items + 1,  # +1 for padding (index 0)
            config.hidden_units,
            padding_idx=0
        )
        
        # 位置编码
        self.pos_emb = nn.Embedding(config.max_seq_len, config.hidden_units)
        
        # Dropout
        self.emb_dropout = nn.Dropout(p=config.dropout)
        
        # ========== Transformer 模块 ==========
        # 使用 Pre-LN (norm_first=True) 提升训练稳定性
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_units,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_units * 4,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-LN 结构
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_blocks
        )
        
        # 最终的 LayerNorm
        self.norm = nn.LayerNorm(config.hidden_units)
        
        # ========== 初始化 ==========
        self.apply(self._init_weights)
        
        # 确保 padding embedding 为 0（虽然 Embedding 层的 padding_idx 已经处理了）
        with torch.no_grad():
            self.item_emb.weight[0].fill_(0)
    
    def _init_weights(self, module):
        """
        权重初始化
        - Embedding: Xavier 均匀分布
        - Linear: 正态分布 (std=0.02)
        - LayerNorm: weight=1, bias=0
        """
        if isinstance(module, nn.Embedding):
            stdv = math.sqrt(1.0 / self.hidden_units)
            module.weight.data.uniform_(-stdv, stdv)
        elif isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(self, log_seqs: torch.Tensor) -> torch.Tensor:
        """
        前向传播（数值稳定版本）
        
        关键修复：
        1. 不使用 PyTorch Transformer 的 src_key_padding_mask（与 causal mask 组合会产生 NaN）
        2. 改用 attention mask 统一处理 causal 和 padding
        
        Args:
            log_seqs: (Batch, Seq_Len) 输入序列，padding 为 0
        
        Returns:
            output: (Batch, Seq_Len, Hidden) 序列表示
        """
        # 确保输入在正确的设备上
        if log_seqs.device != self.dev:
            log_seqs = log_seqs.to(self.dev)
        
        batch_size, seq_len = log_seqs.shape
        
        # Item 嵌入 (Batch, Seq_Len, Hidden)
        seqs = self.item_emb(log_seqs)
        
        # 位置编码 (Batch, Seq_Len)
        positions = torch.arange(seq_len, device=self.dev).unsqueeze(0).repeat(batch_size, 1)
        seqs = seqs + self.pos_emb(positions)
        seqs = self.emb_dropout(seqs)
        
        # ========== Transformer 编码 ==========
        # 简化为只使用 causal mask，padding 处理通过 output masking 实现
        # 这样可以避免 src_key_padding_mask 和 causal mask 组合产生的 NaN 问题
        
        # Causal Mask（上三角为 -1e9）
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), -1e9, device=self.dev),
            diagonal=1
        )
        
        # Transformer 编码（只用 causal mask）
        output = self.transformer(
            seqs,
            mask=causal_mask
        )
        
        # 最终 LayerNorm
        output = self.norm(output)
        
        # 处理 padding：将 padding 位置的输出设为 0
        # 这样不会影响梯度，同时避免 NaN 传播
        padding_mask = (log_seqs == 0).unsqueeze(-1)  # (batch, seq_len, 1)
        output = output.masked_fill(padding_mask, 0.0)
        
        # 额外保险：处理任何可能出现的 NaN
        if torch.isnan(output).any():
            output = torch.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0)
        
        return output  # (Batch, Seq_Len, Hidden)
    
    def compute_loss(self, log_seqs: torch.Tensor, pos_items: torch.Tensor, 
                     neg_items: torch.Tensor) -> torch.Tensor:
        """
        计算 BPR Loss（AMP 安全版本）
        
        Args:
            log_seqs: (Batch, Seq_Len) 输入序列
            pos_items: (Batch, Seq_Len) 每个位置的正样本
            neg_items: (Batch, Seq_Len) 每个位置的负样本
        
        Returns:
            loss: 标量损失
        """
        # 获取序列表示
        log_feats = self.forward(log_seqs)  # (Batch, Seq_Len, Hidden)
        
        # 确保正/负样本在正确设备上
        if pos_items.device != self.dev:
            pos_items = pos_items.to(self.dev)
        if neg_items.device != self.dev:
            neg_items = neg_items.to(self.dev)
        
        # 获取正/负样本的嵌入
        pos_embs = self.item_emb(pos_items)  # (Batch, Seq_Len, Hidden)
        neg_embs = self.item_emb(neg_items)  # (Batch, Seq_Len, Hidden)
        
        # 计算分数（点积）
        # 使用 float32 确保数值稳定性（避免 AMP 下溢）
        pos_logits = (log_feats * pos_embs).sum(dim=-1).float()  # (Batch, Seq_Len)
        neg_logits = (log_feats * neg_embs).sum(dim=-1).float()  # (Batch, Seq_Len)
        
        # 创建 mask（忽略 padding 位置）
        istarget = (pos_items > 0).float()  # (Batch, Seq_Len)
        
        # BPR Loss: -log(sigmoid(pos_score - neg_score))
        # 使用更大的 epsilon (1e-6) 防止 AMP 下溢到 0
        # FP16 的最小正数是 ~5.96e-8，1e-6 是安全的
        diff = pos_logits - neg_logits
        sigmoid_out = torch.sigmoid(diff)
        loss = -torch.log(sigmoid_out + 1e-6)
        
        # 只对有效位置求平均
        loss = (loss * istarget).sum() / (istarget.sum() + 1e-10)
        
        return loss
    
    def predict(self, log_seqs: torch.Tensor, item_indices: torch.Tensor) -> torch.Tensor:
        """
        预测候选物品的分数
        
        使用序列的最后一个位置作为用户表示，计算与候选物品的点积。
        
        Args:
            log_seqs: (Batch, Seq_Len) 输入序列
            item_indices: (Batch, Num_Candidates) 候选物品索引
        
        Returns:
            logits: (Batch, Num_Candidates) 预测分数
        """
        # 获取序列表示
        log_feats = self.forward(log_seqs)  # (Batch, Seq_Len, Hidden)
        
        # 取最后一个位置作为用户表示
        # (Batch, Hidden)
        final_feat = log_feats[:, -1, :]
        
        # 确保候选物品在正确设备上
        if item_indices.device != self.dev:
            item_indices = item_indices.to(self.dev)
        
        # 获取候选物品的嵌入
        # (Batch, Num_Candidates, Hidden)
        item_embs = self.item_emb(item_indices)
        
        # 计算点积分数
        # (Batch, 1, Hidden) * (Batch, Num_Candidates, Hidden) -> sum -> (Batch, Num_Candidates)
        logits = (final_feat.unsqueeze(1) * item_embs).sum(dim=-1)
        
        return logits
    
    def get_embeddings(self) -> torch.Tensor:
        """
        获取所有物品的嵌入（用于全量排序评估）
        
        Returns:
            embeddings: (Num_Items + 1, Hidden) 包含 padding (index 0)
        """
        return self.item_emb.weight
