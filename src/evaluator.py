"""
Evaluator
=========
评估指标实现：HR@K 和 NDCG@K - 修复版

这个文件现在主要包含指标计算函数，实际的评估逻辑在 trainer.py 中实现。
"""

import numpy as np
from typing import Dict, List


def compute_hr_at_k(ranks: np.ndarray, k: int) -> float:
    """
    计算 HR@K (Hit Ratio)
    
    HR@K = 正样本排名 <= K 的比例
    
    Args:
        ranks: 正样本的排名数组 (从 1 开始)
        k: 截断位置
    
    Returns:
        HR@K 值
    """
    return float(np.mean(ranks <= k))


def compute_ndcg_at_k(ranks: np.ndarray, k: int) -> float:
    """
    计算 NDCG@K (Normalized Discounted Cumulative Gain)
    
    DCG@K = sum( relevance_i / log2(rank_i + 1) ) for rank_i <= K
    
    对于二元相关性（正样本=1，负样本=0）：
    - 如果正样本排名为 r，DCG = 1 / log2(r + 1)
    - 理想情况下正样本排第1，IDCG = 1 / log2(2) = 1
    - NDCG = DCG / IDCG = DCG
    
    Args:
        ranks: 正样本的排名数组 (从 1 开始)
        k: 截断位置
    
    Returns:
        NDCG@K 值
    """
    # 只考虑排名 <= K 的样本
    dcg = np.where(ranks <= k, 1.0 / np.log2(ranks + 1), 0.0)
    return float(np.mean(dcg))


def compute_mrr(ranks: np.ndarray) -> float:
    """
    计算 MRR (Mean Reciprocal Rank)
    
    MRR = mean(1 / rank)
    
    Args:
        ranks: 排名数组 (从 1 开始)
    
    Returns:
        MRR 值
    """
    return float(np.mean(1.0 / ranks))


class Evaluator:
    """
    推荐系统评估器
    
    注意：实际的评估逻辑现在在 trainer.py 的 evaluate 方法中实现。
    此类保留用于向后兼容和指标计算。
    """
    
    def __init__(self, config):
        self.top_k_list = config.top_k_list
    
    def compute_metrics(self, ranks: np.ndarray) -> Dict[str, float]:
        """
        计算所有评估指标
        
        Args:
            ranks: (num_samples,) 正样本排名数组 (从 1 开始)
        
        Returns:
            metrics: 包含 HR@K, NDCG@K, MRR 的字典
        """
        metrics = {}
        
        for k in self.top_k_list:
            metrics[f'HR@{k}'] = compute_hr_at_k(ranks, k)
            metrics[f'NDCG@{k}'] = compute_ndcg_at_k(ranks, k)
        
        metrics['MRR'] = compute_mrr(ranks)
        metrics['mean_rank'] = float(np.mean(ranks))
        metrics['median_rank'] = float(np.median(ranks))
        
        return metrics


def evaluate_model(model, data_loader, device, top_k_list=[5, 10, 20]):
    """
    简化的评估函数（用于快速测试）
    
    Args:
        model: SASRec 模型
        data_loader: 评估数据加载器
        device: 计算设备
        top_k_list: 评估的 K 值列表
    
    Returns:
        metrics: 评估指标字典
    """
    import torch
    
    model.eval()
    all_ranks = []
    
    with torch.no_grad():
        for batch in data_loader:
            if len(batch) == 3:
                seqs, targets, neg_pools = batch
            else:
                continue
            
            seqs = seqs.to(device)
            targets = targets.to(device)
            neg_pools = neg_pools.to(device)
            
            # 过滤无效样本
            valid_mask = targets > 0
            if valid_mask.sum() == 0:
                continue
            
            valid_seqs = seqs[valid_mask]
            valid_targets = targets[valid_mask]
            valid_neg_pools = neg_pools[valid_mask]
            
            # 构建候选集
            candidates = torch.cat([
                valid_targets.unsqueeze(1),
                valid_neg_pools
            ], dim=1)
            
            # 预测
            logits = model.predict(valid_seqs, candidates)
            rankings = torch.argsort(logits, descending=True, dim=1)
            
            # 计算排名
            for i in range(rankings.size(0)):
                rank = (rankings[i] == 0).nonzero(as_tuple=True)[0].item() + 1
                all_ranks.append(rank)
    
    if len(all_ranks) == 0:
        return {f'HR@{k}': 0.0 for k in top_k_list}
    
    ranks_array = np.array(all_ranks)
    
    metrics = {}
    for k in top_k_list:
        metrics[f'HR@{k}'] = compute_hr_at_k(ranks_array, k)
        metrics[f'NDCG@{k}'] = compute_ndcg_at_k(ranks_array, k)
    
    metrics['MRR'] = compute_mrr(ranks_array)
    
    return metrics
