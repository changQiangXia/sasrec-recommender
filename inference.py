#!/usr/bin/env python3
"""
SASRec 推理脚本
===============
为单个用户生成推荐

用法:
    python inference.py --user_id 123 --top_k 10
"""

import argparse
import torch
import pickle
import sys
sys.path.insert(0, 'src')

from src.model import SASRec
from src.config import get_config


def load_model(checkpoint_path, config):
    """加载训练好的模型"""
    # 需要知道 num_items，从数据缓存中获取
    import os
    cache_files = [f for f in os.listdir('./data/movielens') if f.startswith('.cache_')]
    if not cache_files:
        raise FileNotFoundError("No cache file found")
    
    latest_cache = sorted(cache_files)[-1]
    with open(f'./data/movielens/{latest_cache}', 'rb') as f:
        _, _, _, stats = pickle.load(f)
    
    # 创建模型
    model = SASRec(num_items=stats['num_items'], config=config)
    
    # 加载权重
    checkpoint = torch.load(checkpoint_path, map_location=config.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, stats


def recommend_for_user(model, user_history, all_items, device, top_k=10):
    """
    为用户生成推荐
    
    Args:
        model: SASRec 模型
        user_history: 用户历史物品列表 [item1, item2, ...]
        all_items: 所有候选物品列表
        device: 计算设备
        top_k: 推荐数量
    
    Returns:
        top_k_items: 推荐的 top-k 个物品
        scores: 对应的分数
    """
    config = model.config
    max_seq_len = config.max_seq_len
    
    # 准备输入序列
    seq = user_history[-max_seq_len:]
    if len(seq) < max_seq_len:
        seq = [0] * (max_seq_len - len(seq)) + seq
    
    seq_tensor = torch.tensor([seq], dtype=torch.long).to(device)
    
    # 准备候选物品
    candidate_items = torch.tensor([all_items], dtype=torch.long).to(device)
    
    with torch.no_grad():
        scores = model.predict(seq_tensor, candidate_items)
        scores = scores.squeeze(0)  # (num_candidates,)
    
    # 获取 top-k
    top_scores, top_indices = torch.topk(scores, k=min(top_k, len(all_items)))
    
    # 转换回原始物品 ID
    top_k_items = [all_items[idx] for idx in top_indices.cpu().tolist()]
    top_scores = top_scores.cpu().tolist()
    
    return top_k_items, top_scores


def main():
    parser = argparse.ArgumentParser(description='SASRec Inference')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/best.pt',
                       help='模型检查点路径')
    parser.add_argument('--user_history', type=str, required=True,
                       help='用户历史，逗号分隔的物品ID，如 "1,2,3,4,5"')
    parser.add_argument('--top_k', type=int, default=10,
                       help='推荐数量')
    parser.add_argument('--exclude_history', action='store_true',
                       help='排除用户历史物品')
    
    args = parser.parse_args()
    
    # 配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = get_config('movielens')
    config.device = device
    config.max_seq_len = 200
    
    print(f"Loading model from {args.checkpoint}...")
    model, stats = load_model(args.checkpoint, config)
    model = model.to(device)
    
    # 解析用户历史
    user_history = [int(x) for x in args.user_history.split(',')]
    print(f"\nUser history: {user_history}")
    
    # 准备候选池
    if args.exclude_history:
        # 排除用户已交互的物品
        candidate_pool = [i for i in range(1, stats['num_items'] + 1) 
                         if i not in user_history]
    else:
        candidate_pool = list(range(1, stats['num_items'] + 1))
    
    print(f"Generating top-{args.top_k} recommendations from {len(candidate_pool)} candidates...")
    
    # 生成推荐
    top_items, scores = recommend_for_user(
        model, user_history, candidate_pool, device, args.top_k
    )
    
    print(f"\n{'='*50}")
    print(f"Top-{args.top_k} Recommendations:")
    print(f"{'='*50}")
    for rank, (item_id, score) in enumerate(zip(top_items, scores), 1):
        print(f"  {rank}. Item {item_id:6d}  Score: {score:.4f}")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
