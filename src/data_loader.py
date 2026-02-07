"""
Data Loader
===========
生成序列形式的正负样本 - 严格修复版

关键修复：
1. 数据划分无泄露：训练/验证/测试集使用不同的输入序列和目标
2. 评估数据固定：每个验证/测试样本使用固定的100个负样本
3. 负采样增强：支持动态负采样和静态评估负样本
"""

import os
import random
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class SequenceTrainDataset(Dataset):
    """
    训练数据集 - 每个位置都有正负样本
    支持动态负采样（每个epoch采样不同的负样本）
    
    优化：使用 numpy 向量化采样，大幅减少 __getitem__ 时间
    """
    
    def __init__(self, user_seqs: Dict[int, List[int]], max_seq_len: int, 
                 num_items: int, neg_sampling_strategy: str = "mixed",
                 num_neg_per_pos: int = 1, popular_items: Optional[List[int]] = None,
                 popular_alpha: float = 0.75):
        """
        Args:
            user_seqs: 用户交互序列 {user_id: [item1, item2, ...]}
            max_seq_len: 最大序列长度
            num_items: 物品总数（用于负采样）
            neg_sampling_strategy: 负采样策略 (random/popular/mixed)
            num_neg_per_pos: 每个正样本对应的负样本数
            popular_items: 热门物品列表
            popular_alpha: 混合采样中热门物品比例
        """
        self.user_seqs = user_seqs
        self.users = list(user_seqs.keys())
        self.max_seq_len = max_seq_len
        self.num_items = num_items
        self.neg_sampling_strategy = neg_sampling_strategy
        self.num_neg_per_pos = num_neg_per_pos
        # 限制热门物品数量以提高效率
        self.popular_items = popular_items[:2000] if popular_items else []
        self.popular_alpha = popular_alpha
        
        # 预计算每个用户的正样本集合（使用 set 提高查询效率）
        self.user_pos_items = {uid: set(items) for uid, items in user_seqs.items()}
        
        # 为每个用户预计算负样本候选池（排除正样本的热门物品）
        # 这样可以避免在 __getitem__ 中进行列表推导
        self.user_neg_candidates = {}
        if neg_sampling_strategy in ["popular", "mixed"]:
            for uid, pos_set in self.user_pos_items.items():
                candidates = [i for i in self.popular_items if i not in pos_set]
                self.user_neg_candidates[uid] = candidates if candidates else None
    
    def __len__(self):
        return len(self.users)
    
    def _sample_negatives_batch(self, user_id: int, n: int) -> List[int]:
        """
        批量采样 n 个负样本（使用 numpy 向量化）
        比循环调用 _sample_negative 快 10-100 倍
        """
        user_pos = self.user_pos_items[user_id]
        negs = []
        
        if self.neg_sampling_strategy == "random":
            # 使用 numpy 批量生成随机数
            max_attempts = n * 10
            attempts = 0
            while len(negs) < n and attempts < max_attempts:
                batch_size = min(n - len(negs), 1000)
                candidates = np.random.randint(1, self.num_items + 1, size=batch_size)
                # 过滤掉正样本
                valid = [c for c in candidates if c not in user_pos]
                negs.extend(valid[:n - len(negs)])
                attempts += batch_size
            
            # 如果不足，随机填充
            while len(negs) < n:
                negs.append(random.randint(1, self.num_items))
        
        elif self.neg_sampling_strategy == "popular":
            # 从预计算的候选池中采样
            candidates = self.user_neg_candidates.get(user_id)
            if candidates:
                # 使用 numpy 随机选择
                indices = np.random.choice(len(candidates), size=min(n, len(candidates)), replace=True)
                negs = [candidates[i] for i in indices]
            else:
                # 兜底：随机采样
                negs = np.random.randint(1, self.num_items + 1, size=n).tolist()
        
        else:  # mixed - 混合采样
            num_from_popular = int(n * self.popular_alpha)
            num_random = n - num_from_popular
            
            # 从热门采样
            candidates = self.user_neg_candidates.get(user_id)
            if candidates and num_from_popular > 0:
                indices = np.random.choice(len(candidates), size=min(num_from_popular, len(candidates)), replace=True)
                negs.extend([candidates[i] for i in indices])
            
            # 随机采样剩余部分
            if num_random > 0:
                max_attempts = num_random * 10
                attempts = 0
                while len(negs) < n and attempts < max_attempts:
                    batch_size = min(n - len(negs), 1000)
                    candidates = np.random.randint(1, self.num_items + 1, size=batch_size)
                    valid = [c for c in candidates if c not in user_pos]
                    negs.extend(valid[:n - len(negs)])
                    attempts += batch_size
                
                # 填充
                while len(negs) < n:
                    negs.append(random.randint(1, self.num_items))
        
        return negs[:n]
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        返回一个用户的训练样本
        
        Returns:
            seq: (max_seq_len,) - 输入序列
            pos_items: (max_seq_len,) - 每个位置的下一个物品（正样本）
            neg_items: (max_seq_len,) - 每个位置的负样本
        """
        user_id = self.users[idx]
        items = self.user_seqs[user_id]
        
        # 序列太短无法训练
        if len(items) < 2:
            return (
                torch.zeros(self.max_seq_len, dtype=torch.long),
                torch.zeros(self.max_seq_len, dtype=torch.long),
                torch.zeros(self.max_seq_len, dtype=torch.long)
            )
        
        # 取最近的 max_seq_len+1 个物品
        items = items[-(self.max_seq_len + 1):]
        
        # 输入序列和正样本序列
        seq = items[:-1]
        pos = items[1:]
        
        # 填充到 max_seq_len
        seq_len = len(seq)
        pad_len = self.max_seq_len - seq_len
        if pad_len > 0:
            seq = [0] * pad_len + seq
            pos = [0] * pad_len + pos
        
        # 计算需要采样的负样本数量
        num_valid_pos = seq_len  # 非 padding 的正样本数量
        
        if num_valid_pos > 0:
            # 批量采样负样本（向量化，比循环快得多）
            negs = self._sample_negatives_batch(user_id, num_valid_pos)
            # 填充 padding 位置
            neg = [0] * pad_len + negs
        else:
            neg = [0] * self.max_seq_len
        
        return (
            torch.tensor(seq, dtype=torch.long),
            torch.tensor(pos, dtype=torch.long),
            torch.tensor(neg, dtype=torch.long)
        )


class SequenceEvalDataset(Dataset):
    """
    评估数据集 - 使用固定的负样本进行公平评估
    
    关键设计：
    1. 每个用户的测试样本包含：输入序列 + 目标正样本 + 固定负样本池
    2. 使用用户特定的随机种子保证负样本固定且可复现
    3. 延迟采样策略：负样本在 __getitem__ 时按需生成（避免初始化过慢）
    """
    
    def __init__(self, user_seqs: Dict[int, List[int]], max_seq_len: int,
                 num_items: int, num_neg_samples: int = 100, 
                 popular_items: Optional[List[int]] = None,
                 seed: int = 42):
        """
        Args:
            user_seqs: 用户交互序列
            max_seq_len: 最大序列长度
            num_items: 物品总数
            num_neg_samples: 每个正样本对应的负样本数
            popular_items: 热门物品列表（用于更有挑战性的负采样）
            seed: 全局随机种子基础
        """
        self.user_seqs = user_seqs
        self.users = list(user_seqs.keys())
        self.max_seq_len = max_seq_len
        self.num_items = num_items
        self.num_neg_samples = num_neg_samples
        
        # 限制热门物品数量以提高效率
        self.popular_items = popular_items[:2000] if popular_items else []
        self.base_seed = seed
        
        # 预计算每个用户的正样本集合
        self.user_pos_items = {uid: set(items) for uid, items in user_seqs.items()}
        
        # 注意：负样本不再在 __init__ 中预采样，而是延迟到 __getitem__
        # 这样每个用户的负样本仍然固定（通过 uid 派生种子）
    
    def __len__(self):
        return len(self.users)
    
    def _get_neg_pool(self, user_id: int) -> List[int]:
        """
        为指定用户生成固定的负样本池
        使用用户ID派生种子，保证同一用户总是得到相同的负样本
        """
        pos_set = self.user_pos_items[user_id]
        
        # 使用用户ID派生种子，保证可复现性
        user_seed = (self.base_seed + user_id) % (2**31)
        rng = random.Random(user_seed)
        
        # 采样策略：混合热门和随机负样本
        neg_pool = []
        
        # 50% 从热门物品中采样（但排除正样本）
        popular_candidates = [i for i in self.popular_items if i not in pos_set]
        num_from_popular = min(self.num_neg_samples // 2, len(popular_candidates))
        if num_from_popular > 0:
            neg_pool.extend(rng.sample(popular_candidates, num_from_popular))
        
        # 剩余随机采样
        remaining = self.num_neg_samples - len(neg_pool)
        random_negs = []
        attempts = 0
        max_attempts = remaining * 20  # 增加尝试次数上限
        
        while len(random_negs) < remaining and attempts < max_attempts:
            neg = rng.randint(1, self.num_items)
            if neg not in pos_set and neg not in random_negs:
                random_negs.append(neg)
            attempts += 1
            
        neg_pool.extend(random_negs)
        
        # 如果采样不足，用随机物品填充
        while len(neg_pool) < self.num_neg_samples:
            neg_pool.append(rng.randint(1, self.num_items))
            
        return neg_pool[:self.num_neg_samples]
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        返回一个用户的评估样本
        
        Returns:
            seq: (max_seq_len,) - 输入序列
            target: (1,) - 目标正样本
            neg_pool: (num_neg_samples,) - 固定的负样本池
        """
        user_id = self.users[idx]
        items = self.user_seqs[user_id]
        
        # 序列太短无法评估
        if len(items) < 1:
            return (
                torch.zeros(self.max_seq_len, dtype=torch.long),
                torch.tensor(0, dtype=torch.long),
                torch.zeros(self.num_neg_samples, dtype=torch.long)
            )
        
        # 目标正样本是序列的最后一个物品
        target = items[-1]
        
        # 输入序列是除最后一个外的所有物品
        seq = items[:-1]
        
        # 填充到 max_seq_len
        if len(seq) < self.max_seq_len:
            pad_len = self.max_seq_len - len(seq)
            seq = [0] * pad_len + seq
        elif len(seq) > self.max_seq_len:
            seq = seq[-self.max_seq_len:]
        
        # 按需生成负样本池（固定且可复现）
        neg_pool = self._get_neg_pool(user_id)
        
        return (
            torch.tensor(seq, dtype=torch.long),
            torch.tensor(target, dtype=torch.long),
            torch.tensor(neg_pool, dtype=torch.long)
        )


class MovieLensDataProcessor:
    """MovieLens 数据处理器 - 带缓存"""
    
    _cache = {}  # 类级别缓存
    
    @staticmethod
    def process(data_dir: str, max_seq_len: int = 200, min_interactions: int = 5):
        import pickle
        import hashlib
        
        ratings_file = os.path.join(data_dir, "ratings.csv")
        if not os.path.exists(ratings_file):
            raise FileNotFoundError(f"找不到 {ratings_file}，请下载 MovieLens 数据集")
        
        # 检查缓存（包含配置参数以确保一致性）
        cache_key = f"{data_dir}_{max_seq_len}_{min_interactions}_v2"  # v2 表示修复后的版本
        cache_file = os.path.join(data_dir, f".cache_{hashlib.md5(cache_key.encode()).hexdigest()}.pkl")
        
        if cache_key in MovieLensDataProcessor._cache:
            print("Using in-memory cache...")
            return MovieLensDataProcessor._cache[cache_key]
        
        if os.path.exists(cache_file):
            print(f"Loading from cache: {cache_file}")
            try:
                with open(cache_file, 'rb') as f:
                    result = pickle.load(f)
                    MovieLensDataProcessor._cache[cache_key] = result
                    return result
            except Exception as e:
                print(f"  Cache load failed: {e}, reprocessing...")
        
        print("Loading MovieLens ratings (this may take 10-30 seconds)...")
        df = pd.read_csv(ratings_file)
        print(f"  Loaded {len(df):,} ratings, sorting...")
        df = df.sort_values(['userId', 'timestamp'])
        
        # 重新映射 item id (从1开始，0是padding)
        print("  Processing items...")
        unique_items = df['movieId'].unique()
        item2id = {old: new for new, old in enumerate(unique_items, start=1)}
        df['item_id'] = df['movieId'].map(item2id)
        
        # 构建用户序列
        print("  Building user sequences...")
        user_seqs_full = {}
        for user_id, group in df.groupby('userId'):
            items = group['item_id'].tolist()
            if len(items) >= min_interactions:
                user_seqs_full[user_id] = items
        
        num_users = len(user_seqs_full)
        num_items = len(item2id)
        
        print(f"Processed: {num_users} users, {num_items} items")
        
        # ==================== 严格的数据划分（无泄露） ====================
        # 划分策略：
        # - 训练集：使用前 n-2 个物品（用于学习序列模式）
        # - 验证集：使用前 n-1 个物品，预测第 n 个
        # - 测试集：使用全部 n 个物品，预测第 n+1 个？不，使用 n-1 个预测第 n 个
        # 
        # 修正：验证集和测试集应该有不同的目标
        # - 验证：序列 items[:-1]，目标 items[-1]
        # - 测试：序列 items[:-1]（但items包含测试目标的前一个）
        
        # 实际上，标准的留一法是：
        # - 训练：items[:-2] 作为历史，预测 items[-2] 之后的
        # - 验证：items[:-1] 作为历史，预测最后一个 items[-1]
        # - 测试：items[:] 作为历史，预测下一个（但数据中没有）
        
        # 正确的 SASRec 划分：
        # 训练时，每个用户的所有前缀都用于训练（除了最后两个）
        # 验证时，使用 items[:-1] 预测 items[-1] 是否在候选中排第一
        # 测试时，使用 items[:] 预测？—— 实际上测试集的目标应该是下一个物品
        
        # 这里我们采用更清晰的划分：
        # - 训练集序列：items[:-2] （足够长用于训练）
        # - 验证集序列：items[:-1] （用于生成验证样本）
        # - 测试集序列：items[:]   （用于生成测试样本）
        
        train_seqs = {}
        val_seqs = {}
        test_seqs = {}
        
        for uid, items in user_seqs_full.items():
            # 至少3个物品才能划分
            if len(items) < 3:
                continue
            
            # 训练：使用前 n-2 个（这样验证和测试的目标不会泄露）
            train_seqs[uid] = items[:-2]
            # 验证：使用前 n-1 个（目标 items[-2] 用于验证）
            val_seqs[uid] = items[:-1]
            # 测试：使用全部 n 个（目标 items[-1] 用于测试）
            test_seqs[uid] = items
        
        print(f"Split: {len(train_seqs)} users for train/val/test")
        
        # 计算热门物品（用于负采样）
        all_items = [i for items in user_seqs_full.values() for i in items]
        item_counts = pd.Series(all_items).value_counts()
        popular_items = item_counts.index.tolist()
        
        stats = {
            'num_users': len(train_seqs),
            'num_items': num_items,
            'item2id': item2id,
            'popular_items': popular_items,
            'avg_seq_len': np.mean([len(s) for s in user_seqs_full.values()]),
            'id2item': {v: k for k, v in item2id.items()}  # 反向映射
        }
        
        result = (train_seqs, val_seqs, test_seqs, stats)
        
        # 保存缓存
        print(f"  Saving cache...")
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)
            MovieLensDataProcessor._cache[cache_key] = result
            print(f"  Cache saved: {cache_file}")
        except Exception as e:
            print(f"  Warning: could not save cache: {e}")
        
        return result


def get_data_loaders(config, clear_cache: bool = False):
    """
    获取数据加载器
    
    Args:
        config: 配置对象
        clear_cache: 是否清除缓存
    
    Returns:
        train_loader, val_loader, test_loader, stats
    """
    # 处理数据
    processor = MovieLensDataProcessor()
    train_seqs, val_seqs, test_seqs, stats = processor.process(
        config.data_dir, 
        max_seq_len=config.max_seq_len
    )
    
    popular_items = stats.get('popular_items', [])
    
    # ==================== 创建训练数据集 ====================
    train_dataset = SequenceTrainDataset(
        train_seqs, 
        config.max_seq_len, 
        stats['num_items'],
        neg_sampling_strategy=config.neg_sampling_strategy,
        popular_items=popular_items,
        popular_alpha=config.popular_items_alpha
    )
    
    # ==================== 创建验证数据集（固定负样本） ====================
    val_dataset = SequenceEvalDataset(
        val_seqs,
        config.max_seq_len,
        stats['num_items'],
        num_neg_samples=config.eval_neg_samples,
        popular_items=popular_items,
        seed=config.seed  # 固定种子保证可复现
    )
    
    # ==================== 创建测试数据集（固定负样本） ====================
    test_dataset = SequenceEvalDataset(
        test_seqs,
        config.max_seq_len,
        stats['num_items'],
        num_neg_samples=config.eval_neg_samples,
        popular_items=popular_items,
        seed=config.seed + 1  # 不同的种子，与验证集区分
    )
    
    # ==================== 创建 DataLoader ====================
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=(config.num_workers > 0 and config.device.type == 'cuda'),
        drop_last=True  # 避免单样本批次导致BN等问题
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=(config.num_workers > 0 and config.device.type == 'cuda')
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=(config.num_workers > 0 and config.device.type == 'cuda')
    )
    
    return train_loader, val_loader, test_loader, stats
