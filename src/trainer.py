"""
Trainer
=======
修复后的训练器

关键修复：
1. 正确的评估逻辑：使用固定负样本池进行采样评估（1正 + N负）
2. 学习率调度器：Warmup + Cosine Annealing
3. 修复标签硬编码问题
4. 增加训练稳定性监控
"""

import os
import math
import warnings
import torch
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np

# 过滤掉不影响功能的 PyTorch 警告
warnings.filterwarnings('ignore', message='Support for mismatched src_key_padding_mask and mask is deprecated')
warnings.filterwarnings('ignore', message='enable_nested_tensor is True, but self.use_nested_tensor is False')


class WarmupCosineScheduler:
    """
    Warmup + Cosine Annealing 学习率调度器
    
    适用于大 Batch Size 训练：
    - Warmup 阶段：线性增加学习率，稳定训练初期
    - Cosine 阶段：余弦退火，精细调整
    """
    
    def __init__(self, optimizer, warmup_steps, total_steps, base_lr, min_lr=1e-6):
        """
        Args:
            optimizer: PyTorch 优化器
            warmup_steps: warmup 步数
            total_steps: 总训练步数
            base_lr: 初始学习率
            min_lr: 最小学习率
        """
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.current_step = 0
    
    def step(self):
        """执行一步调度"""
        self.current_step += 1
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr
    
    def get_lr(self):
        """计算当前学习率"""
        if self.current_step < self.warmup_steps:
            # Warmup 阶段：线性增加
            return self.base_lr * (self.current_step / self.warmup_steps)
        else:
            # Cosine Annealing 阶段
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            progress = min(1.0, progress)  # 防止超出
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return self.min_lr + (self.base_lr - self.min_lr) * cosine_decay


class Trainer:
    """
    SASRec 训练器 - 修复版
    """
    
    def __init__(self, model, config, train_loader, val_loader, test_loader):
        self.model = model.to(config.device)
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        # 计算总训练步数
        self.total_steps = config.epochs * len(train_loader)
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.98)
        )
        
        # 学习率调度器
        self.scheduler = WarmupCosineScheduler(
            self.optimizer,
            warmup_steps=config.warmup_steps,
            total_steps=self.total_steps,
            base_lr=config.lr,
            min_lr=config.min_lr
        )
        
        # 混合精度
        self.use_amp = config.use_amp
        self.scaler = GradScaler() if config.use_amp else None
        
        # 创建保存目录
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        
        # 最佳指标追踪
        self.best_metric = 0.0
        self.best_epoch = 0
        self.metrics_history = []
    
    def train_epoch(self, epoch):
        """
        训练一个 epoch
        
        Returns:
            avg_loss: 平均损失
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, (seqs, pos_items, neg_items) in enumerate(pbar):
            # 移动数据到设备
            seqs = seqs.to(self.config.device, non_blocking=True)
            pos_items = pos_items.to(self.config.device, non_blocking=True)
            neg_items = neg_items.to(self.config.device, non_blocking=True)
            
            # 检查是否有有效样本
            if (pos_items > 0).sum() == 0:
                continue
            
            self.optimizer.zero_grad()
            
            # 混合精度训练
            if self.use_amp:
                with autocast():
                    loss = self.model.compute_loss(seqs, pos_items, neg_items)
                
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss = self.model.compute_loss(seqs, pos_items, neg_items)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
            
            # 学习率调度
            current_lr = self.scheduler.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f"{total_loss/num_batches:.4f}",
                'lr': f"{current_lr:.2e}"
            })
        
        return total_loss / max(num_batches, 1)
    
    @torch.no_grad()
    def evaluate(self, loader, split="val"):
        """
        评估模型 - 使用固定负样本池进行采样评估
        
        评估逻辑：
        1. 对于每个用户，给定输入序列
        2. 目标是从 {1正样本 + N负样本} 中选出正样本
        3. 计算 Hit Ratio@K 和 NDCG@K
        
        Args:
            loader: 数据加载器（SequenceEvalDataset）
            split: 数据集名称（val/test）
        
        Returns:
            metrics: 包含 HR@K, NDCG@K, MRR 的字典
        """
        self.model.eval()
        
        all_ranks = []  # 收集所有正样本的排名
        all_scores = []  # 收集分数用于调试
        
        for batch in tqdm(loader, desc=f"Eval {split}"):
            seqs, targets, neg_pools = batch
            
            # 移动数据到设备
            seqs = seqs.to(self.config.device, non_blocking=True)
            targets = targets.to(self.config.device, non_blocking=True)
            neg_pools = neg_pools.to(self.config.device, non_blocking=True)
            
            batch_size = seqs.size(0)
            
            # 过滤无效样本（目标为0表示序列太短）
            valid_mask = targets > 0
            if valid_mask.sum() == 0:
                continue
            
            valid_seqs = seqs[valid_mask]
            valid_targets = targets[valid_mask]
            valid_neg_pools = neg_pools[valid_mask]
            valid_batch_size = valid_seqs.size(0)
            
            # 构建候选集：1个正样本 + N个负样本
            # candidates[i] = [target_i, neg_pool_i[0], neg_pool_i[1], ...]
            candidates = torch.cat([
                valid_targets.unsqueeze(1),  # (B, 1)
                valid_neg_pools              # (B, num_neg_samples)
            ], dim=1)  # (B, 1 + num_neg_samples)
            
            # 批量预测
            logits = self.model.predict(valid_seqs, candidates)  # (B, 1 + num_neg)
            
            # 计算排名（分数降序排列，正样本在位置0）
            # 注意：正样本总是在 candidates 的第 0 列
            rankings = torch.argsort(logits, descending=True, dim=1)  # (B, num_candidates)
            
            # 找到正样本（列0）的排名
            # rankings[i, j] 表示第 i 个样本中原第 j 列的新位置
            # 我们需要找到列0的新位置
            for i in range(valid_batch_size):
                # 找到 0 在 rankings[i] 中的位置
                rank = (rankings[i] == 0).nonzero(as_tuple=True)[0].item() + 1  # 从1开始计数
                all_ranks.append(rank)
        
        if len(all_ranks) == 0:
            return {f'HR@{k}': 0.0 for k in self.config.top_k_list} | {f'NDCG@{k}': 0.0 for k in self.config.top_k_list} | {'MRR': 0.0}
        
        # 转换为 numpy 数组
        all_ranks = np.array(all_ranks)
        
        # 计算指标
        metrics = {}
        for k in self.config.top_k_list:
            # HR@K: 正样本排名 <= K 的比例
            metrics[f'HR@{k}'] = np.mean(all_ranks <= k)
            
            # NDCG@K: 考虑排名的折扣累积增益
            # DCG = 1 / log2(rank + 1) 如果 rank <= K，否则 0
            # 因为只有一个正样本，IDCG = 1（理想情况下排第1）
            dcg = np.where(all_ranks <= k, 1.0 / np.log2(all_ranks + 1), 0.0)
            metrics[f'NDCG@{k}'] = np.mean(dcg)
        
        # MRR: 平均倒数排名
        metrics['MRR'] = np.mean(1.0 / all_ranks)
        
        # 添加统计信息
        metrics['mean_rank'] = np.mean(all_ranks)
        metrics['median_rank'] = np.median(all_ranks)
        
        return metrics
    
    def save_checkpoint(self, epoch, metric):
        """保存最佳模型"""
        checkpoint_path = os.path.join(self.config.checkpoint_dir, 'best.pt')
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': {
                'current_step': self.scheduler.current_step,
                'warmup_steps': self.scheduler.warmup_steps,
                'total_steps': self.scheduler.total_steps,
                'base_lr': self.scheduler.base_lr,
                'min_lr': self.scheduler.min_lr,
            },
            'metric': metric,
            'config': self.config
        }, checkpoint_path)
        print(f"💾 Best model saved (NDCG@10={metric:.4f})")
    
    def load_checkpoint(self, path):
        """加载模型"""
        ckpt = torch.load(path, map_location=self.config.device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        
        # 恢复优化器状态
        if 'optimizer_state_dict' in ckpt:
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        
        # 恢复调度器状态
        if 'scheduler_state_dict' in ckpt:
            sched_state = ckpt['scheduler_state_dict']
            self.scheduler.current_step = sched_state.get('current_step', 0)
            self.scheduler.warmup_steps = sched_state.get('warmup_steps', self.config.warmup_steps)
            self.scheduler.total_steps = sched_state.get('total_steps', self.total_steps)
            self.scheduler.base_lr = sched_state.get('base_lr', self.config.lr)
            self.scheduler.min_lr = sched_state.get('min_lr', self.config.min_lr)
        
        print(f"✅ Loaded checkpoint from epoch {ckpt.get('epoch', 'unknown')}")
    
    def train(self):
        """完整训练流程"""
        print(f"\n🚀 Training on {self.config.device}, AMP={self.use_amp}")
        print(f"   Total steps: {self.total_steps}, Warmup: {self.config.warmup_steps}")
        print(f"   Eval neg samples: {self.config.eval_neg_samples}")
        
        for epoch in range(1, self.config.epochs + 1):
            # 训练
            avg_loss = self.train_epoch(epoch)
            print(f"Epoch {epoch}: Train Loss = {avg_loss:.4f}, LR = {self.scheduler.get_lr():.2e}")
            
            # 评估
            if epoch % 5 == 0 or epoch == 1:
                val_metrics = self.evaluate(self.val_loader, "val")
                
                # 打印详细指标
                print(f"  Val:  HR@10={val_metrics['HR@10']:.4f}, NDCG@10={val_metrics['NDCG@10']:.4f}, MRR={val_metrics['MRR']:.4f}")
                print(f"        Mean Rank={val_metrics['mean_rank']:.1f}, Median Rank={val_metrics['median_rank']:.1f}")
                
                # 保存历史
                self.metrics_history.append({
                    'epoch': epoch,
                    'loss': avg_loss,
                    **val_metrics
                })
                
                # 保存最佳模型
                if val_metrics['NDCG@10'] > self.best_metric:
                    self.best_metric = val_metrics['NDCG@10']
                    self.best_epoch = epoch
                    self.save_checkpoint(epoch, self.best_metric)
                
                # 早停检查
                if epoch - self.best_epoch >= self.config.early_stop_patience:
                    print(f"⏹️ Early stopping at epoch {epoch} (no improvement for {self.config.early_stop_patience} epochs)")
                    break
        
        # 训练结束，输出最终结果
        print(f"\n{'='*60}")
        print(f"🏁 Training Complete")
        print(f"   Best epoch: {self.best_epoch}")
        print(f"   Best Val NDCG@10: {self.best_metric:.4f}")
        print(f"{'='*60}")
        
        # 加载最佳模型进行测试
        best_path = os.path.join(self.config.checkpoint_dir, 'best.pt')
        if os.path.exists(best_path):
            self.load_checkpoint(best_path)
            test_metrics = self.evaluate(self.test_loader, "test")
            
            print(f"\n📝 Test Results:")
            print(f"   HR@10={test_metrics['HR@10']:.4f}, NDCG@10={test_metrics['NDCG@10']:.4f}, MRR={test_metrics['MRR']:.4f}")
            print(f"   Mean Rank={test_metrics['mean_rank']:.1f}, Median Rank={test_metrics['median_rank']:.1f}")
            
            return test_metrics
        else:
            print("⚠️ No checkpoint found, skipping test evaluation")
            return {}
