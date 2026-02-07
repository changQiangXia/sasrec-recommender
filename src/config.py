"""
Configuration Manager for SASRec
==============================
超参数配置管理 - 修复版

针对 RTX 4090 (24GB VRAM) 优化

关键修复：
1. 所有参数都有合理默认值
2. 支持 MovieLens 和 Taobao 数据集
3. 设备类型使用 torch.device
"""

import torch
from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    """
    SASRec 配置类
    
    所有参数都经过精心调整，适用于 MovieLens 25M 数据集
    """
    
    # ========== 数据配置 ==========
    dataset: str = "movielens"  # 可选: movielens, taobao
    data_dir: str = "./data/movielens"
    
    # 序列长度配置
    # MovieLens: 用户平均观影约 100-200 部，设为 200 可以覆盖大部分序列
    max_seq_len: int = 200
    
    # 训练/验证/测试分割（留一法）
    # 不需要显式设置比例，使用留一法划分
    
    # ========== 模型架构 ==========
    # MovieLens 25M 有 6万+ 物品，2500万+ 交互，需要足够大的模型容量
    hidden_units: int = 128      # 嵌入维度（原 SASRec 默认 64 太小）
    num_blocks: int = 2          # Transformer 层数（2-4）
    num_heads: int = 4           # 注意力头数（必须整除 hidden_units）
    dropout: float = 0.2         # Dropout 率
    
    # ========== 训练配置 ==========
    # 针对 RTX 4090 24GB VRAM 优化
    batch_size: int = 2048       # 大 batch 训练更稳定
    epochs: int = 200
    
    # 学习率策略（大 batch 需要 warmup + cosine annealing）
    lr: float = 0.001            # 初始学习率
    weight_decay: float = 1e-4   # L2 正则化
    
    # Warmup + Cosine Annealing
    warmup_steps: int = 2000     # Warmup 步数（约占前 1-2 个 epoch）
    min_lr: float = 1e-6         # 最小学习率
    
    # ========== 负采样配置 ==========
    # 训练时每个正样本对应的负样本数
    # 增大此值可以让训练任务更接近真实场景
    num_neg_samples: int = 1     # 训练时的负采样数（1:1）
    
    # 负采样策略: random, popular, mixed
    neg_sampling_strategy: str = "mixed"
    
    # 混合采样中热门物品的比例
    popular_items_alpha: float = 0.75
    
    # ========== 评估配置 ==========
    # 关键修复：评估时使用固定的负样本池
    top_k_list: List[int] = field(default_factory=lambda: [5, 10, 20])
    eval_batch_size: int = 512   # 评估 batch size（可以比训练小）
    eval_neg_samples: int = 100  # 评估时每个正样本对应的负样本数
                                 # 总计 101 个候选（1正+100负）
    
    # ========== 硬件/混合精度配置 ==========
    # 设备类型使用 torch.device
    device: torch.device = field(
        default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    use_amp: bool = True         # 启用混合精度 (torch.cuda.amp)
    num_workers: int = 0         # DataLoader 工作进程（0=主进程，避免多进程问题）
    
    # ========== 日志与保存 ==========
    seed: int = 42
    log_interval: int = 100      # 每多少 step 打印日志
    save_interval: int = 10      # 每多少 epoch 保存模型
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    
    # ========== 早停配置 ==========
    early_stop_patience: int = 20   # 早停耐心值（多少个 epoch 不提升就停止）
    early_stop_metric: str = "NDCG@10"  # 早停监控指标
    
    def __post_init__(self):
        """配置校验"""
        # 确保 hidden_units 能被 num_heads 整除
        if self.hidden_units % self.num_heads != 0:
            raise ValueError(
                f"hidden_units ({self.hidden_units}) 必须能被 num_heads ({self.num_heads}) 整除"
            )
        
        # 确保 device 是 torch.device 类型
        if isinstance(self.device, str):
            self.device = torch.device(self.device)
        
        # 针对不同数据集调整参数
        if self.dataset == "taobao":
            self.max_seq_len = 50      # 淘宝行为序列通常较短
            self.batch_size = 4096     # 数据量更大，可以增大 batch
            self.eval_neg_samples = 200  # 更多候选
    
    def to_dict(self):
        """转换为字典，方便日志记录"""
        result = {}
        for k, v in self.__dict__.items():
            if k == 'device':
                result[k] = str(v)
            elif not k.startswith('_'):
                result[k] = v
        return result
    
    def __str__(self):
        """字符串表示"""
        lines = ["="*60, "SASRec Configuration", "="*60]
        for key, value in sorted(self.to_dict().items()):
            lines.append(f"  {key:25s}: {value}")
        lines.append("="*60)
        return "\n".join(lines)


def get_config(dataset: str = "movielens") -> Config:
    """
    获取预设配置
    
    Args:
        dataset: 数据集名称 (movielens/taobao)
    
    Returns:
        Config 对象
    """
    config = Config()
    config.dataset = dataset
    
    if dataset == "movielens":
        config.data_dir = "./data/movielens"
    elif dataset == "taobao":
        config.data_dir = "./data/taobao"
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    return config
