#!/usr/bin/env python3
"""
SASRec Recommender System - Main Entry
======================================
基于 Transformer 的序列推荐系统主入口 - 修复版

关键修复：
1. 正确处理 torch.device
2. 更好的错误处理
3. 清晰的日志输出
"""

import os
import sys
import argparse
import torch

# 添加 src 到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.config import get_config, Config
from src.data_loader import get_data_loaders
from src.model import SASRec
from src.trainer import Trainer
from src.utils import set_seed, count_parameters


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='SASRec Training and Evaluation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 数据集
    parser.add_argument('--dataset', type=str, default='movielens',
                       choices=['movielens', 'taobao'],
                       help='数据集名称')
    parser.add_argument('--data_dir', type=str, default=None,
                       help='数据目录（默认: ./data/{dataset}）')
    
    # 模型参数
    parser.add_argument('--hidden_units', type=int, default=128,
                       help='隐藏层维度')
    parser.add_argument('--num_blocks', type=int, default=2,
                       help='Transformer 层数')
    parser.add_argument('--num_heads', type=int, default=4,
                       help='注意力头数')
    parser.add_argument('--dropout', type=float, default=0.2,
                       help='Dropout 率')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=2048,
                       help='训练 batch size')
    parser.add_argument('--epochs', type=int, default=200,
                       help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='初始学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='权重衰减')
    parser.add_argument('--warmup_steps', type=int, default=2000,
                       help='Warmup 步数')
    
    # 负采样
    parser.add_argument('--neg_strategy', type=str, default='mixed',
                       choices=['random', 'popular', 'mixed'],
                       help='负采样策略')
    parser.add_argument('--popular_alpha', type=float, default=0.75,
                       help='混合采样中热门物品比例')
    parser.add_argument('--eval_neg_samples', type=int, default=100,
                       help='评估时负样本数')
    
    # 混合精度
    parser.add_argument('--no_amp', action='store_true',
                       help='禁用自动混合精度')
    
    # 其他
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    parser.add_argument('--num_workers', type=int, default=0,
                       help='DataLoader 工作进程数')
    
    # 模式
    parser.add_argument('--mode', type=str, default='train',
                       choices=['train', 'eval'],
                       help='运行模式')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='评估时加载的检查点路径')
    parser.add_argument('--resume', type=str, default=None,
                       help='恢复训练的检查点路径')
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 检测设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"🔥 Device: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA: {torch.version.cuda}")
        print(f"   PyTorch: {torch.__version__}")
    print(f"{'='*60}")
    
    # 获取配置
    config = get_config(args.dataset)
    
    # 用命令行参数覆盖配置
    config.device = device
    config.data_dir = args.data_dir if args.data_dir else f"./data/{args.dataset}"
    config.hidden_units = args.hidden_units
    config.num_blocks = args.num_blocks
    config.num_heads = args.num_heads
    config.dropout = args.dropout
    config.batch_size = args.batch_size
    config.epochs = args.epochs
    config.lr = args.lr
    config.weight_decay = args.weight_decay
    config.warmup_steps = args.warmup_steps
    config.neg_sampling_strategy = args.neg_strategy
    config.popular_items_alpha = args.popular_alpha
    config.eval_neg_samples = args.eval_neg_samples
    config.use_amp = (not args.no_amp) and torch.cuda.is_available()
    config.seed = args.seed
    config.num_workers = args.num_workers
    
    # 打印配置
    print(f"\n📋 Configuration:")
    print(config)
    
    # 加载数据
    print("\n📂 Loading data...")
    try:
        train_loader, val_loader, test_loader, stats = get_data_loaders(config)
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print(f"\n请下载 MovieLens 25M 数据集:")
        print(f"  wget https://files.grouplens.org/datasets/movielens/ml-25m.zip")
        print(f"  unzip ml-25m.zip -d ./data/movielens/")
        return 1
    except Exception as e:
        print(f"\n❌ Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    num_items = stats['num_items']
    print(f"\n📊 Data Statistics:")
    print(f"   Users: {stats['num_users']:,}")
    print(f"   Items: {num_items:,}")
    print(f"   Avg Seq Len: {stats['avg_seq_len']:.1f}")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    print(f"   Test batches: {len(test_loader)}")
    
    # 创建模型
    print("\n🏗️  Building model...")
    model = SASRec(num_items=num_items, config=config)
    num_params = count_parameters(model)
    print(f"   Parameters: {num_params:,} ({num_params/1e6:.2f}M)")
    
    # 创建训练器
    trainer = Trainer(model, config, train_loader, val_loader, test_loader)
    
    # 恢复检查点
    if args.resume:
        if not os.path.exists(args.resume):
            print(f"\n❌ Checkpoint not found: {args.resume}")
            return 1
        print(f"\n📥 Resuming from {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # 训练或评估
    if args.mode == 'train':
        print("\n🏃 Starting training...")
        try:
            test_metrics = trainer.train()
            print("\n✅ Training completed successfully!")
            return 0
        except KeyboardInterrupt:
            print("\n\n⚠️ Training interrupted by user")
            # 保存中断时的模型
            interrupt_path = os.path.join(config.checkpoint_dir, 'interrupted.pt')
            trainer.save_checkpoint(trainer.scheduler.current_step // len(train_loader), 0)
            print(f"Checkpoint saved to {interrupt_path}")
            return 0
        except Exception as e:
            print(f"\n❌ Training failed: {e}")
            import traceback
            traceback.print_exc()
            return 1
    
    elif args.mode == 'eval':
        if args.checkpoint is None:
            print("❌ Error: --checkpoint required for eval mode")
            return 1
        if not os.path.exists(args.checkpoint):
            print(f"❌ Error: Checkpoint not found: {args.checkpoint}")
            return 1
        
        print(f"\n📊 Evaluating {args.checkpoint}")
        trainer.load_checkpoint(args.checkpoint)
        
        val_metrics = trainer.evaluate(val_loader, "val")
        test_metrics = trainer.evaluate(test_loader, "test")
        
        # 打印到控制台
        print(f"\n📈 Evaluation Results:")
        print(f"{'='*60}")
        print(f"Val Results:")
        print(f"  HR@10:  {val_metrics['HR@10']:.4f}")
        print(f"  NDCG@10: {val_metrics['NDCG@10']:.4f}")
        print(f"  MRR:    {val_metrics['MRR']:.4f}")
        print(f"\nTest Results:")
        print(f"  HR@10:  {test_metrics['HR@10']:.4f}")
        print(f"  NDCG@10: {test_metrics['NDCG@10']:.4f}")
        print(f"  MRR:    {test_metrics['MRR']:.4f}")
        print(f"{'='*60}")
        
        # 保存结果到文件
        results_dir = "./results"
        os.makedirs(results_dir, exist_ok=True)
        
        # 生成文件名（包含时间戳）
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_name = os.path.basename(args.checkpoint).replace('.pt', '')
        result_file = os.path.join(results_dir, f"eval_{checkpoint_name}_{timestamp}.txt")
        
        # 写入结果
        with open(result_file, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("SASRec Evaluation Results\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"Evaluation Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Checkpoint: {args.checkpoint}\n")
            f.write(f"Dataset: {config.dataset}\n")
            f.write(f"Num Items: {num_items:,}\n")
            f.write(f"Num Users: {stats['num_users']:,}\n\n")
            
            f.write("-"*60 + "\n")
            f.write("Validation Results:\n")
            f.write("-"*60 + "\n")
            f.write(f"  HR@10:   {val_metrics['HR@10']:.4f}\n")
            f.write(f"  NDCG@10: {val_metrics['NDCG@10']:.4f}\n")
            f.write(f"  MRR:     {val_metrics['MRR']:.4f}\n")
            f.write(f"  Mean Rank: {val_metrics['mean_rank']:.1f}\n")
            f.write(f"  Median Rank: {val_metrics['median_rank']:.1f}\n\n")
            
            f.write("-"*60 + "\n")
            f.write("Test Results:\n")
            f.write("-"*60 + "\n")
            f.write(f"  HR@10:   {test_metrics['HR@10']:.4f}\n")
            f.write(f"  NDCG@10: {test_metrics['NDCG@10']:.4f}\n")
            f.write(f"  MRR:     {test_metrics['MRR']:.4f}\n")
            f.write(f"  Mean Rank: {test_metrics['mean_rank']:.1f}\n")
            f.write(f"  Median Rank: {test_metrics['median_rank']:.1f}\n\n")
            
            # 其他指标
            for k in [5, 10, 20]:
                if f'HR@{k}' in val_metrics:
                    f.write(f"  HR@{k}:    {val_metrics[f'HR@{k}']:.4f} (Val)  {test_metrics[f'HR@{k}']:.4f} (Test)\n")
                if f'NDCG@{k}' in val_metrics:
                    f.write(f"  NDCG@{k}:  {val_metrics[f'NDCG@{k}']:.4f} (Val)  {test_metrics[f'NDCG@{k}']:.4f} (Test)\n")
            
            f.write("\n" + "="*60 + "\n")
        
        print(f"\n💾 Results saved to: {result_file}")
        
        return 0


if __name__ == '__main__':
    sys.exit(main())
