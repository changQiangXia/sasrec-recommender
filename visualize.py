#!/usr/bin/env python3
"""
训练过程可视化
"""

import matplotlib.pyplot as plt
import numpy as np
import pickle
import os

def plot_training_history(checkpoint_dir='./checkpoints'):
    """从 checkpoints 中提取训练历史并可视化"""
    
    # 尝试加载最佳模型的 checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, 'best.pt')
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 如果有训练历史记录
    if 'metrics_history' in checkpoint:
        history = checkpoint['metrics_history']
        
        epochs = [h['epoch'] for h in history]
        losses = [h['loss'] for h in history]
        ndcg10 = [h['NDCG@10'] for h in history]
        hr10 = [h['HR@10'] for h in history]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Loss
        axes[0, 0].plot(epochs, losses, 'b-o')
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True)
        
        # NDCG@10
        axes[0, 1].plot(epochs, ndcg10, 'g-s')
        axes[0, 1].set_title('Validation NDCG@10')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('NDCG@10')
        axes[0, 1].grid(True)
        
        # HR@10
        axes[1, 0].plot(epochs, hr10, 'r-^')
        axes[1, 0].set_title('Validation HR@10')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('HR@10')
        axes[1, 0].grid(True)
        
        # 综合对比
        ax2 = axes[1, 1]
        ax2.plot(epochs, losses, 'b-o', label='Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss', color='b')
        ax2.tick_params(axis='y', labelcolor='b')
        
        ax3 = ax2.twinx()
        ax3.plot(epochs, ndcg10, 'g-s', label='NDCG@10')
        ax3.set_ylabel('NDCG@10', color='g')
        ax3.tick_params(axis='y', labelcolor='g')
        
        axes[1, 1].set_title('Loss vs NDCG@10')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('./training_history.png', dpi=150)
        print("可视化已保存到 ./training_history.png")
        plt.show()
    else:
        print("No training history found in checkpoint")


if __name__ == '__main__':
    import torch
    plot_training_history()
