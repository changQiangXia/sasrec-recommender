#!/usr/bin/env python3
"""
导出 SASRec 为 ONNX 格式
供 Node.js/浏览器使用
"""

import torch
import sys
sys.path.insert(0, 'src')

from src.model import SASRec
from src.config import get_config

def export_model():
    config = get_config('movielens')
    config.device = 'cpu'
    
    # 加载模型
    model = SASRec(num_items=59047, config=config)
    checkpoint = torch.load('./checkpoints/best.pt', map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 示例输入
    seq_len = 200
    batch_size = 1
    num_candidates = 100
    
    dummy_seq = torch.randint(0, 59047, (batch_size, seq_len))
    dummy_candidates = torch.randint(0, 59047, (batch_size, num_candidates))
    
    # 导出 ONNX
    torch.onnx.export(
        model,
        (dummy_seq, dummy_candidates),
        './checkpoints/sasrec.onnx',
        input_names=['seq', 'candidates'],
        output_names=['scores'],
        dynamic_axes={
            'seq': {0: 'batch_size'},
            'candidates': {0: 'batch_size', 1: 'num_candidates'},
            'scores': {0: 'batch_size', 1: 'num_candidates'}
        },
        opset_version=11
    )
    
    print("✅ Model exported to ./checkpoints/sasrec.onnx")

if __name__ == '__main__':
    export_model()
