"""
Utils
=====
工具函数：随机种子、日志等
"""

import os
import random
import logging
import numpy as np
import torch


def set_seed(seed: int = 42):
    """
    设置随机种子，保证实验可复现
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 多 GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"🎲 Random seed set to {seed}")


def setup_logger(name: str, log_file: str = None, level=logging.INFO):
    """
    设置日志记录器
    
    Args:
        name: 日志器名称
        log_file: 日志文件路径 (可选)
        level: 日志级别
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 控制台输出
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 文件输出
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def count_parameters(model: torch.nn.Module) -> int:
    """
    统计模型参数数量
    
    Args:
        model: PyTorch 模型
        
    Returns:
        可训练参数数量
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_size(size_bytes: int) -> str:
    """
    将字节转换为易读格式
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


def get_device_info():
    """
    获取 GPU 信息
    """
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        total_memory = torch.cuda.get_device_properties(0).total_memory
        return {
            'available': True,
            'name': device_name,
            'memory': format_size(total_memory),
            'cuda_version': torch.version.cuda
        }
    return {'available': False}


def print_config(config):
    """
    打印配置信息
    """
    print("\n" + "="*60)
    print("📋 Configuration:")
    print("="*60)
    for key, value in sorted(config.to_dict().items()):
        print(f"  {key:25s}: {value}")
    print("="*60 + "\n")
