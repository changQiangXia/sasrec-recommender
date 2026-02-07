# SASRec Recommender System 🎬

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

基于 Transformer 的工业级序列推荐系统实现，针对 RTX 4090 优化，提供完整的训练、评估和 API 服务。

## ✨ 特性

- **🧠 Transformer 架构**: Multi-Head Self-Attention + Pre-LN，强大的序列建模能力
- **⚡ 混合精度训练**: `torch.cuda.amp` 加速训练，适配 24GB 显存
- **🎯 增强负采样**: 混合策略（热门+随机），解决训练任务太简单问题
- **📈 学习率调度**: Warmup + Cosine Annealing，大 Batch 优化
- **🚀 FastAPI 服务**: 高性能异步 API，支持批量推荐和相似物品查询
- **💻 Next.js 前端**: 美观的交互界面，实时推荐展示

## 📁 项目结构

```
sasrec-recommender/
├── src/                      # 核心代码
│   ├── model.py              # SASRec 模型实现
│   ├── data_loader.py        # 数据加载与负采样
│   ├── trainer.py            # 训练器 (AMP 混合精度)
│   ├── evaluator.py          # 评估指标 (HR@K, NDCG@K, MRR)
│   ├── config.py             # 配置管理
│   └── utils.py              # 工具函数
├── frontend-nextjs/          # Next.js 前端界面
│   ├── app/                  # 页面组件
│   ├── components/           # 可复用组件
│   └── public/               # 静态资源
├── data/                     # 数据目录
│   └── movielens/            # MovieLens 25M 数据集
├── checkpoints/              # 模型检查点
├── notebooks/                # Jupyter 分析笔记本
├── main.py                   # 训练/评估入口
├── api_server.py             # FastAPI 服务
├── inference.py              # 命令行推理脚本
├── export_item_mapping.py    # 导出物品映射表
├── requirements.txt          # Python 依赖
└── README.md                 # 本文件
```

## 🚀 快速开始

### 环境要求

- Python 3.10+
- CUDA 12.x (推荐)
- PyTorch 2.1+
- 24GB 显存 (RTX 4090，可选)

### 安装依赖

```bash
# 克隆项目
git clone https://github.com/yourusername/sasrec-recommender.git
cd sasrec-recommender

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 1️⃣ 数据准备

**MovieLens 25M**:

```bash
cd data/movielens
wget https://files.grouplens.org/datasets/movielens/ml-25m.zip
unzip ml-25m.zip
mv ml-25m/ratings.csv .
mv ml-25m/movies.csv .
rm -rf ml-25m ml-25m.zip
cd ../..
```

### 2️⃣ 训练模型

```bash
# 基础训练（显存 ~9GB，约 2-3 小时）
python main.py \
    --dataset movielens \
    --batch_size 2048 \
    --epochs 200 \
    --lr 0.001 \
    --hidden_units 128 \
    --neg_strategy mixed

# 大 batch 加速（显存 ~16GB）
python main.py \
    --dataset movielens \
    --batch_size 4096 \
    --epochs 100 \
    --lr 0.002 \
    --warmup_steps 1000
```

### 3️⃣ 评估模型

```bash
python main.py --mode eval --checkpoint ./checkpoints/best.pt
```

### 4️⃣ 启动 API 服务

```bash
# 启动 FastAPI 服务
python api_server.py

# 或使用 uvicorn
uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload
```

服务启动后访问：
- API 文档: http://localhost:8000/docs
- 健康检查: http://localhost:8000/health

### 5️⃣ 启动前端界面

```bash
cd frontend-nextjs

# 安装依赖
npm install

# 开发模式
npm run dev

# 生产构建
npm run build
npm start
```

访问 http://localhost:3000 使用推荐界面。

## 📊 预期性能指标（MovieLens 25M）

| 指标 | 预期值 | 说明 |
|------|--------|------|
| HR@10 | 0.75 - 0.85 | 命中率@10 |
| NDCG@10 | 0.45 - 0.55 | 归一化折损累计增益@10 |
| MRR | 0.35 - 0.45 | 平均倒数排名 |

## 🔧 API 使用示例

### 生成推荐

```bash
curl -X POST "http://localhost:8000/recommend" \
  -H "Content-Type: application/json" \
  -d '{
    "user_history": [1, 2, 3, 4, 5],
    "top_k": 10,
    "exclude_history": true
  }'
```

**响应**:
```json
{
  "user_history": [1, 2, 3, 4, 5],
  "recommendations": [
    {"rank": 1, "item_id": 219, "score": 1.0},
    {"rank": 2, "item_id": 84, "score": 0.8286},
    ...
  ],
  "inference_time_ms": 45.23
}
```

### 批量推荐

```bash
curl -X POST "http://localhost:8000/recommend_batch" \
  -H "Content-Type: application/json" \
  -d '{
    "user_histories": [[1,2,3], [4,5,6]],
    "top_k": 5
  }'
```

### 查询相似物品

```bash
curl "http://localhost:8000/similar_items/1?top_k=5"
```

## 🗺️ 物品映射表

模型内部使用数字 ID，可通过映射表查询电影名称：

```bash
# 生成映射表
python export_item_mapping.py

# 查询电影
grep "^42," results/item_mapping_simple.csv
# 输出: 42,Forrest Gump (1994)
```

映射表位置:
- `results/item_mapping.csv` - 完整信息（ID、标题、类型）
- `results/item_mapping_simple.csv` - 简化版

## ⚙️ 关键参数

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `hidden_units` | 128 | 隐藏层维度，增大以承载更多信息 |
| `num_blocks` | 2 | Transformer 层数 |
| `num_heads` | 4 | 注意力头数 |
| `batch_size` | 2048 | RTX 4090 24GB 可支持 |
| `lr` | 0.001 | 学习率，大 Batch 需要较大值 |
| `neg_strategy` | mixed | 负采样策略（random/popular/mixed）|
| `popular_alpha` | 0.75 | 混合采样中热门物品比例 |

## 🐛 关键修复记录

### 1. 梯度 NaN 导致模型无法学习

**问题**: PyTorch Transformer 的 `src_key_padding_mask` 与 `causal_mask` 组合使用时产生 NaN

**修复**: 统一使用 causal_mask，输出后手动 mask padding

```python
# 修复前: Grad norm = nan, ParamΔ = 0.000000
# 修复后: Grad norm = 0.2335, ParamΔ = 0.00151736 ✅
```

### 2. 数据划分泄露

**问题**: 验证集和测试集包含了训练数据

**修复**: 使用留一法严格划分

### 3. 评估逻辑错误

**问题**: 评估时只用 1 正 + 1 负做二分类

**修复**: 使用 1 正 + 100 负的排序评估

## 📚 引用

```bibtex
@article{kang2018self,
  title={Self-attentive sequential recommendation},
  author={Kang, Wang-Cheng and McAuley, Julian},
  journal={ICDM},
  year={2018}
}
```

## 📄 License

MIT License - 详见 [LICENSE](LICENSE) 文件
