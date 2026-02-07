#!/usr/bin/env python3
"""
SASRec FastAPI 服务
===================
提供推荐 API 接口

启动命令:
    uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload

或者:
    python api_server.py
"""

import os
import sys
import pickle
import torch
from typing import List, Dict, Optional
from datetime import datetime
import uvicorn

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, 'src')
from src.model import SASRec
from src.config import get_config

# ============ 配置 ============
CHECKPOINT_PATH = "./checkpoints/best.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 512  # 批量推理的batch size

# ============ 全局变量（模型缓存）============
model = None
stats = None
item_embeddings = None  # 预计算的物品嵌入

app = FastAPI(
    title="SASRec Recommendation API",
    description="基于 Transformer 的序列推荐系统",
    version="1.0.0"
)

# 允许跨域（开发时使用）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应限制为具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============ 数据模型 ============

class RecommendRequest(BaseModel):
    user_history: List[int]  # 用户历史物品列表，如 [1, 2, 3, 4, 5]
    top_k: int = 10          # 推荐数量
    exclude_history: bool = True  # 是否排除已交互物品

class RecommendResponse(BaseModel):
    user_history: List[int]
    recommendations: List[Dict]
    inference_time_ms: float

class BatchRecommendRequest(BaseModel):
    user_histories: List[List[int]]  # 多个用户的历史
    top_k: int = 10
    exclude_history: bool = True

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    num_items: int
    timestamp: str

# ============ 模型加载 ============

def load_model():
    """加载模型到全局变量"""
    global model, stats, item_embeddings
    
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")
    
    # 加载数据缓存获取 num_items
    cache_files = [f for f in os.listdir('./data/movielens') if f.startswith('.cache_')]
    if not cache_files:
        raise FileNotFoundError("No cache file found")
    
    latest_cache = sorted(cache_files)[-1]
    with open(f'./data/movielens/{latest_cache}', 'rb') as f:
        _, _, _, stats = pickle.load(f)
    
    # 创建模型
    config = get_config('movielens')
    config.device = DEVICE
    config.max_seq_len = 200
    
    model = SASRec(num_items=stats['num_items'], config=config)
    
    # 加载权重
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(DEVICE)
    
    # 预计算物品嵌入（加速推理）
    with torch.no_grad():
        item_embeddings = model.item_emb.weight.data.cpu()
    
    print(f"✅ Model loaded: {CHECKPOINT_PATH}")
    print(f"   Device: {DEVICE}")
    print(f"   Num items: {stats['num_items']:,}")

# ============ API 端点 ============

@app.on_event("startup")
async def startup_event():
    """服务启动时加载模型"""
    load_model()

@app.get("/", response_model=HealthResponse)
async def root():
    """健康检查"""
    return HealthResponse(
        status="running",
        model_loaded=model is not None,
        device=str(DEVICE),
        num_items=stats['num_items'] if stats else 0,
        timestamp=datetime.now().isoformat()
    )

@app.get("/health")
async def health():
    """健康检查（简化版）"""
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/recommend", response_model=RecommendResponse)
async def recommend(request: RecommendRequest):
    """
    为单个用户生成推荐
    
    Example:
        POST /recommend
        {
            "user_history": [1, 2, 3, 4, 5],
            "top_k": 10,
            "exclude_history": true
        }
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    import time
    start_time = time.time()
    
    # 准备输入
    user_history = request.user_history
    max_seq_len = 200
    
    # 截断或填充序列（padding 在前，有效数据在后）
    if len(user_history) > max_seq_len:
        user_history = user_history[-max_seq_len:]
    
    seq = [0] * (max_seq_len - len(user_history)) + user_history
    seq_tensor = torch.tensor([seq], dtype=torch.long).to(DEVICE)
    
    # 准备候选池
    if request.exclude_history:
        candidate_pool = [i for i in range(1, stats['num_items'] + 1) 
                         if i not in user_history]
    else:
        candidate_pool = list(range(1, stats['num_items'] + 1))
    
    # 批量推理（避免一次性加载所有物品导致 OOM）
    import torch.nn.functional as F
    all_scores = []
    with torch.no_grad():
        for i in range(0, len(candidate_pool), BATCH_SIZE):
            batch_items = candidate_pool[i:i + BATCH_SIZE]
            candidates = torch.tensor([batch_items], dtype=torch.long).to(DEVICE)
            scores = model.predict(seq_tensor, candidates)
            all_scores.extend(scores.squeeze(0).cpu().tolist())
    
    # 获取 top-k 原始分数
    import heapq
    top_indices = heapq.nlargest(request.top_k, range(len(all_scores)), 
                                  key=lambda i: all_scores[i])
    
    # 获取 top-k 的原始分数，并做 min-max 归一化到 0-1
    # 这样第一名显示 100%，最后一名显示 0%，中间有区分度
    top_raw_scores = [all_scores[idx] for idx in top_indices]
    min_score = min(top_raw_scores)
    max_score = max(top_raw_scores)
    
    if max_score > min_score:
        normalized_scores = [(s - min_score) / (max_score - min_score) for s in top_raw_scores]
    else:
        normalized_scores = [1.0] * len(top_raw_scores)  # 避免除零
    
    # 构建结果
    recommendations = []
    for rank, (idx, score) in enumerate(zip(top_indices, normalized_scores), 1):
        item_id = candidate_pool[idx]
        recommendations.append({
            "rank": rank,
            "item_id": item_id,
            "score": round(score, 4)  # 0-1 范围，如 0.9537
        })
    
    inference_time = (time.time() - start_time) * 1000  # ms
    
    return RecommendResponse(
        user_history=request.user_history,
        recommendations=recommendations,
        inference_time_ms=round(inference_time, 2)
    )

@app.post("/recommend_batch")
async def recommend_batch(request: BatchRecommendRequest):
    """
    批量为多个用户生成推荐
    
    Example:
        POST /recommend_batch
        {
            "user_histories": [[1,2,3], [4,5,6]],
            "top_k": 10
        }
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    results = []
    for user_history in request.user_histories:
        single_request = RecommendRequest(
            user_history=user_history,
            top_k=request.top_k,
            exclude_history=request.exclude_history
        )
        result = await recommend(single_request)
        results.append({
            "user_history": user_history,
            "recommendations": result.recommendations
        })
    
    return {"results": results, "count": len(results)}

@app.get("/similar_items/{item_id}")
async def similar_items(item_id: int, top_k: int = 10):
    """
    获取与指定物品相似的物品（基于嵌入向量）
    
    Args:
        item_id: 查询的物品ID
        top_k: 返回相似物品数量
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    if item_id < 1 or item_id > stats['num_items']:
        raise HTTPException(status_code=400, detail=f"Item ID out of range")
    
    # 获取目标物品的嵌入
    target_emb = item_embeddings[item_id]
    
    # 计算余弦相似度
    similarities = torch.nn.functional.cosine_similarity(
        target_emb.unsqueeze(0),
        item_embeddings,
        dim=1
    )
    
    # 排除自己，获取 top-k
    similarities[item_id] = -1  # 排除自己
    top_scores, top_indices = torch.topk(similarities, k=min(top_k, len(similarities)-1))
    
    similar_items = []
    for idx, score in zip(top_indices.tolist(), top_scores.tolist()):
        similar_items.append({
            "item_id": idx,
            "similarity": round(score, 4)
        })
    
    return {
        "query_item": item_id,
        "similar_items": similar_items
    }

# ============ 启动 ============

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
