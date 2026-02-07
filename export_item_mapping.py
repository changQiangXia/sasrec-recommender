#!/usr/bin/env python3
"""
导出物品ID到电影名称的映射表
============================

用法:
    python export_item_mapping.py

输出:
    ./results/item_mapping.csv - 包含原始movieId、映射后的item_id、电影标题、类型
"""

import os
import sys
import pandas as pd
import pickle

def export_mapping():
    """导出物品映射表"""
    
    # 检查数据文件是否存在
    data_dir = "./data/movielens"
    movies_file = os.path.join(data_dir, "movies.csv")
    
    if not os.path.exists(movies_file):
        print(f"❌ 错误: 找不到 {movies_file}")
        print("请先下载 MovieLens 数据集")
        sys.exit(1)
    
    # 读取电影信息
    print("📂 读取电影数据...")
    movies_df = pd.read_csv(movies_file)
    
    # 读取缓存获取映射关系
    cache_files = [f for f in os.listdir(data_dir) if f.startswith('.cache_')]
    if not cache_files:
        print("❌ 错误: 找不到缓存文件，请先运行数据预处理")
        sys.exit(1)
    
    latest_cache = sorted(cache_files)[-1]
    print(f"📂 读取缓存: {latest_cache}")
    
    with open(os.path.join(data_dir, latest_cache), 'rb') as f:
        _, _, _, stats = pickle.load(f)
    
    item2id = stats['item2id']  # 原始movieId -> 新item_id
    id2item = stats.get('id2item', {v: k for k, v in item2id.items()})
    
    print(f"📊 共有 {len(item2id)} 个物品")
    
    # 创建映射表
    mapping_data = []
    
    for original_movie_id, new_item_id in item2id.items():
        # 查找电影信息
        movie_info = movies_df[movies_df['movieId'] == original_movie_id]
        
        if not movie_info.empty:
            title = movie_info.iloc[0]['title']
            genres = movie_info.iloc[0]['genres']
        else:
            title = "Unknown"
            genres = "Unknown"
        
        mapping_data.append({
            'item_id': new_item_id,        # 模型使用的ID (1, 2, 3...)
            'original_movie_id': original_movie_id,  # 原始MovieLens ID
            'title': title,
            'genres': genres
        })
    
    # 创建DataFrame并排序
    mapping_df = pd.DataFrame(mapping_data)
    mapping_df = mapping_df.sort_values('item_id')
    
    # 保存结果
    output_dir = "./results"
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存完整映射表
    output_file = os.path.join(output_dir, "item_mapping.csv")
    mapping_df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"\n✅ 完整映射表已保存: {output_file}")
    
    # 保存只包含常用字段的简化版
    simple_df = mapping_df[['item_id', 'title']].copy()
    simple_file = os.path.join(output_dir, "item_mapping_simple.csv")
    simple_df.to_csv(simple_file, index=False, encoding='utf-8')
    print(f"✅ 简化映射表已保存: {simple_file}")
    
    # 显示前20个作为示例
    print("\n📋 前20个物品映射示例:")
    print("=" * 80)
    print(f"{'item_id':<10} {'original_movie_id':<20} {'title':<50}")
    print("-" * 80)
    for _, row in mapping_df.head(20).iterrows():
        title = row['title'][:47] + "..." if len(row['title']) > 50 else row['title']
        print(f"{row['item_id']:<10} {row['original_movie_id']:<20} {title:<50}")
    print("=" * 80)
    
    # 统计信息
    print(f"\n📈 统计信息:")
    print(f"  总物品数: {len(mapping_df)}")
    print(f"  item_id 范围: {mapping_df['item_id'].min()} - {mapping_df['item_id'].max()}")
    
    # 按类型统计
    print(f"\n🎬 热门类型 Top 10:")
    all_genres = []
    for genres in mapping_df['genres']:
        if genres != "Unknown":
            all_genres.extend(genres.split('|'))
    
    genre_counts = pd.Series(all_genres).value_counts().head(10)
    for genre, count in genre_counts.items():
        print(f"  {genre}: {count}部")
    
    print(f"\n💡 使用说明:")
    print(f"  1. 在前端输入 item_id (如: 1, 2, 3) 作为用户历史")
    print(f"  2. 模型会返回推荐的 item_id")
    print(f"  3. 使用此映射表查找对应的电影名称")

if __name__ == '__main__':
    export_mapping()
