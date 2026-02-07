#!/bin/bash
# 上传模型文件到 GitHub Release
# 需要设置 GITHUB_TOKEN 环境变量

REPO="changQiangXia/sasrec-recommender"
TAG="v1.0.0"
MODEL_FILE="checkpoints/best.pt"

if [ ! -f "$MODEL_FILE" ]; then
    echo "❌ 模型文件不存在: $MODEL_FILE"
    exit 1
fi

if [ -z "$GITHUB_TOKEN" ]; then
    echo "❌ 请设置 GITHUB_TOKEN 环境变量"
    echo "获取方式: https://github.com/settings/tokens"
    echo "需要的权限: repo"
    exit 1
fi

echo "🚀 创建 Release $TAG..."

# 创建 Release
curl -X POST \
    -H "Authorization: token $GITHUB_TOKEN" \
    -H "Accept: application/vnd.github.v3+json" \
    https://api.github.com/repos/$REPO/releases \
    -d "{\"tag_name\":\"$TAG\",\"name\":\"Model Checkpoint\",\"body\":\"SASRec trained model for MovieLens 25M\"}" \
    > release_info.json 2>/dev/null

# 提取 upload_url
UPLOAD_URL=$(cat release_info.json | grep -o '"upload_url": "[^"]*' | cut -d'"' -f4 | sed 's/{?name,label}//')

if [ -z "$UPLOAD_URL" ]; then
    echo "❌ 创建 Release 失败，可能已存在"
    # 获取已有 release 的 upload_url
    curl -H "Authorization: token $GITHUB_TOKEN" \
        https://api.github.com/repos/$REPO/releases/tags/$TAG \
        > release_info.json 2>/dev/null
    UPLOAD_URL=$(cat release_info.json | grep -o '"upload_url": "[^"]*' | cut -d'"' -f4 | sed 's/{?name,label}//')
fi

echo "📤 上传模型文件..."
curl -X POST \
    -H "Authorization: token $GITHUB_TOKEN" \
    -H "Content-Type: application/octet-stream" \
    --data-binary @$MODEL_FILE \
    "$UPLOAD_URL?name=best.pt" \
    > upload_result.json 2>/dev/null

if [ $? -eq 0 ]; then
    echo "✅ 上传成功！"
    echo "下载链接: https://github.com/$REPO/releases/download/$TAG/best.pt"
    rm -f release_info.json upload_result.json
else
    echo "❌ 上传失败"
    cat upload_result.json
fi
