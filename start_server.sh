#!/bin/bash
# 启动 SASRec API 服务

echo "🚀 Starting SASRec API Server..."
echo "   Server will be available at: http://0.0.0.0:8000"
echo "   Health check: http://YOUR_SERVER_IP:8000/health"
echo ""

# 检查模型是否存在
if [ ! -f "./checkpoints/best.pt" ]; then
    echo "❌ Error: Model checkpoint not found at ./checkpoints/best.pt"
    echo "   Please train the model first or check the checkpoint path."
    exit 1
fi

# 启动服务
python api_server.py
