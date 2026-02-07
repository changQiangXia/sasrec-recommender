"use client";

import { useState, useEffect } from 'react';

// 定义推荐结果类型
type Recommendation = {
  rank: number;
  item_id: number;
  score: number;
};

type ApiResponse = {
  user_history: number[];
  recommendations: Recommendation[];
  inference_time_ms: number;
};

export default function Home() {
  const [recommendations, setRecommendations] = useState<Recommendation[]>([]);
  const [loading, setLoading] = useState(false);
  const [serverStatus, setServerStatus] = useState<"online" | "offline" | "checking">("checking");
  const [inferenceTime, setInferenceTime] = useState<number>(0);
  const [error, setError] = useState<string>("");
  
  // 表单状态
  const [serverUrl, setServerUrl] = useState("http://106.39.200.227:8000");
  const [userHistory, setUserHistory] = useState("1, 2, 3, 4, 5");
  const [topK, setTopK] = useState(10);

  // 页面加载时检查服务器状态
  useEffect(() => {
    checkServerStatus();
  }, [serverUrl]);

  // 检查服务器连接
  const checkServerStatus = async () => {
    setServerStatus("checking");
    setError("");
    try {
      const res = await fetch(`${serverUrl}/health`, {
        method: 'GET',
        mode: 'cors',
      });
      if (res.ok) {
        setServerStatus("online");
      } else {
        setServerStatus("offline");
      }
    } catch (error) {
      setServerStatus("offline");
    }
  };

  // 获取推荐
  const fetchRecommendations = async () => {
    setLoading(true);
    setError("");
    // 注意：不再清空 recommendations，保留旧结果直到新结果返回
    
    try {
      // 解析用户历史
      const history = userHistory
        .split(/[,，\s]+/)
        .map(s => parseInt(s.trim()))
        .filter(n => !isNaN(n));

      if (history.length === 0) {
        setError("请输入有效的用户历史记录");
        setLoading(false);
        return;
      }

      // 添加超时控制
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 30000); // 30秒超时

      const res = await fetch(`${serverUrl}/recommend`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_history: history,
          top_k: topK,
          exclude_history: true
        }),
        signal: controller.signal
      });

      clearTimeout(timeoutId);

      if (!res.ok) {
        const errorData = await res.json().catch(() => ({}));
        throw new Error(errorData.detail || `HTTP ${res.status}: 请求失败`);
      }

      const data: ApiResponse = await res.json();
      
      // 验证数据
      if (!data.recommendations || data.recommendations.length === 0) {
        throw new Error("返回的推荐结果为空");
      }
      
      setRecommendations(data.recommendations);
      setInferenceTime(data.inference_time_ms);
      setServerStatus("online");
      setError("");
    } catch (error: any) {
      console.error("请求失败:", error);
      setServerStatus("offline");
      
      if (error.name === 'AbortError') {
        setError("请求超时 (30s)，请检查服务器是否正常运行");
      } else {
        setError(`请求失败: ${error.message}`);
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="min-h-screen bg-black text-white p-4 md:p-8 font-sans selection:bg-purple-500 selection:text-white">
      {/* 顶部导航 */}
      <header className="max-w-7xl mx-auto flex flex-col md:flex-row justify-between items-center mb-12 border-b border-gray-800 pb-6 gap-4">
        <div>
          <h1 className="text-3xl md:text-4xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-purple-400 to-cyan-400">
            SASRec
            <span className="text-xs md:text-sm bg-gray-800 text-gray-300 px-2 py-1 rounded ml-3 font-normal">
              Transformer Powered
            </span>
          </h1>
          <p className="text-gray-500 text-sm mt-2">序列推荐系统</p>
        </div>
        
        <div className="flex items-center gap-3">
          <span className={`w-2 h-2 rounded-full ${
            serverStatus === "online" ? "bg-green-500 animate-pulse" :
            serverStatus === "checking" ? "bg-yellow-500" : "bg-red-500"
          }`}></span>
          <span className="text-sm text-gray-400">
            {serverStatus === "online" ? "服务器在线" :
             serverStatus === "checking" ? "检查中..." : "服务器离线"}
          </span>
        </div>
      </header>

      {/* 主内容区 */}
      <div className="max-w-7xl mx-auto">
        
        {/* 配置面板 */}
        <div className="bg-gray-900/50 border border-gray-800 rounded-2xl p-6 mb-8">
          <h2 className="text-lg font-semibold mb-4 text-gray-300">API 配置</h2>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
            <div>
              <label className="block text-xs text-gray-500 uppercase tracking-wider mb-2">
                服务器地址
              </label>
              <input
                type="text"
                value={serverUrl}
                onChange={(e) => setServerUrl(e.target.value)}
                className="w-full px-4 py-2 bg-black border border-gray-700 rounded-lg text-white focus:outline-none focus:border-purple-500 transition"
                placeholder="http://localhost:8000"
              />
            </div>
            
            <div>
              <label className="block text-xs text-gray-500 uppercase tracking-wider mb-2">
                推荐数量
              </label>
              <input
                type="number"
                min={1}
                max={100}
                value={topK}
                onChange={(e) => setTopK(parseInt(e.target.value) || 10)}
                className="w-full px-4 py-2 bg-black border border-gray-700 rounded-lg text-white focus:outline-none focus:border-purple-500 transition"
              />
            </div>
          </div>
          
          <div>
            <label className="block text-xs text-gray-500 uppercase tracking-wider mb-2">
              用户历史记录 (物品ID，逗号分隔)
            </label>
            <input
              type="text"
              value={userHistory}
              onChange={(e) => setUserHistory(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && fetchRecommendations()}
              className="w-full px-4 py-2 bg-black border border-gray-700 rounded-lg text-white focus:outline-none focus:border-purple-500 transition"
              placeholder="例如: 1, 2, 3, 4, 5"
            />
          </div>
        </div>

        {/* 错误提示 */}
        {error && (
          <div className="mb-6 p-4 bg-red-900/30 border border-red-800 rounded-xl text-red-300">
            <p className="font-semibold mb-1">请求出错</p>
            <p className="text-sm">{error}</p>
            <button 
              onClick={checkServerStatus}
              className="mt-2 text-xs underline hover:text-red-200"
            >
              重新检查服务器状态
            </button>
          </div>
        )}

        {/* 操作按钮 */}
        <div className="flex justify-between items-center mb-8">
          <h2 className="text-xl text-gray-300">
            {recommendations.length > 0 ? 
              `推荐结果 (${recommendations.length}个)` : 
              "点击按钮获取推荐"
            }
          </h2>
          
          <button
            onClick={fetchRecommendations}
            disabled={loading || serverStatus === "offline"}
            className="px-6 py-3 bg-white text-black font-bold rounded-full hover:bg-gray-200 transition-all duration-200 active:scale-95 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
          >
            {loading ? (
              <>
                <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none"/>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"/>
                </svg>
                计算中...
              </>
            ) : (
              "获取推荐"
            )}
          </button>
        </div>

        {/* 推理时间 */}
        {inferenceTime > 0 && !loading && (
          <div className="mb-6 flex items-center gap-2">
            <span className="text-sm text-gray-500">推理耗时:</span>
            <span className="text-sm font-mono text-cyan-400">{inferenceTime.toFixed(2)} ms</span>
          </div>
        )}

        {/* 推荐结果网格 */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {recommendations.map((item, index) => (
            <div
              key={`${item.rank}-${item.item_id}`}
              className="group relative bg-gray-900 border border-gray-800 rounded-xl p-6 hover:border-purple-500/50 transition-all duration-300 hover:shadow-[0_0_30px_rgba(168,85,247,0.15)]"
              style={{ 
                animation: `fadeIn 0.5s ease-out ${index * 0.05}s backwards`,
                opacity: 0
              }}
            >
              {/* 排名 */}
              <div className={`absolute top-4 left-4 w-8 h-8 rounded-full flex items-center justify-center text-sm font-bold ${
                item.rank <= 3 
                  ? "bg-gradient-to-br from-purple-500 to-pink-500" 
                  : "bg-gray-800 text-gray-400"
              }`}>
                {item.rank}
              </div>

              {/* 匹配度得分 */}
              <div className="absolute top-4 right-4 text-right">
                <span className="text-2xl font-bold text-cyan-400">
                  {(item.score * 100).toFixed(1)}%
                </span>
                <p className="text-xs text-gray-500">匹配度</p>
              </div>

              {/* 物品信息 */}
              <div className="mt-12">
                <span className="inline-block px-2 py-1 bg-gray-800 rounded text-xs text-gray-400 mb-2">
                  物品 ID
                </span>
                <h3 className="text-2xl font-bold text-white mb-3">
                  #{item.item_id}
                </h3>
                
                {/* 得分进度条 */}
                <div className="w-full bg-gray-800 h-1.5 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-gradient-to-r from-purple-500 to-cyan-500 transition-all duration-1000"
                    style={{ 
                      width: `${item.score * 100}%`,
                      transitionDelay: `${index * 50}ms`
                    }}
                  ></div>
                </div>
                <p className="text-xs text-gray-500 mt-1">
                  原始分数: {item.score.toFixed(6)}
                </p>
              </div>
            </div>
          ))}
        </div>

        {/* 空状态 */}
        {!loading && recommendations.length === 0 && !error && (
          <div className="text-center py-20 text-gray-600">
            <svg className="w-16 h-16 mx-auto mb-4 opacity-50" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
            </svg>
            <p>输入用户历史记录并点击"获取推荐"</p>
          </div>
        )}
      </div>

      {/* 简单的 CSS 动画 */}
      <style jsx global>{`
        @keyframes fadeIn {
          from { opacity: 0; transform: translateY(20px); }
          to { opacity: 1; transform: translateY(0); }
        }
      `}</style>
    </main>
  );
}
