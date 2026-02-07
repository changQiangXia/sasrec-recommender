# SASRec Frontend - Next.js + TypeScript

基于 Next.js + TypeScript + Tailwind CSS 的极简风格推荐系统前端。

## 特性

- ⚡ Next.js 14 + App Router
- 🔒 TypeScript 类型安全
- 🎨 Tailwind CSS 暗色主题
- 📱 响应式设计
- 🔄 实时服务器状态检测
- ✨ 流畅动画效果

## 快速开始

### 1. 安装依赖

```bash
cd frontend-nextjs
npm install
```

### 2. 配置环境变量

```bash
cp .env.local.example .env.local
# 编辑 .env.local，设置你的服务器地址
```

### 3. 开发模式启动

```bash
npm run dev
```

访问 http://localhost:3000

### 4. 生产构建

```bash
npm run build
npm start
```

## API 配置

前端默认连接 `http://localhost:8000`，你可以在页面上直接修改服务器地址，或通过环境变量配置：

```bash
# .env.local
NEXT_PUBLIC_API_URL=http://your-server-ip:8000
```

## 界面预览

- 暗色科技风界面
- 实时服务器状态指示
- 输入用户历史记录获取推荐
- 推荐结果以卡片形式展示，带排名和匹配度
- 显示推理耗时

## 技术栈

- **框架**: Next.js 14
- **语言**: TypeScript
- **样式**: Tailwind CSS
- **字体**: Inter (Google Fonts)
- **图标**: Lucide React (可选)
