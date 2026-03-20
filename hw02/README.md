# HW02：论文导读与 DeepSeek Chatbot 实践

## 任务一：论文导读

### 1. 论文信息
- **论文题目**：WorkForceAgent-R1: Incentivizing Reasoning Capability in LLM-based Web Agents via Reinforcement Learning
- **作者**：Yuchen Zhuang, Di Jin, Jiaao Chen 等
- **发表年份**：2025
- **论文链接**：https://arxiv.org/abs/2501.12948

### 2. 导读生成方式
使用 DeepSeek 大模型生成导读，提示词要求包含研究背景、核心方法、主要结果和个人小结。生成后进行了人工润色。

### 3. 配图来源
所有配图均从论文 PDF 中手动截图，保存于 `images/` 文件夹：
- `architecture.png`（模型架构图）
- `result_table.png`（实验结果表）
- `training_curve.png`（训练曲线）

### 4. 文件说明
- 导读文档：`导读_WorkForceAgent-R1.md`
- 配图文件夹：`images/`
## 任务二：Chatbot 示例代码（调用 DeepSeek 官方 API）

### 使用平台
- **平台**：DeepSeek 官方 API
- **模型**：deepseek-chat
- **API 地址**：`https://api.deepseek.com/v1`

### 配置方法
1. 注册 DeepSeek 开放平台，获取 API Key。
2. 设置环境变量 `DEEPSEEK_API_KEY`，或在代码中直接填写（注意不要提交到 GitHub）。
3. 安装依赖：`pip install openai`
4. 运行脚本：`python chatbot.py`

### 注意事项
- API Key 请妥善保管，不要公开。
- 免费额度请参考 DeepSeek 官方文档。


