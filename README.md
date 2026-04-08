# any2api

一个统一的 AI 模型代理 API，将多个 AI 提供商（OpenAI、Anthropic）的接口统一为 OpenAI 兼容格式。

## 部署

**Build command（构建命令）：**
```
pnpm --filter @workspace/api-server run build
```

**Run command（运行命令）：**
```
node --enable-source-maps artifacts/api-server/dist/index.mjs
```

健康检查路径：`/api/healthz`，端口：`8080`

## API 端点

Base path: `/api`

| 方法 | 路径 | 描述 |
|------|------|------|
| GET | `/api/healthz` | 健康检查 |
| GET | `/api/v1/models` | 列出所有支持的模型（OpenAI 兼容格式）|
| POST | `/api/v1/chat/completions` | OpenAI 兼容的对话接口（支持流式输出）|
| POST | `/api/v1/messages` | Anthropic 兼容的消息接口（支持流式输出）|

## 模型路由

| 模型系列 | 转发目标 |
|----------|----------|
| claude-opus、claude-sonnet 等 Claude 系列 | Anthropic |
| gpt-5、gemini、grok、o3 等其他模型 | OpenAI |

支持的模型别名包括：`claude-opus-4.6`、`claude-sonnet-4.5`、`gemini-2.5-pro`、`grok-4`、`o3`、`gpt-5` 等。

## 使用示例

### Chat Completions（OpenAI 兼容）

```bash
curl -X POST https://your-domain/api/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <any-token>" \
  -d '{
    "model": "gpt-5",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

### Messages（Anthropic 兼容）

```bash
curl -X POST https://your-domain/api/v1/messages \
  -H "Content-Type: application/json" \
  -H "x-api-key: <any-token>" \
  -d '{
    "model": "claude-opus-4.6",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 1024
  }'
```

### 流式输出

在请求体中添加 `"stream": true` 即可启用流式输出（SSE 格式）。

## 环境变量

| 变量名 | 说明 |
|--------|------|
| `AI_INTEGRATIONS_OPENAI_BASE_URL` | OpenAI 代理 base URL |
| `AI_INTEGRATIONS_OPENAI_API_KEY` | OpenAI API 密钥 |
| `AI_INTEGRATIONS_ANTHROPIC_BASE_URL` | Anthropic 代理 base URL |
| `AI_INTEGRATIONS_ANTHROPIC_API_KEY` | Anthropic API 密钥 |

## 技术栈

- **运行时**：Node.js 24
- **框架**：Express 5
- **语言**：TypeScript
- **包管理**：pnpm workspaces
- **构建**：esbuild
- **AI SDK**：openai、@anthropic-ai/sdk
