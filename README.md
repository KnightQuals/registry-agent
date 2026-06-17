# Registry Agent

轻量级工具注册型智能体示例：使用兼容 OpenAI Chat Completions 的模型服务解析用户意图，调用本地注册工具，再通过 Web 页面或 JSON API 返回结果。

## 功能

- 通过 `@register_tool` 注册 Python 工具
- 自动从函数签名和 `Annotated` 参数生成工具描述
- 支持多个工具并发执行
- 提供 Gradio Web 界面
- 提供 FastAPI JSON 接口
- 当前内置天气查询和远程媒体检索工具

## 本地运行

创建虚拟环境并安装依赖：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

准备环境变量：

```bash
cp .env.example .env
```

编辑 `.env`，至少配置：

```env
QWEN_API_URL=http://127.0.0.1:8502/v1/chat/completions
QWEN_MODEL_NAME=qwen
SENIVERSE_API_KEY=your_seniverse_api_key
```

如果要使用媒体检索工具，还需要本机或可访问机器上有对应服务：

```env
REMOTE_MEDIA_SERVICE_URL=http://127.0.0.1:8600/api/search
```

启动 Web 和 API 服务：

```bash
python web_server.py
```

访问：

- Web: <http://127.0.0.1:8500>
- API: `POST http://127.0.0.1:8500/api/chat`

API 请求示例：

```bash
curl -X POST http://127.0.0.1:8500/api/chat \
  -H "Content-Type: application/json" \
  -d '{"query":"查一下武汉天气"}'
```

命令行模式：

```bash
python main.py
```

## 配置项

| 变量 | 说明 | 默认值 |
|---|---|---|
| `QWEN_API_URL` | 兼容 OpenAI 的 Chat Completions 接口 | `http://127.0.0.1:8502/v1/chat/completions` |
| `QWEN_MODEL_NAME` | 模型名称 | `qwen` |
| `SENIVERSE_API_KEY` | 心知天气 API Key | 无 |
| `REMOTE_MEDIA_SERVICE_URL` | 媒体检索服务地址 | `http://127.0.0.1:8600/api/search` |
| `OPENAI_API_KEY` | `sample_agent_test.py` 示例使用的 API Key | 无 |
| `OPENAI_BASE_URL` | `sample_agent_test.py` 示例使用的 Base URL | `http://127.0.0.1:8502/v1` |

## 新增工具

在 `tools/` 下创建模块，并使用 `@register_tool` 装饰函数：

```python
from typing import Annotated
from . import register_tool

@register_tool
def echo(text: Annotated[str, "要返回的文本", True]) -> str:
    """原样返回输入文本。"""
    return text
```

然后在 `agent_engine.py` 中导入该模块，让装饰器完成注册。
