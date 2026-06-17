import requests
import json
import os
from typing import Annotated
from . import register_tool

# 请确保 IP 和端口正确
REMOTE_SERVICE_URL = os.getenv("REMOTE_MEDIA_SERVICE_URL", "http://127.0.0.1:8600/api/search")


@register_tool
def search_remote_media(
        query: Annotated[str, "对想要查找的图片或视频内容的描述，例如'会议视频'、'猫的照片'", True],
        directory: Annotated[str, "指定搜索的目标文件夹路径。如果不填，将搜索服务器默认配置的资料库。", False] = None
) -> str:
    """
    调用远程多媒体检索服务，查找图片或视频。
    当用户询问“找一下会议视频”、“有没有xxx的图片”时使用。
    """
    print(f"🌐 [调用远程检索] 目标: {REMOTE_SERVICE_URL}")
    print(f"    关键词: {query}, 指定目录: {directory}")

    # 构造请求载荷
    payload = {
        "query": query,
        "top_k": 3
    }

    # 关键修改：适配新接口的 List 类型
    if directory:
        # 如果用户指定了目录，将其包装为 List 发送
        payload["directory"] = [directory]
    else:
        # 如果用户没指定，不发送 directory 字段
        # 让后端服务使用它代码里配置的 DEFAULT_SEARCH_DIRS
        pass

    try:
        # 发送请求
        response = requests.post(REMOTE_SERVICE_URL, json=payload, timeout=30)

        if response.status_code != 200:
            return f"远程服务报错: HTTP {response.status_code} - {response.text}"

        data = response.json()

        if data.get("status") != "success":
            return f"检索服务反馈错误: {data.get('msg')}"

        results = data.get("results", [])
        if not results:
            return "远程服务反馈：在指定范围内未找到匹配的媒体文件。"

        # 格式化结果给大模型看
        report = f"共找到 {len(results)} 个结果：\n"
        for item in results:
            icon = "🎬" if item['type'] == 'video' else "🖼️"
            report += f"{icon} 文件名: {item['filename']}\n"
            report += f"   相关性: {item['score']:.2f}\n"
            # report += f"   语义描述: {item['desc']}\n"
            report += f"   下载链接: {item['download_url']}\n"

            # 如果是图片，构造 Markdown 图片语法，以便在网页端渲染预览
            if item['type'] == 'image':
                report += f"   预览: ![{item['filename']}]({item['download_url']})\n"

            report += "---\n"

        return report

    except requests.exceptions.ConnectionError:
        return f"连接失败：无法连接到检索服务 ({REMOTE_SERVICE_URL})，请确认该服务已启动且端口 8600 开放。"
    except Exception as e:
        return f"调用过程中发生未知错误: {str(e)}"
