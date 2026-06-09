"""Streamlit 前端：多轮对话界面 + 示例问题 + Loading spinner。"""

import json
import os
import uuid

import httpx
import streamlit as st

# 配置
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# Page config
st.set_page_config(
    page_title="张雪峰 AI 志愿填报顾问",
    page_icon="🎓",
    layout="wide",
)

# CSS 样式优化
st.markdown(
    """
<style>
.chat-message {
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
}
.user-message {
    background-color: #e3f2fd;
    border-left: 4px solid #2196f3;
}
.assistant-message {
    background-color: #f5f5f5;
    border-left: 4px solid #4caf50;
}
.role-label {
    font-weight: bold;
    margin-bottom: 0.25rem;
}
</style>
""",
    unsafe_allow_html=True,
)

# 示例问题
SAMPLE_QUESTIONS = [
    "河南560分想学金融，你怎么看？",
    "计算机和临床医学选哪个？",
    "普通家庭孩子选什么专业好？",
    "新闻学专业就业前景怎么样？",
    "600分在河南能上什么好学校？",
    "电气工程及其自动化好不好就业？",
]


def init_session():
    """初始化 Streamlit session state。"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())


def render_message(role: str, content: str):
    """渲染单条聊天消息。"""
    css_class = "user-message" if role == "user" else "assistant-message"
    label = "🧑 你" if role == "user" else "🎓 张雪峰"
    st.markdown(
        f"""
<div class="chat-message {css_class}">
    <div class="role-label">{label}</div>
    <div>{content}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def call_backend(message: str, history: list[dict]) -> str:
    """调用后端同步聊天接口。"""
    url = f"{BACKEND_URL}/chat/sync"
    payload = {
        "message": message,
        "history": history,
        "session_id": st.session_state.session_id,
    }
    try:
        with httpx.Client(timeout=60.0) as client:
            response = client.post(url, json=payload)
            response.raise_for_status()
            return response.json().get("message", "未收到有效回复")
    except httpx.ConnectError:
        return "⚠️ 无法连接到后端服务，请确认后端已启动（http://localhost:8000）"
    except httpx.HTTPStatusError as e:
        return f"⚠️ 后端返回错误: {e.response.status_code}"
    except Exception as e:
        return f"⚠️ 请求失败: {str(e)}"


def main():
    init_session()

    # 标题
    st.title("🎓 张雪峰 AI 志愿填报顾问")
    st.caption(
        "基于张雪峰思维操作系统的数据驱动志愿填报建议。"
        "我以张雪峰视角和你聊，基于公开言论推断，非本人观点。"
    )

    # 示例问题按钮（一行6个）
    st.markdown("### 💡 示例问题")
    cols = st.columns(3)
    for i, question in enumerate(SAMPLE_QUESTIONS):
        col = cols[i % 3]
        if col.button(question, key=f"sample_{i}", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": question})
            # 触发自动回复
            st.session_state.trigger_reply = question

    # 历史聊天记录（可折叠）
    if st.session_state.messages:
        with st.expander(f"📜 历史对话 ({len(st.session_state.messages)} 条)", expanded=False):
            for i, msg in enumerate(st.session_state.messages):
                render_message(msg["role"], msg["content"])

    # 聊天历史展示
    st.markdown("### 💬 对话")
    for msg in st.session_state.messages:
        render_message(msg["role"], msg["content"])

    # 输入框
    user_input = st.chat_input("输入你的问题，例如：河南560分选什么专业好？")

    # 处理用户输入
    message = user_input or st.session_state.get("trigger_reply")

    if message:
        # 清除触发标记
        if "trigger_reply" in st.session_state:
            del st.session_state.trigger_reply

        # 如果不是示例按钮触发的（已经在上面添加了），添加用户消息
        if message != st.session_state.get("trigger_reply") or message not in [m["content"] for m in st.session_state.messages if m["role"] == "user"]:
            if not st.session_state.messages or st.session_state.messages[-1]["content"] != message:
                st.session_state.messages.append({"role": "user", "content": message})

        # 调用后端
        with st.spinner("🎓 张雪峰正在查看数据并思考中..."):
            # 构建历史（不含最新消息）
            history = st.session_state.messages[:-1] if st.session_state.messages else []
            response = call_backend(message, history)

        st.session_state.messages.append({"role": "assistant", "content": response})

        # 刷新页面显示最新回复
        st.rerun()


if __name__ == "__main__":
    main()
