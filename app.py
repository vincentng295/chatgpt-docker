import os
import json
import base64
import time
from typing import List, Dict, Any

import streamlit as st
from openai import OpenAI
import db  # Import module CSDL của chúng ta

# -----------------------------
# ⚙️ App Config
# -----------------------------
st.set_page_config(
    page_title="ChatGPT-like • Streamlit",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# 🧭 Helpers & Database Integration
# -----------------------------

# Khởi tạo CSDL khi ứng dụng bắt đầu
db.init_db()

def get_client(api_key: str | None) -> OpenAI | None:
    if not api_key:
        return None
    try:
        return OpenAI(api_key=api_key)
    except Exception:
        return None

def image_to_data_url(file) -> str:
    bytes_data = file.read()
    mime = file.type or "image/png"
    b64 = base64.b64encode(bytes_data).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def init_state():
    if "api_key" not in st.session_state:
        # Lấy API key từ biến môi trường của Docker
        st.session_state.api_key = os.getenv("OPENAI_API_KEY", "")
    if "current_chat_id" not in st.session_state:
        # Chọn cuộc trò chuyện đầu tiên trong danh sách (nếu có)
        all_chats = db.get_all_chats()
        st.session_state.current_chat_id = all_chats[0]['id'] if all_chats else None
    if "settings" not in st.session_state:
        st.session_state.settings = {
            "model": "gpt-4o-mini",
            "max_output_tokens": 1024,
            "system_prompt": "Bạn là Huệ — một trợ lý hữu ích, nói tiếng Việt, súc tích và thân thiện.",
        }

def new_chat():
    chat_id = str(int(time.time() * 1000))
    title = f"Cuộc trò chuyện mới"
    messages = [{"role": "system", "content": st.session_state.settings["system_prompt"]}]
    db.save_chat(chat_id, title, messages)
    st.session_state.current_chat_id = chat_id
    st.rerun()

def get_current_chat() -> Dict[str, Any] | None:
    return db.get_chat_by_id(st.session_state.current_chat_id)

def ensure_chat_selected():
    if not st.session_state.current_chat_id:
        new_chat()

# ... (Giữ nguyên các hàm tool_get_weather, TOOLS_SPEC, FUNCTIONS_MAP)
def tool_get_weather(args: Dict[str, Any]) -> Dict[str, Any]:
    city = args.get("city", "")
    return {"city": city, "forecast": [{"day": "Hôm nay", "temp_c": 31, "condition": "Nắng"}], "source": "demo-local"}
TOOLS_SPEC = [{"type": "function", "function": {"name": "get_weather", "description": "Tra thời tiết", "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "Tên thành phố"}}, "required": ["city"]}}}]
FUNCTIONS_MAP = {"get_weather": tool_get_weather}
# -----------------------------
# 🧱 UI — Sidebar
# -----------------------------
init_state()

with st.sidebar:
    st.header("⚙️ Cài đặt")

    st.caption("OpenAI API key được nạp từ biến môi trường.")
    api_key_input = st.text_input("OPENAI_API_KEY", value=st.session_state.api_key, type="password", disabled=True)
    
    client = get_client(st.session_state.api_key)
    if client is None:
        st.error("Chưa có API key hợp lệ. Hãy thiết lập trong file .env và khởi động lại Docker.")

    st.session_state.settings["model"] = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o"], index=0)
    st.session_state.settings["max_output_tokens"] = st.slider("Giới hạn token trả lời", 64, 8192, 1024)

    with st.expander("🎛️ System prompt"):
        st.session_state.settings["system_prompt"] = st.text_area("Nội dung", value=st.session_state.settings["system_prompt"], height=120)
        if st.button("Áp dụng cho chat hiện tại"):
            chat = get_current_chat()
            if chat and chat["messages"]:
                chat["messages"][0]["content"] = st.session_state.settings["system_prompt"]
                db.save_chat(chat["id"], chat["title"], chat["messages"])
                st.success("Đã cập nhật system prompt.")

    st.divider()
    st.subheader("💬 Lịch sử trò chuyện")
    
    if st.button("➕ Tạo cuộc trò chuyện mới", use_container_width=True):
        new_chat()

    all_chats = db.get_all_chats()
    for chat_meta in all_chats:
        cid = chat_meta["id"]
        with st.container():
            col1, col2 = st.columns([0.85, 0.15])
            if col1.button(chat_meta["title"], key=f"select_{cid}", use_container_width=True):
                st.session_state.current_chat_id = cid
                st.rerun()
            if col2.button("🗑️", key=f"del_{cid}"):
                db.delete_chat(cid)
                if st.session_state.current_chat_id == cid:
                    st.session_state.current_chat_id = None
                st.rerun()

# -----------------------------
# 🧱 UI — Main area
# -----------------------------
if not st.session_state.current_chat_id:
    all_chats = db.get_all_chats()
    if all_chats:
        st.session_state.current_chat_id = all_chats[0]['id']
    else:
        new_chat()

chat = get_current_chat()

if chat:
    st.title(chat["title"])

    # Show messages
    for msg in chat["messages"]:
        if msg["role"] == "system": 
            continue

        avatar = "🧑‍💻" if msg["role"] == "user" else "🤖"
        with st.chat_message(msg["role"], avatar=avatar):
            content = msg.get("content")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict):
                        if part.get("type") == "text":
                            if msg["role"] == "user":
                                text = part["text"]
                                lines = text.split('\n')

                                if len(lines) > 3:
                                    # Hiển thị 3 dòng đầu và "Xem thêm"
                                    st.text("\n".join(lines[:3]) + "...")  # Hiển thị 3 dòng đầu
                                    with st.expander("Xem thêm", expanded=False):
                                        st.text(text)  # Hiển thị toàn bộ
                                else:
                                    st.text(text)  # Hiển thị nội dung bình thường
                            else:
                                st.markdown(part["text"])  # AI vẫn parse markdown
                        elif part.get("type") == "image_url":
                            st.image(part["image_url"]["url"], caption="Ảnh đã gửi")
            elif isinstance(content, str):
                if msg["role"] == "user":
                    lines = content.split('\n')
                    if len(lines) > 3:
                        # Hiển thị 3 dòng đầu và "Xem thêm"
                        st.text("\n".join(lines[:3]) + "...")  # Hiển thị 3 dòng đầu
                        with st.expander("Xem thêm", expanded=False):
                            st.text(content)  # Hiển thị toàn bộ
                    else:
                        st.text(content)  # Hiển thị nội dung bình thường
                else:
                    st.markdown(content)  # assistant -> markdown

    # Chat input and processing
    prompt = st.chat_input("Nhập tin nhắn...")

    # Sử dụng session_state để lưu trữ thông tin về uploaded_files
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []
    
    uploaded_files = st.file_uploader("Đính kèm ảnh", type=["png", "jpg", "jpeg", "webp"], accept_multiple_files=True)

    # Cập nhật thông tin upload files vào session_state
    if uploaded_files:
        st.session_state.uploaded_files = uploaded_files  # Lưu trữ file uploaded
    else:
        st.session_state.uploaded_files = []  # Reset nếu không có files

    if prompt:
        user_message_content = []

        if prompt:
            user_message_content.append({"type": "text", "text": prompt})

        # Nếu có files uploaded, thêm vào messages
        if st.session_state.uploaded_files:
            for uploaded_file in st.session_state.uploaded_files:
                data_url = image_to_data_url(uploaded_file)
                user_message_content.append({"type": "image_url", "image_url": {"url": data_url}})

            # Xóa trạng thái upload ảnh sau khi đã xử lý
            st.session_state.uploaded_files = []  # Reset trạng thái upload

        chat["messages"].append({"role": "user", "content": user_message_content})

        if client:
            with st.chat_message("assistant", avatar="🤖"):
                with st.spinner("Đang suy nghĩ..."):
                    try:
                        stream = client.chat.completions.create(
                            model=st.session_state.settings["model"],
                            messages=[m for m in chat["messages"] if m["role"] != "system"],
                            stream=True,
                            max_tokens=st.session_state.settings["max_output_tokens"],
                        )
                        response = st.write_stream(stream)
                        chat["messages"].append({"role": "assistant", "content": response})

                        # Cập nhật title nếu là tin nhắn thứ 2 (user -> assistant)
                        if len(chat["messages"]) == 3:
                            try:
                                title_prompt = f"Tóm tắt cuộc trò chuyện sau thành một tiêu đề ngắn gọn (dưới 5 từ) bằng tiếng Việt: User: {prompt[:50]}... Assistant: {response[:50]}..."
                                title_response = client.chat.completions.create(
                                    model="gpt-4o-mini",
                                    messages=[{"role": "user", "content": title_prompt}],
                                    temperature=0.2,
                                )
                                new_title = title_response.choices[0].message.content.strip().strip('"')
                                chat["title"] = new_title
                            except Exception:
                                chat["title"] = prompt[:30]  # Fallback

                        # Lưu lại toàn bộ cuộc trò chuyện vào CSDL
                        db.save_chat(chat["id"], chat["title"], chat["messages"])
                        st.rerun()

                    except Exception as e:
                        st.error(f"Lỗi khi gọi API OpenAI: {e}")
        else:
            st.error("Chưa cấu hình API key.")
else:
    st.info("Hãy chọn một cuộc trò chuyện hoặc tạo mới.")