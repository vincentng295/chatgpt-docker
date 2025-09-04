import os
import json
import base64
import time
from typing import List, Dict, Any
import gittojson

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
        st.session_state.current_chat_id = None
    if "settings" not in st.session_state:
        st.session_state.settings = {
            "model": "gpt-4o-mini",
            "max_output_tokens": 1024,
            "render_mode" : "Markdown",
            "system_prompt": "Bạn là Huệ — một trợ lý hữu ích, nói tiếng Việt, súc tích và thân thiện.",
        }
    if "models_list" not in st.session_state:
        st.session_state.models_list = []
    # 👇 State cho chế độ chỉnh sửa
    if "editing_index" not in st.session_state:
        st.session_state.editing_index = None
    if "edit_text" not in st.session_state:
        st.session_state.edit_text = ""
    if "edit_github_url" not in st.session_state:
        st.session_state.edit_github_url = ""
    if "edit_image_urls" not in st.session_state:
        st.session_state.edit_image_urls = []

def decompose_user_content(content):
    """Tách nội dung user thành (text, [image_urls], repo_url)."""
    texts, image_urls, repo_url = [], [], None
    if isinstance(content, list):
        for c in content:
            if isinstance(c, dict):
                if c.get("type") == "text":
                    texts.append(c.get("text", ""))
                elif c.get("type") == "image_url":
                    url = c.get("image_url", {}).get("url")
                    if url:
                        image_urls.append(url)
                elif c.get("type") == "repojson":
                    repo_url = c.get("url")
    elif isinstance(content, str):
        texts.append(content)
    return "\n".join([t for t in texts if t is not None]), image_urls, repo_url

def build_user_content(text, new_image_files, keep_old_image_urls, github_url):
    """Gom nội dung user thành schema messages của OpenAI."""
    content = []
    if text and text.strip():
        content.append({"type": "text", "text": text})

    # Giữ ảnh cũ (nếu có)
    for old_url in keep_old_image_urls or []:
        content.append({"type": "image_url", "image_url": {"url": old_url}})

    # Thêm ảnh mới (nếu có)
    for f in (new_image_files or []):
        content.append({"type": "image_url", "image_url": {"url": image_to_data_url(f)}})

    # Thêm repo (nếu có)
    if github_url:
        repo_json_file = gittojson.repo_to_json(github_url)
        repo_json = json.load(repo_json_file)
        repo_json_str = json.dumps(repo_json, ensure_ascii=False)
        content.append({
            "type": "repojson",
            "url": github_url,
            "text": repo_json_str
        })
    return content

def prepare_messages_for_api(messages):
    """Chuyển đổi toàn bộ chat['messages'] sang dạng an toàn để gửi API."""
    messages_for_api = []
    for msg in messages:
        safe_content = []

        if isinstance(msg["content"], list):
            for c in msg["content"]:
                if isinstance(c, dict):
                    if c.get("type") in ("text", "image_url"):
                        if c.get("type") == "image_url" and msg["role"] == "assistant":
                            # Ảnh của model -> chuyển thành user thông báo
                            messages_for_api.append({
                                "role": "user",
                                "content": [{"type": "text", "text": "Ảnh model đã gửi:"}, c]
                            })
                        else:
                            safe_content.append(c)
                    elif c.get("type") == "repojson":
                        safe_content.append({
                            "type": "text",
                            "text": f"Repo JSON import từ {c['url']}:\n{c['text']}"
                        })
        elif isinstance(msg["content"], str):
            safe_content.append({"type": "text", "text": msg["content"]})

        if safe_content:
            messages_for_api.append({"role": msg["role"], "content": safe_content})

    return messages_for_api

def new_chat():
    chat_id = str(int(time.time() * 1000))
    title = "Cuộc trò chuyện mới"
    messages = [{"role": "system", "content": st.session_state.settings["system_prompt"]}]
    db.save_chat(chat_id, title, messages, st.session_state.settings)
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

    if not st.session_state.models_list:
        try:
            # 🔄 Lấy danh sách model từ OpenAI
            models = client.models.list()
            st.session_state.models_list = sorted([m.id for m in models.data])
        except Exception as e:
            st.error(f"Không lấy được danh sách model: {e}")
            st.session_state.models_list = []

    # Nếu không có model nào thì fallback
    if not st.session_state.models_list:
        st.session_state.models_list = ["gpt-4o-mini"]

    # Hiển thị dropdown chọn model
    current_model = st.session_state.settings.get("model", st.session_state.models_list[0])
    st.session_state.settings["model"] = st.selectbox(
        "Model",
        st.session_state.models_list,
        index=st.session_state.models_list.index(current_model) if current_model in st.session_state.models_list else 0
    )
    st.session_state.settings["max_output_tokens"] = st.slider("Giới hạn token trả lời", 64, 8192, 8192)

    # 🔀 Thêm tùy chọn hiển thị output
    st.session_state.settings["render_mode"] = st.radio(
        "Hiển thị câu trả lời AI dưới dạng:",
        ["Markdown", "Text only"],
        index=0
    )

    with st.expander("🎛️ System prompt"):
        st.session_state.settings["system_prompt"] = st.text_area("Nội dung", value=st.session_state.settings["system_prompt"], height=120)

    # Thêm nút lưu thiết lập
    if st.button("💾 Lưu thiết lập"):
        chat = get_current_chat()
        if chat:
            db.save_chat(chat["id"], chat["title"], chat["messages"], st.session_state.settings)
            st.success("Đã lưu thiết lập thành công ✅")
        else:
            st.warning("Chưa có cuộc trò chuyện nào để lưu.")

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
    if not all_chats:
        new_chat()

chat = get_current_chat()

if chat:
    if chat.get("settings"):
        st.session_state.settings.update(chat["settings"])
    st.title(chat["title"])

    # Show messages
    for i, msg in enumerate(chat["messages"]):
        if msg["role"] == "system": 
            continue

        avatar = "🧑‍💻" if msg["role"] == "user" else "🤖"
        with st.chat_message(msg["role"], avatar=avatar):
            content = msg.get("content")
            if isinstance(content, list):
                for j, part in enumerate(content):
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
                                if st.session_state.settings.get("render_mode") == "Text only":
                                    st.text(part["text"])
                                else:
                                    st.markdown(part["text"])

                        elif part.get("type") == "image_url":
                            st.image(part["image_url"]["url"], caption="Ảnh đã gửi")

                        elif part.get("type") == "repojson":
                            st.write(f"Github repo: {part.get('url')}")
                            st.download_button(
                                "📥 repo.json",
                                data=part.get("text").encode("utf-8"),
                                file_name="repo.json",
                                mime="application/json",
                                key=f"repojson_{i}_{j}"
                            )

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
                    if st.session_state.settings.get("render_mode") == "Text only":
                        st.text(content)   # hiển thị dạng raw text
                    else:
                        st.markdown(content)  # hiển thị dạng markdown

        if msg["role"] == "user":
            # Hàng nút thao tác
            # Nút 3 chấm (menu)
            with st.expander("⋮", expanded=False):
                if st.button("✏️ Sửa", key=f"edit_btn_{i}"):
                    st.session_state.editing_index = i
                    text0, img_urls0, repo0 = decompose_user_content(msg["content"])
                    st.session_state.edit_text = text0 or ""
                    st.session_state.edit_github_url = repo0 or ""
                    st.session_state.edit_image_urls = img_urls0 or []
                    st.rerun()

                if st.button("🗑️ Xóa từ đây", key=f"del_from_btn_{i}"):
                    chat["messages"] = chat["messages"][:i]
                    db.save_chat(chat["id"], chat["title"], chat["messages"], st.session_state.settings)
                    st.success("Đã xóa. Cuộc trò chuyện được bắt đầu lại từ mốc này.")
                    st.rerun()

            # Nếu đang chỉnh sửa đúng message này -> hiển thị form
            if st.session_state.editing_index == i:
                st.info("Đang chỉnh sửa tin nhắn này. Lưu để khởi động lại cuộc trò chuyện từ đây.")
                new_text = st.text_area(
                    "Nội dung văn bản",
                    value=st.session_state.edit_text,
                    height=180,
                    key=f"edit_text_{i}"
                )

                keep_old = st.checkbox(
                    "Giữ các ảnh đã gửi trước đó",
                    value=True,
                    key=f"keep_old_{i}"
                )

                new_images = st.file_uploader(
                    "Thêm ảnh mới (tuỳ chọn)",
                    type=["png", "jpg", "jpeg", "webp"],
                    accept_multiple_files=True,
                    key=f"edit_upload_{i}"
                )

                new_github = st.text_input(
                    "GitHub repo (tuỳ chọn, để trống nếu muốn bỏ)",
                    value=st.session_state.edit_github_url,
                    key=f"edit_github_{i}"
                )

                col_save, col_cancel = st.columns(2)
                with col_save:
                    if st.button("💾 Lưu & Restart", key=f"save_edit_{i}"):
                        try:
                            # Xây content mới
                            content_new = build_user_content(
                                text=new_text,
                                new_image_files=new_images,
                                keep_old_image_urls=(st.session_state.edit_image_urls if keep_old else []),
                                github_url=new_github.strip(),
                            )

                            # Cập nhật message i
                            chat["messages"][i]["content"] = content_new

                            # Cắt bỏ mọi message sau i (restart từ đây)
                            chat["messages"] = chat["messages"][:i+1]

                            # Gọi lại AI để trả lời từ lịch sử đã cắt
                            if client:
                                with st.chat_message("assistant", avatar="🤖"):
                                    with st.spinner("Đang tạo phản hồi mới..."):
                                        msgs_api = prepare_messages_for_api(chat["messages"])

                                        # Ảnh: vẫn hỗ trợ image model nếu bạn chọn
                                        if st.session_state.settings["model"].startswith(("dall-e-", "gpt-image-")):
                                            image = client.images.generate(
                                                model=st.session_state.settings["model"],
                                                prompt=new_text or "Hãy tạo một hình ảnh minh hoạ.",
                                                size="1024x1024"
                                            )
                                            img_url = image.data[0].url
                                            st.image(img_url, caption=f'Ảnh AI ({st.session_state.settings["model"]})')
                                            chat["messages"].append({
                                                "role": "assistant",
                                                "content": [{"type": "image_url", "image_url": {"url": img_url}}]
                                            })
                                        else:
                                            stream = client.chat.completions.create(
                                                model=st.session_state.settings["model"],
                                                messages=msgs_api,
                                                stream=True,
                                                max_completion_tokens=st.session_state.settings["max_output_tokens"],
                                            )
                                            response = st.write_stream(stream)
                                            chat["messages"].append({"role": "assistant", "content": response})

                                # Cập nhật title nếu đây là lượt tương tác đầu tiên
                                if len(chat["messages"]) == 3:
                                    try:
                                        title_prompt = f"Tóm tắt cuộc trò chuyện sau thành một tiêu đề ngắn gọn (dưới 5 từ) bằng tiếng Việt: User: {(new_text or '')[:50]}... Assistant: {str(chat['messages'][-1].get('content',''))[:50]}..."
                                        title_response = client.chat.completions.create(
                                            model="gpt-4o-mini",
                                            messages=[{"role": "user", "content": title_prompt}],
                                            temperature=0.2,
                                        )
                                        new_title = title_response.choices[0].message.content.strip().strip('"')
                                        chat["title"] = new_title or chat["title"]
                                    except Exception:
                                        pass

                            # Lưu và thoát chế độ chỉnh sửa
                            db.save_chat(chat["id"], chat["title"], chat["messages"], st.session_state.settings)
                            st.session_state.editing_index = None
                            st.session_state.edit_text = ""
                            st.session_state.edit_github_url = ""
                            st.session_state.edit_image_urls = []
                            st.success("Đã lưu chỉnh sửa và khởi động lại từ mốc này.")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Lỗi khi lưu chỉnh sửa: {e}")

                with col_cancel:
                    if st.button("Hủy", key=f"cancel_edit_{i}"):
                        st.session_state.editing_index = None
                        st.rerun()


    # Chat input and processing
    prompt = st.chat_input("Nhập tin nhắn...")

    # Sử dụng session_state để lưu trữ thông tin về uploaded_files
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []
    
    uploaded_files = st.file_uploader("Đính kèm ảnh", type=["png", "jpg", "jpeg", "webp"], accept_multiple_files=True)

    # Cập nhật thông tin upload files vào session_state
    if uploaded_files:
        st.session_state.uploaded_files = uploaded_files
    else:
        st.session_state.uploaded_files = []

    st.markdown("**Hoặc kết nối GitHub repo:**")

    if "github_url" not in st.session_state:
        st.session_state.github_url = ""

    col1, col2 = st.columns([0.85, 0.15])
    with col1:
        github_url_input = st.text_input(
            "URL GitHub repo",
            value=st.session_state.github_url,
            placeholder="https://github.com/owner/repo",
            label_visibility="collapsed"
        )
    with col2:
        if st.button("❌", help="Xóa URL GitHub"):
            st.session_state.github_url = ""
            st.rerun()

    # Khi user nhập repo mới
    if github_url_input and github_url_input != st.session_state.github_url:
        st.session_state.github_url = github_url_input
        try:
            with st.spinner("⏳ Đang tải repo..."):
                repo_json_file = gittojson.repo_to_json(st.session_state.github_url)
                repo_json = json.load(repo_json_file)
                repo_json_str = json.dumps(repo_json, ensure_ascii=False)

                chat["messages"].append({
                    "role": "user",
                    "content": [
                        {
                            "type": "repojson",
                            "url": st.session_state.github_url,
                            "text": repo_json_str
                        }
                    ]
                })
                db.save_chat(chat["id"], chat["title"], chat["messages"], st.session_state.settings)
                st.success("✅ Repo đã được thêm vào cuộc trò chuyện")
                st.rerun()
        except Exception as e:
            st.error(f"Lỗi tải repo: {e}")


    if prompt:
        user_message_content = []

        user_message_content.append({"type": "text", "text": prompt})

        # Nếu có files uploaded, thêm vào messages
        if st.session_state.uploaded_files:
            for uploaded_file in st.session_state.uploaded_files:
                data_url = image_to_data_url(uploaded_file)
                user_message_content.append({"type": "image_url", "image_url": {"url": data_url}})

        # Nếu có GitHub repo, thêm vào messages
        if st.session_state.github_url:
            try:
                repo_json_file = gittojson.repo_to_json(st.session_state.github_url)
                repo_json = json.load(repo_json_file)
                repo_json_str = json.dumps(repo_json, ensure_ascii=False)

                chat["messages"].append({
                    "role": "user",
                    "content": [
                        {
                            "type": "repojson",
                            "url": st.session_state.github_url,
                            "text": repo_json_str
                        }
                    ]
                })
            except Exception as e:
                st.error(f"Lỗi tải repo: {e}")

        # Append message của user
        chat["messages"].append({"role": "user", "content": user_message_content})

        # Reset trạng thái sau khi gửi prompt
        st.session_state.uploaded_files = []
        st.session_state.github_url = ""

        if client:
            with st.chat_message("assistant", avatar="🤖"):
                with st.spinner("Đang suy nghĩ..."):
                    try:
                        # Chuẩn bị messages, loại bỏ các phần không cần thiết
                        messages_for_api = prepare_messages_for_api(chat["messages"])

                        if st.session_state.settings["model"].startswith("dall-e-") or st.session_state.settings["model"].startswith("gpt-image-"):
                            # Gọi image API
                            image = client.images.generate(
                                model=st.session_state.settings["model"],
                                prompt=prompt,
                                size="1024x1024"
                            )
                            img_url = image.data[0].url
                            with st.chat_message("assistant", avatar="🤖"):
                                st.image(img_url, caption=f'Ảnh AI ({st.session_state.settings["model"]})')
                            chat["messages"].append({
                                "role": "assistant",
                                "content": [{"type": "image_url", "image_url": {"url": img_url}}]
                            })
                            db.save_chat(chat["id"], chat["title"], chat["messages"])
                            st.rerun()
                        else:
                            stream = client.chat.completions.create(
                                model=st.session_state.settings["model"],
                                messages=messages_for_api,
                                stream=True,
                                max_completion_tokens=st.session_state.settings["max_output_tokens"],
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
                        db.save_chat(chat["id"], chat["title"], chat["messages"], st.session_state.settings)
                        st.rerun()

                    except Exception as e:
                        st.error(f"Lỗi khi gọi API OpenAI: {e}")
        else:
            st.error("Chưa cấu hình API key.")
else:
    st.info("Hãy chọn một cuộc trò chuyện hoặc tạo mới.")