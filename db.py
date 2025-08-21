import os
import psycopg2
import json
import streamlit as st

# Lấy URL kết nối từ biến môi trường
DATABASE_URL = os.environ.get("DATABASE_URL")

@st.cache_resource
def get_db_connection():
    """Tạo và cache kết nối CSDL."""
    conn = psycopg2.connect(DATABASE_URL)
    return conn

def init_db():
    """Khởi tạo bảng trong CSDL nếu chưa tồn tại."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id VARCHAR(255) PRIMARY KEY,
                    title TEXT NOT NULL,
                    messages JSONB,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                );
            """)
        conn.commit()
    except Exception as e:
        conn.rollback()
        st.error(f"Lỗi khởi tạo CSDL: {e}")
    # Chú ý: không đóng connection ở đây vì nó được cache

def get_all_chats():
    """Lấy danh sách tất cả các cuộc trò chuyện (id và title)."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id, title FROM conversations ORDER BY created_at DESC;")
            chats = cur.fetchall()
            # Chuyển đổi từ list of tuples sang list of dicts
            return [{"id": row[0], "title": row[1]} for row in chats]
    except Exception as e:
        st.error(f"Lỗi khi lấy danh sách chat: {e}")
        return []

def get_chat_by_id(chat_id):
    """Lấy chi tiết một cuộc trò chuyện bằng ID."""
    if not chat_id:
        return None
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id, title, messages FROM conversations WHERE id = %s;", (chat_id,))
            chat = cur.fetchone()
            if chat:
                return {"id": chat[0], "title": chat[1], "messages": chat[2]}
            return None
    except Exception as e:
        st.error(f"Lỗi khi lấy chi tiết chat: {e}")
        return None

def save_chat(chat_id, title, messages):
    """Lưu hoặc cập nhật một cuộc trò chuyện."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            # Dùng ON CONFLICT để vừa INSERT (tạo mới) vừa UPDATE (cập nhật)
            cur.execute("""
                INSERT INTO conversations (id, title, messages)
                VALUES (%s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    title = EXCLUDED.title,
                    messages = EXCLUDED.messages;
            """, (chat_id, title, json.dumps(messages)))
        conn.commit()
    except Exception as e:
        conn.rollback()
        st.error(f"Lỗi khi lưu chat: {e}")

def delete_chat(chat_id):
    """Xóa một cuộc trò chuyện."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM conversations WHERE id = %s;", (chat_id,))
        conn.commit()
    except Exception as e:
        conn.rollback()
        st.error(f"Lỗi khi xóa chat: {e}")