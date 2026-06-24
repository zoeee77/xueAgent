-- ============================================================
-- 9. 会话持久化表（会话管理 + 聊天记录）
-- 版本: 1.0
-- 创建时间: 2026-06-09
-- ============================================================

-- 会话表
CREATE TABLE IF NOT EXISTS chat_sessions (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(100) NOT NULL UNIQUE,
    user_id VARCHAR(100) NOT NULL,
    title VARCHAR(200) DEFAULT '新对话',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_sessions_user ON chat_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_id ON chat_sessions(session_id);
CREATE INDEX IF NOT EXISTS idx_sessions_updated ON chat_sessions(user_id, updated_at DESC);

-- 消息表
CREATE TABLE IF NOT EXISTS chat_messages (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(100) NOT NULL,
    role VARCHAR(20) NOT NULL,          -- 'user' | 'assistant'
    content TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    FOREIGN KEY (session_id) REFERENCES chat_sessions(session_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_messages_session ON chat_messages(session_id);
CREATE INDEX IF NOT EXISTS idx_messages_session_order ON chat_messages(session_id, id ASC);

-- ============================================================
-- 验证查询
-- ============================================================
SELECT 'session_tables_created' as status, COUNT(*) as table_count 
FROM information_schema.tables 
WHERE table_schema = 'public' 
  AND table_name IN ('chat_sessions', 'chat_messages');
