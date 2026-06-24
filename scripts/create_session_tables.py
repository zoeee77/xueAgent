"""会话表创建脚本：创建 chat_sessions 和 chat_messages 表。"""

import logging
import os

import psycopg2

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "5432")),
    "dbname": os.getenv("DB_NAME", "xueAgent"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "root"),
}

CREATE_SQL = """
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
    role VARCHAR(20) NOT NULL,
    content TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    FOREIGN KEY (session_id) REFERENCES chat_sessions(session_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_messages_session ON chat_messages(session_id);
CREATE INDEX IF NOT EXISTS idx_messages_session_order ON chat_messages(session_id, id ASC);
"""

def main():
    logger.info("Connecting to PostgreSQL...")
    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = True

    try:
        logger.info("Creating session tables...")
        with conn.cursor() as cur:
            cur.execute(CREATE_SQL)

        logger.info("Verifying tables...")
        with conn.cursor() as cur:
            cur.execute("""
                SELECT table_name FROM information_schema.tables
                WHERE table_schema = 'public'
                AND table_name IN ('chat_sessions', 'chat_messages')
                ORDER BY table_name
            """)
            tables = [row[0] for row in cur.fetchall()]

        if len(tables) == 2:
            logger.info("✅ Session tables created successfully: %s", tables)
        else:
            logger.warning("⚠️  Some tables may be missing. Found: %s", tables)

    except Exception as e:
        logger.error("❌ Failed to create session tables: %s", e)
        raise
    finally:
        conn.close()

if __name__ == "__main__":
    main()
