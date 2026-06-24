# 会话持久化升级设计文档

**版本**: 1.0  
**创建时间**: 2026-06-09  
**状态**: 待实现

## 1. 目标

实现聊天记录持久化 + 多会话管理，用户刷新或重新进入系统后可以查看历史对话，并支持切换不同会话。

## 2. 架构约束

- 不允许修改 MemoryManager 的核心逻辑
- 必须新增 PostgreSQLSessionStore 类，实现与现有 SessionStore 完全一致的接口
- 数据库操作必须封装在 SessionStore 内，不允许出现在 main.py
- 保持现有系统结构稳定（最小侵入）

## 3. 数据库设计

### 3.1 sessions 表

```sql
CREATE TABLE IF NOT EXISTS chat_sessions (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(100) NOT NULL UNIQUE,
    user_id VARCHAR(100) NOT NULL,
    title VARCHAR(200),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_sessions_user ON chat_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_id ON chat_sessions(session_id);
```

### 3.2 messages 表

```sql
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
```

## 4. 代码设计

### 4.1 PostgreSQLSessionStore 类

**文件位置**: `backend/session/postgres_session_store.py`

**接口设计**:

```python
class PostgreSQLSessionStore:
    # 与现有 SessionStore 兼容的接口
    def get(self, session_id: str) -> list[dict]
    def set(self, session_id: str, value: list[dict])
    def append(self, session_id: str, item: dict)
    
    # 新增接口（多会话管理）
    async def create_session(self, user_id: str, session_id: str, title: str = None) -> bool
    async def list_sessions(self, user_id: str, limit: int = 20) -> list[dict]
    async def delete_session(self, session_id: str) -> bool
    
    # 预留：方案 C 缓存扩展点
    def _cache_key(self, session_id: str) -> str
    def _invalidate_cache(self, session_id: str)
```

### 4.2 main.py 修改

**修改范围**:
1. 导入 PostgreSQLSessionStore
2. 替换全局 `session_store` 实例
3. /chat 接口：从 session_store 读取/写入改为异步调用
4. /chat/sync 接口：同上

**修改量**: ~30 行代码变更

## 5. 数据流

```
用户请求 → main.py → PostgreSQLSessionStore.get(session_id) → 查询 DB → 返回历史
LLM 回复 → main.py → PostgreSQLSessionStore.append(session_id, msg) → 写入 DB
```

## 6. 方案 C 预留扩展

在 PostgreSQLSessionStore 中预留以下方法签名（不实现）：

```python
def _get_from_cache(self, session_id: str) -> Optional[list[dict]]
def _set_to_cache(self, session_id: str, messages: list[dict])
def _build_cache_layer(self) -> None  # 未来可接入 Redis/内存缓存
```

## 7. 接口调用示例

### 创建会话
```http
POST /session/create
{
  "user_id": "user_123",
  "title": "河南580分志愿咨询"
}
```

### 获取会话列表
```http
GET /session/list?user_id=user_123
```

### 聊天（指定会话）
```http
POST /chat
{
  "session_id": "sess_abc123",
  "message": "计算机专业怎么样？"
}
```
