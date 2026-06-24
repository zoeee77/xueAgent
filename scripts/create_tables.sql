-- xueAgent PostgreSQL 表结构
-- 版本: 1.0
-- 创建时间: 2026-06-08

-- ============================================================
-- 1. 专业表
-- ============================================================
CREATE TABLE IF NOT EXISTS majors (
    name VARCHAR(100) PRIMARY KEY,
    description TEXT,
    avg_salary INTEGER,
    employment_rate DECIMAL(3,2),
    resource_threshold VARCHAR(20),
    personality_fit JSONB DEFAULT '[]',
    keywords JSONB DEFAULT '[]',
    courses JSONB DEFAULT '[]',
    industries JSONB DEFAULT '[]',
    career_paths JSONB DEFAULT '[]'
);

CREATE INDEX IF NOT EXISTS idx_majors_salary ON majors(avg_salary);
CREATE INDEX IF NOT EXISTS idx_majors_employment ON majors(employment_rate);

-- ============================================================
-- 2. 院校信息表
-- ============================================================
CREATE TABLE IF NOT EXISTS universities (
    name VARCHAR(200) PRIMARY KEY,
    province VARCHAR(50),
    tier VARCHAR(20),
    min_score_2025 INTEGER,
    avg_score_2025 INTEGER,
    rank_range VARCHAR(50),
    description TEXT
);

CREATE INDEX IF NOT EXISTS idx_universities_province ON universities(province);
CREATE INDEX IF NOT EXISTS idx_universities_tier ON universities(tier);
CREATE INDEX IF NOT EXISTS idx_universities_min_score ON universities(min_score_2025);

-- ============================================================
-- 3. 行业信息表
-- ============================================================
CREATE TABLE IF NOT EXISTS industries (
    name VARCHAR(100) PRIMARY KEY,
    entry_barrier VARCHAR(20),
    family_resource_dependent BOOLEAN DEFAULT FALSE,
    salary_range JSONB,
    graduate_distribution JSONB,
    top_employers JSONB DEFAULT '[]',
    description TEXT
);

CREATE INDEX IF NOT EXISTS idx_industries_barrier ON industries(entry_barrier);

-- ============================================================
-- 4. 分数线表（批次线 + 院校投档线）
-- ============================================================
CREATE TABLE IF NOT EXISTS batch_scores (
    id SERIAL PRIMARY KEY,
    province VARCHAR(50),
    year INTEGER,
    category VARCHAR(20),
    batch VARCHAR(20),
    subject_type VARCHAR(20),
    score_line INTEGER,
    UNIQUE(province, year, category, batch, subject_type)
);

CREATE TABLE IF NOT EXISTS school_scores (
    id SERIAL PRIMARY KEY,
    province VARCHAR(50),
    year INTEGER,
    subject_type VARCHAR(20),
    school_name VARCHAR(200),
    min_score INTEGER,
    avg_score INTEGER,
    min_rank INTEGER,
    admission_count INTEGER,
    provincial_line INTEGER,
    line_diff INTEGER
);

CREATE INDEX IF NOT EXISTS idx_batch_scores_province ON batch_scores(province, year);
CREATE INDEX IF NOT EXISTS idx_school_scores_school ON school_scores(school_name);
CREATE INDEX IF NOT EXISTS idx_school_scores_province ON school_scores(province, year);

-- ============================================================
-- 5. 院校基础信息表（完整列表）
-- ============================================================
CREATE TABLE IF NOT EXISTS university_list (
    code VARCHAR(20) PRIMARY KEY,
    school_name VARCHAR(200) UNIQUE NOT NULL,
    province VARCHAR(50),
    competent_department VARCHAR(100),
    location VARCHAR(100),
    level VARCHAR(20),
    is_private BOOLEAN DEFAULT FALSE,
    tier VARCHAR(20)
);

CREATE INDEX IF NOT EXISTS idx_university_list_name ON university_list(school_name);
CREATE INDEX IF NOT EXISTS idx_university_list_tier ON university_list(tier);

-- ============================================================
-- 6. 学科评估表
-- ============================================================
CREATE TABLE IF NOT EXISTS subject_reviews (
    id VARCHAR(100) PRIMARY KEY,
    round INTEGER,
    year INTEGER,
    category_name VARCHAR(100),
    subject_code VARCHAR(20),
    subject_name VARCHAR(100),
    school_code VARCHAR(20),
    school_name VARCHAR(200),
    rank INTEGER,
    grade VARCHAR(10)
);

CREATE INDEX IF NOT EXISTS idx_subject_reviews_school ON subject_reviews(school_name);
CREATE INDEX IF NOT EXISTS idx_subject_reviews_subject ON subject_reviews(subject_name);
CREATE INDEX IF NOT EXISTS idx_subject_reviews_grade ON subject_reviews(grade);

-- ============================================================
-- 7. 决策规则表
-- ============================================================
CREATE TABLE IF NOT EXISTS decision_rules (
    key VARCHAR(100) PRIMARY KEY,
    value JSONB NOT NULL
);

-- ============================================================
-- 8. 用户记忆表（从 SQLite 迁移）
-- ============================================================
CREATE TABLE IF NOT EXISTS user_memory (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(100),
    topic VARCHAR(200),
    content TEXT,
    timestamp BIGINT,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_user_memory_user ON user_memory(user_id);
CREATE INDEX IF NOT EXISTS idx_user_memory_timestamp ON user_memory(timestamp);

-- ============================================================
-- 验证查询
-- ============================================================
SELECT 'tables_created' as status, COUNT(*) as table_count 
FROM information_schema.tables 
WHERE table_schema = 'public' 
  AND table_name IN ('majors', 'universities', 'industries', 
                     'batch_scores', 'school_scores', 'university_list',
                     'subject_reviews', 'decision_rules', 'user_memory');
