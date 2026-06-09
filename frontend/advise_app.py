"""Streamlit 高级咨询页面：多智能体 Agent 系统前端入口。"""

import json
import os
import uuid

import httpx
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(
    page_title="张雪峰 AI 志愿填报顾问 - 高级咨询",
    page_icon="🎯",
    layout="wide",
)

st.markdown(
    """
<style>
.stMetric { background-color: #f0f2f6; border-radius: 0.5rem; padding: 1rem; }
.plan-card { padding: 1.5rem; border-radius: 0.5rem; margin: 0.5rem 0; border: 1px solid #e0e0e0; }
.risk-chong { border-left: 4px solid #f44336; background: #ffebee; }
.risk-wen { border-left: 4px solid #ff9800; background: #fff3e0; }
.risk-bao { border-left: 4px solid #4caf50; background: #e8f5e9; }
</style>
""",
    unsafe_allow_html=True,
)

SAMPLE_PROFILES = [
    {"score": 580, "province": "河南", "interests": ["计算机"], "personality": "偏理性", "family_resources": "普通"},
    {"score": 620, "province": "山东", "interests": ["金融", "经济"], "personality": "外向", "family_resources": "中等"},
    {"score": 550, "province": "河北", "interests": ["医学", "护理"], "personality": "细心", "family_resources": "普通"},
    {"score": 500, "province": "黑龙江", "interests": ["师范"], "personality": "稳重", "family_resources": "一般"},
]


def call_advise(score, province, interests, personality, family_resources, session_id):
    """调用后端 /advise 接口。"""
    url = f"{BACKEND_URL}/advise"
    payload = {
        "score": score,
        "province": province,
        "interests": interests,
        "personality": personality,
        "family_resources": family_resources,
        "session_id": session_id,
    }
    try:
        with httpx.Client(timeout=120.0) as client:
            response = client.post(url, json=payload)
            response.raise_for_status()
            return response.json()
    except httpx.ConnectError:
        return {"success": False, "error": "无法连接到后端服务，请确认后端已启动（http://localhost:8000）"}
    except httpx.HTTPStatusError as e:
        return {"success": False, "error": f"后端返回错误: {e.response.status_code} - {e.response.text[:200]}"}
    except Exception as e:
        return {"success": False, "error": f"请求失败: {str(e)}"}


def render_risk_label(risk):
    """渲染风险等级标签。"""
    color_map = {"冲": "🔴", "稳": "🟡", "保": "🟢"}
    emoji = color_map.get(risk, "⚪")
    return f"{emoji} {risk}"


def display_user_profile(profile):
    """展示用户画像。"""
    st.subheader("📋 一、用户画像")
    cols = st.columns(4)
    cols[0].metric("分数", profile.get("score", "N/A"))
    cols[1].metric("省份", profile.get("province", "N/A"))
    cols[2].metric("风险偏好", render_risk_label(profile.get("risk_preference", "")))
    cols[3].metric("兴趣", ", ".join(profile.get("interests", [])) or "未指定")

    if profile.get("constraints"):
        st.info("约束条件: " + ", ".join(profile["constraints"]))
    if profile.get("personality"):
        st.info("性格特点: " + profile["personality"])


def display_data_result(data_result):
    """展示数据检索结果。"""
    st.subheader("📊 二、数据检索结果")
    majors = data_result.get("majors", [])
    industries = data_result.get("industries", [])

    if majors:
        st.markdown("**推荐专业 Top 5：**")
        for m in majors[:5]:
            st.markdown(
                f"- **{m.get('name', 'N/A')}** | 就业率: {m.get('employment_rate', 0)*100:.0f}% | "
                f"均薪: ¥{m.get('avg_salary', 0):,}"
            )

    if industries:
        st.markdown("**相关行业：**")
        for ind in industries[:3]:
            st.markdown(f"- **{ind.get('name', 'N/A')}** | {ind.get('description', '')[:100]}")


def display_multi_role(result):
    """展示多角色分析。"""
    st.subheader("🧠 三、多角色分析")
    opinions = result.get("opinions", [])
    for op in opinions:
        with st.expander(f"👤 {op.get('role_name', 'Unknown')} (评分: {op.get('score', 0)})"):
            st.write(op.get("recommendation", ""))
            st.caption(op.get("reasoning", ""))

    consensus = result.get("consensus", "")
    if consensus:
        st.success(f"🤝 共识: {consensus}")

    conflicts = result.get("conflicts", [])
    if conflicts:
        for c in conflicts:
            st.warning(f"⚡ 分歧: {c}")


def display_plans(plans):
    """展示推荐方案。"""
    st.subheader("📝 四、推荐方案")
    options = plans.get("options", [])
    for opt in options:
        risk = opt.get("risk_level", "")
        risk_class = {"冲": "risk-chong", "稳": "risk-wen", "保": "risk-bao"}.get(risk, "")
        st.markdown(
            f"""
<div class="plan-card {risk_class}">
    <h3>{render_risk_label(risk)} {opt.get('major', 'N/A')}</h3>
    <p><b>推荐院校：</b>{', '.join(opt.get('universities', [])) or '暂无匹配'}</p>
    <p><b>理由：</b>{opt.get('reason', '')}</p>
    <p><b>预期分数：</b>{opt.get('expected_score', 'N/A')}</p>
</div>
""",
            unsafe_allow_html=True,
        )


def display_ranked(ranked):
    """展示排序评分。"""
    st.subheader("🏆 五、排序评分")
    items = ranked.get("ranked_list", [])
    for item in items:
        opt = item.get("option", {})
        score = item.get("total_score", 0)
        breakdown = item.get("breakdown", {})
        rank = item.get("rank", 0)

        cols = st.columns([1, 3, 2])
        cols[0].metric(f"#{rank}", f"{score:.0f}分")
        cols[1].markdown(f"**{render_risk_label(opt.get('risk_level', ''))} {opt.get('major', 'N/A')}**")
        cols[2].markdown(
            f"就业: {breakdown.get('employment', 0):.0f} | 匹配: {breakdown.get('match', 0):.0f} | "
            f"薪资: {breakdown.get('salary', 0):.0f} | 风险: {breakdown.get('risk', 0):.0f}"
        )


def display_devil_advocate(result):
    """展示反对意见。"""
    st.subheader("😈 六、反对意见（Devil's Advocate）")
    objections = result.get("objections", [])
    risks = result.get("risks", [])
    suggestions = result.get("alternative_suggestions", [])

    if objections:
        for o in objections:
            st.error(f"❌ {o}")
    if risks:
        for r in risks:
            st.warning(f"⚠️ {r}")
    if suggestions:
        st.info("💡 替代建议: " + ", ".join(suggestions))


def display_explanation(result):
    """展示解释说明。"""
    st.subheader("💡 七、解释说明")
    st.markdown(f"**为什么推荐：** {result.get('why_recommended', 'N/A')}")
    st.markdown(f"**为什么排第一：** {result.get('why_first', 'N/A')}")
    st.markdown(f"**不推荐的原因：** {result.get('why_not_others', 'N/A')}")

    warnings = result.get("risk_warnings", [])
    if warnings:
        st.warning("风险提示: " + "；".join(warnings))


def main():
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

    st.title("🎯 张雪峰 AI 志愿填报顾问 — 高级咨询")
    st.caption("多智能体协作决策：7+ Agent 并行分析，数据驱动，可解释输出")

    # 侧边栏：用户输入
    with st.sidebar:
        st.header("📝 填写你的信息")
        score = st.number_input("高考分数", min_value=300, max_value=750, value=580, step=1)
        province = st.text_input("所在省份", value="河南")

        st.markdown("**兴趣方向**（可多选）")
        interest_options = [
            "计算机", "电子信息", "金融", "医学", "法学", "师范",
            "工程", "文学", "艺术", "管理", "农业", "建筑",
        ]
        interests = st.multiselect("", options=interest_options, default=["计算机"])

        personality = st.selectbox("性格特点", ["偏理性", "偏感性", "外向", "内向", "细心", "果断"])
        family_resources = st.selectbox("家庭资源", ["普通", "中等", "丰富"])

        # 快速示例
        st.markdown("---")
        st.markdown("**快速示例：**")
        for i, sp in enumerate(SAMPLE_PROFILES):
            label = f"{sp['province']} {sp['score']}分 - {sp['interests'][0]}"
            if st.button(label, key=f"quick_{i}", use_container_width=True):
                st.session_state.quick_score = sp["score"]
                st.session_state.quick_province = sp["province"]
                st.session_state.quick_interests = sp["interests"]
                st.session_state.quick_personality = sp["personality"]
                st.session_state.quick_family = sp["family_resources"]
                st.session_state.run_advise = True

    # 应用快速示例
    if st.session_state.get("run_advise"):
        score = st.session_state.pop("quick_score", score)
        province = st.session_state.pop("quick_province", province)
        interests = st.session_state.pop("quick_interests", interests)
        personality = st.session_state.pop("quick_personality", personality)
        family_resources = st.session_state.pop("quick_family", family_resources)
        st.session_state.pop("run_advise")

    # 提交按钮
    col1, col2 = st.columns([1, 4])
    if col1.button("🚀 开始分析", type="primary"):
        st.session_state.run_advise = True

    if st.session_state.get("run_advise"):
        st.session_state.pop("run_advise", None)

        with st.spinner("🤖 多智能体系统正在分析中，这可能需要 30-60 秒..."):
            result = call_advise(score, province, interests, personality, family_resources, st.session_state.session_id)

        if not result.get("success"):
            st.error(f"❌ 分析失败: {result.get('error', '未知错误')}")
            if result.get("trace_id"):
                st.caption(f"Trace ID: {result['trace_id']}")
        else:
            st.success(f"✅ 分析完成 (Trace ID: {result.get('trace_id', 'N/A')})")
            if result.get("is_fallback"):
                st.warning("⚠️ 当前使用降级方案（LLM 服务不可用时启用）")

            # 展示各阶段结果
            if result.get("user_profile"):
                display_user_profile(result["user_profile"])
                st.divider()

            if result.get("data_result"):
                display_data_result(result["data_result"])
                st.divider()

            if result.get("multi_role_result"):
                display_multi_role(result["multi_role_result"])
                st.divider()

            if result.get("plans"):
                display_plans(result["plans"])
                st.divider()

            if result.get("ranked_plans"):
                display_ranked(result["ranked_plans"])
                st.divider()

            if result.get("devil_advocate"):
                display_devil_advocate(result["devil_advocate"])
                st.divider()

            if result.get("explanation"):
                display_explanation(result["explanation"])

            # JSON 原始数据（可折叠）
            with st.expander("📄 查看原始 JSON 数据"):
                st.json(result)

    else:
        # 默认展示引导信息
        st.info("👈 请在左侧填写你的信息，然后点击「开始分析」")
        st.markdown(
            """
### 🧠 多智能体协作流程

1. **用户画像 Agent** — 分析你的分数、兴趣、性格，评估风险偏好
2. **数据检索 Agent** — 语义匹配 + 结构化过滤，从知识库筛选专业
3. **多角色决策 Agent** — 5 位"专家"并行分析（张雪峰/学术导师/行业专家/HR/家长）
4. **方案生成 Agent** — 生成冲/稳/保三套方案
5. **排序评分 Agent** — 多维度综合评分排序
6. **反对 Agent** — 提出反对意见和风险提示
7. **可解释性 Agent** — 输出可解释的推荐理由
"""
        )


if __name__ == "__main__":
    main()
