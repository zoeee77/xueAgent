"""结构化输出引擎：从 LLM 响应中提取和解析 JSON。"""

import json
import re
from typing import Type

from pydantic import BaseModel


def extract_json_from_llm(text: str) -> str:
    """从 LLM 响应中提取 JSON 字符串。

    处理 Markdown 代码块、原始 JSON 等多种格式。

    Args:
        text: LLM 原始响应文本

    Returns:
        提取出的 JSON 字符串
    """
    # 尝试提取 Markdown 代码块中的 JSON
    code_block_patterns = [
        r"```(?:json)?\s*\n([\s\S]*?)\n```",
        r"```\s*([\s\S]*?)\s*```",
    ]
    for pattern in code_block_patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()

    # 尝试找到第一个 { 或 [ 到最后一个 } 或 ]
    for start_char, end_char in [("{", "}"), ("[", "]")]:
        start = text.find(start_char)
        if start != -1:
            end = text.rfind(end_char)
            if end > start:
                return text[start : end + 1]

    # 回退：返回原始文本
    return text.strip()


def _repair_json(text: str) -> str:
    """尝试修复常见的 JSON 格式问题。

    Args:
        text: 可能有格式问题的 JSON 字符串

    Returns:
        修复后的 JSON 字符串
    """
    # 移除尾随逗号（对象和数组中）
    text = re.sub(r",\s*([}\]])", r"\1", text)

    # 修复未加引号的键名（只匹配简单标识符）
    text = re.sub(r"(?<=\{)\s*(\w+)\s*:", r' "\1":', text)
    text = re.sub(r",\s*(\w+)\s*:", r', "\1":', text)

    # 修复单引号为双引号（简单场景）
    # 只在非转义情况下替换
    text = re.sub(r"(?<!\\)'", '"', text)

    return text


def parse_structured(text: str, model: Type[BaseModel]) -> BaseModel:
    """将 LLM 响应文本解析为 Pydantic 模型实例。

    自动提取 JSON 并解析，如果失败则尝试自动修复后重试。

    Args:
        text: LLM 原始响应文本
        model: 目标 Pydantic 模型类

    Returns:
        解析后的模型实例

    Raises:
        ValueError: 解析失败且自动修复也失败
    """
    json_str = extract_json_from_llm(text)

    # 第一次尝试：直接解析
    try:
        data = json.loads(json_str)
        return model.model_validate(data)
    except (json.JSONDecodeError, ValueError):
        pass

    # 第二次尝试：修复 JSON 后解析
    repaired = _repair_json(json_str)
    if repaired != json_str:
        try:
            data = json.loads(repaired)
            return model.model_validate(data)
        except (json.JSONDecodeError, ValueError) as e:
            repair_hint = _repair_json(json_str)
            raise ValueError(
                f"Failed to parse LLM response into {model.__name__}. "
                f"JSON repair hint: {repair_hint[:200]}"
            ) from e

    # 第三次尝试：JSON 有效但模型验证失败
    try:
        data = json.loads(json_str)
        return model.model_validate(data)
    except ValueError as e:
        raise ValueError(
            f"JSON parsed successfully but validation failed for {model.__name__}: {e}"
        ) from e


def build_structured_prompt(
    model: Type[BaseModel], user_input: str, context: dict
) -> str:
    """构建提示词，指导 LLM 输出符合 Pydantic schema 的 JSON。

    Args:
        model: 目标 Pydantic 模型类
        user_input: 用户输入
        context: 上下文信息字典

    Returns:
        完整的提示词字符串
    """
    schema = model.model_json_schema()

    prompt = f"""你是一个结构化数据生成器。请根据以下用户输入和上下文信息，生成符合指定 JSON Schema 的输出。

## 用户输入
{user_input}

## 上下文
{json.dumps(context, ensure_ascii=False, indent=2)}

## 输出格式要求
你必须输出一个合法的 JSON 对象，严格符合以下 JSON Schema：

```json
{json.dumps(schema, ensure_ascii=False, indent=2)}
```

## 重要规则
1. 只输出 JSON，不要包含任何额外的解释或文本
2. 确保所有必填字段都有值
3. 确保数字类型字段是数字，不是字符串
4. 确保枚举字段只能使用指定的值
5. 如果某些信息无法确定，使用 null 或空列表/空字符串

请直接输出 JSON：
"""
    return prompt
