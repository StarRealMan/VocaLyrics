import json
from abc import ABC, abstractmethod
from typing import Any, List
import logging

from core.context import Context
from core.task import Task


class Agent(ABC):
    """
    Agent 基类
    
    所有具体的 Agent (如 Planner, Retriever, Analyst) 都必须继承此类。
    """
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.logger = logging.getLogger(f"Agent.{name}")

    @abstractmethod
    def run(self, context: Context, task: Task) -> Any:
        """
        执行任务的核心方法。
        
        Args:
            context: 当前的全局上下文，包含历史记录和共享内存。
            task: 当前需要执行的具体任务对象。
            
        Returns:
            任务的执行结果。
        """
        pass

    def _get_param(self, task: Task, key: str, default: Any = None) -> Any:
        """
        辅助方法：从 Task 的 input_params 中安全地获取参数。
        """
        return task.input_params.model_dump().get(key, default)

    def _save_to_memory(self, context: Context, task: Task, value: Any):
        """
        辅助方法：如果 Task 指定了 output_key，则将结果保存到 Context 的共享内存中。
        """
        if task.output_key:
            context.set_memory(task.output_key, value)
            context.set_key_description(task.output_key, task.description)

    def _format_memory_content(self, context: Context, keys: List[str]) -> str:
        """
        智能格式化共享内存中的数据，使其适合放入 Prompt。
        会自动识别数据类型（如检索结果列表）并进行精简。
        """
        formatted_parts = []
        
        # 是否可以类别识别？Output类型

        for key in keys:
            data = context.get_memory(key)
            if not data:
                self.logger.warning(f"Key '{key}' not found in memory, skipping.")
                continue

            desc = context.key_descriptions.get(key, "Reference Data")
            header = f"--- Data Source: {key} ({desc}) ---"
            content_str = ""

            # 策略 1: 处理列表 (通常是 Retriever 的结果)
            if isinstance(data, list):
                # 检查是否是检索结果 (包含 payload 字段)
                if len(data) > 0 and isinstance(data[0], dict) and "payload" in data[0]:
                    # 精简检索结果，只保留核心字段
                    items = []
                    for idx, item in enumerate(data):
                        payload = item.get("payload", {})
                        score = item.get("score", 0)
                        # 提取歌名和歌词预览
                        name = payload.get("name", "Unknown")
                        lyrics = payload.get("lyrics_preview") or payload.get("lyrics") or ""
                        # 截断过长的歌词
                        if len(lyrics) > 200: lyrics = lyrics[:200] + "..."
                        
                        items.append(f"[{idx+1}] Title: {name} (Score: {score:.2f})\n    Content: {lyrics}")
                    content_str = "\n".join(items)
                else:
                    # 普通列表
                    content_str = json.dumps(data, indent=2, ensure_ascii=False)
            
            # 策略 2: 处理字典 (通常是 Analyst 或 Parser 的结果)
            elif isinstance(data, dict):
                content_str = json.dumps(data, indent=2, ensure_ascii=False)
            
            # 策略 3: 其他 (字符串等)
            else:
                content_str = str(data)

            formatted_parts.append(f"{header}\n{content_str}\n")

        return "\n".join(formatted_parts)
