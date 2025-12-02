from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
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
        return task.input_params.get(key, default)

    def _save_to_memory(self, context: Context, task: Task, value: Any):
        """
        辅助方法：如果 Task 指定了 output_key，则将结果保存到 Context 的共享内存中。
        """
        if task.output_key:
            context.set_memory(task.output_key, value)
