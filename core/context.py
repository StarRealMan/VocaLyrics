from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from core.task import Task


class Context(BaseModel):
    """
    上下文管理类 (Blackboard Pattern)
    
    用于在 Agent 之间共享数据、存储对话历史以及维护当前的执行计划。
    所有的 Agent 都可以读取 Context 中的信息，并将执行结果写入 Context。
    """
    
    # 对话历史，存储用户和系统的交互记录
    # 格式通常为 [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    chat_history: List[Dict[str, str]] = Field(default_factory=list)
    
    # 共享内存 (黑板)，用于存放 Agent 的中间产物
    # Key 是数据的标识符 (如 "retrieved_songs"), Value 是具体数据
    shared_memory: Dict[str, Any] = Field(default_factory=dict)
    
    # 共享内存元数据，用于记录每个 Key 的描述信息 (通常是生成该数据的 Task description)
    key_descriptions: Dict[str, str] = Field(default_factory=dict)
    
    # 当前的任务计划
    # 由 Planner 生成的一系列 Task 对象
    plan: List[Task] = Field(default_factory=list)

    def add_user_message(self, content: str):
        """添加用户消息到历史记录"""
        self.chat_history.append({"role": "user", "content": content})

    def add_assistant_message(self, content: str):
        """添加助手消息到历史记录"""
        self.chat_history.append({"role": "assistant", "content": content})

    def set_plan(self, plan: List[Task]):
        """设置新的任务计划"""
        self.plan = plan

    def get_memory(self, key: str, default: Any = None) -> Any:
        """从共享内存中获取数据"""
        return self.shared_memory.get(key, default)

    def set_memory(self, key: str, value: Any):
        """向共享内存写入数据"""
        self.shared_memory[key] = value
    
    def set_key_description(self, key: str, description: str):
        """设置共享内存中某个 Key 的描述信息"""
        self.key_descriptions[key] = description
        
    def clear_plan(self):
        """清空当前计划"""
        self.plan = []

    def get_task_by_id(self, task_id: str) -> Optional[Task]:
        """根据 ID 获取任务对象"""
        for task in self.plan:
            if task.id == task_id:
                return task
        return None
