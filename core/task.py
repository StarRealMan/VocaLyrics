from enum import Enum
from typing import Union, Any, Optional, List, Literal
from pydantic import BaseModel, Field
import uuid


class RetrieverInput(BaseModel):
    """Retriever 所需的输入参数模型。"""

    assigned_agent: Literal["Retriever"] = "Retriever"
    request: str = Field(..., description="用户的自然语言检索请求。")

class ParserInput(BaseModel):
    """Parser 所需的输入参数模型。"""

    assigned_agent: Literal["Parser"] = "Parser"
    file_path: str

class AnalystInput(BaseModel):
    """Analyst 所需的输入参数模型。"""

    assigned_agent: Literal["Analyst"] = "Analyst"
    source_key: Optional[str] = None
    source: Optional[str] = None
    retrieved_properties: Optional[List[Literal[
        "defaultName", "year", "producerNames",
        "vsingerNames", "tagNames", "lyrics",
        "ratingScore", "favoritedTimes", "lengthSeconds"
    ]]] = None

class LyricistInput(BaseModel):
    """Lyricist 所需的输入参数模型。"""

    assigned_agent: Literal["Lyricist"] = "Lyricist"
    style: Optional[str] = None
    theme: Optional[str] = None
    midi_key: Optional[str] = None
    source_key: Optional[str] = None
    source: Optional[str] = None

class WriterInput(BaseModel):
    """Writer 所需的输入参数模型。"""

    assigned_agent: Literal["Writer"] = "Writer"
    topic: str
    source_key: Optional[str] = None
    source: Optional[str] = None

class GeneralInput(BaseModel):
    """General Agent 所需的输入参数模型。"""

    assigned_agent: Literal["General"] = "General"
    query: str

class TaskStatus(Enum):
    """
    任务状态枚举
    
    用于追踪任务在生命周期中的当前状态。
    """
    PENDING = "pending"         # 任务已创建，等待执行
    IN_PROGRESS = "in_progress" # 任务正在执行中
    COMPLETED = "completed"     # 任务已成功完成
    FAILED = "failed"           # 任务执行失败

class Task(BaseModel):
    """
    任务数据模型
    
    代表系统需要执行的一个原子操作单元。Planner 会生成一系列 Task，
    Orchestrator 会根据这些 Task 的定义来调度相应的 Agent。
    """
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    """任务的唯一标识符，默认自动生成 UUID。"""
    
    description: str
    """
    任务的自然语言描述。
    例如："查询 Deco*27 的热门歌曲" 或 "分析这首歌的情感"。
    Agent 可以利用这个描述来理解具体的意图。
    """
    
    assigned_agent: str
    """
    指定负责执行该任务的 Agent 名称。
    例如："Retriever", "Analyst", "Lyricist"。
    Orchestrator 会根据这个字段将任务分发给对应的 Agent 实例。
    """
    
    status: TaskStatus = TaskStatus.PENDING
    """当前任务的状态，默认为 PENDING。"""
    
    input_params: Union[RetrieverInput, ParserInput, AnalystInput, LyricistInput, WriterInput, GeneralInput]
    """
    传递给 Agent 的具体结构化参数。
    Planner 可以在这里预先提取出关键信息。
    例如：{"artist": "Deco*27", "limit": 10} 或 {"midi_path": "/path/to/file.mid"}
    """
    
    output_key: Optional[str] = None
    """
    指定任务结果应该存储在共享 Context (Blackboard) 中的哪个 Key 下。
    如果为 None，则结果可能只保存在 Task 对象的 result 字段中，或者由 Agent 决定存储位置。
    指定 Key 可以方便后续的任务通过 Key 来引用这个结果。
    """
    
    result: Any = None
    """
    任务执行后的直接返回结果。
    虽然结果通常也会写入 Context，但在 Task 对象中保留一份副本有助于调试和追踪。
    """

    def mark_in_progress(self):
        """将任务标记为进行中"""
        self.status = TaskStatus.IN_PROGRESS

    def mark_completed(self, result: Any = None):
        """
        将任务标记为完成，并可选地设置结果。
        """
        self.status = TaskStatus.COMPLETED
        if result is not None:
            self.result = result

    def mark_failed(self, error: Any = None):
        """
        将任务标记为失败，并记录错误信息作为结果。
        """
        self.status = TaskStatus.FAILED
        self.result = error
