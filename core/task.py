"""Task management primitives for the multi-agent orchestrator."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional


class TaskStatus(Enum):
	"""Lifecycle states for a task or sub-task."""

	PENDING = auto()
	RUNNING = auto()
	SUCCEEDED = auto()
	FAILED = auto()


@dataclass
class AgentArtifact:
	"""Structured data that agents can attach to their results."""

	kind: str
	payload: Any


@dataclass
class AgentResult:
	"""Normalized output returned by every agent."""

	content: str
	reasoning: Optional[str] = None
	citations: List[Dict[str, Any]] = field(default_factory=list)
	artifacts: List[AgentArtifact] = field(default_factory=list)
	error: Optional[str] = None

	@property
	def ok(self) -> bool:
		return self.error is None


@dataclass
class TaskStep:
	"""Single step inside a planner-produced execution plan."""

	step_id: str
	agent: str
	goal: str
	inputs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Task:
	"""Represents a top-level user task or an agent sub-task."""

	task_id: str
	description: str
	agent: Optional[str] = None
	parent_id: Optional[str] = None
	status: TaskStatus = TaskStatus.PENDING
	created_at: float = field(default_factory=time.time)
	updated_at: float = field(default_factory=time.time)
	metadata: Dict[str, Any] = field(default_factory=dict)
	steps: List[TaskStep] = field(default_factory=list)
	result: Optional[AgentResult] = None
	logs: List[str] = field(default_factory=list)

	def mark(self, status: TaskStatus) -> None:
		self.status = status
		self.updated_at = time.time()

	def append_log(self, message: str) -> None:
		timestamp = time.strftime("%H:%M:%S", time.localtime())
		self.logs.append(f"[{timestamp}] {message}")
		self.updated_at = time.time()


class TaskManager:
	"""Lightweight in-memory task tracker."""

	def __init__(self) -> None:
		self._tasks: Dict[str, Task] = {}

	# ------------------------------------------------------------------
	# Creation helpers
	# ------------------------------------------------------------------
	def create_task(
		self,
		description: str,
		agent: Optional[str] = None,
		parent_id: Optional[str] = None,
		metadata: Optional[Dict[str, Any]] = None,
		steps: Optional[List[TaskStep]] = None,
	) -> Task:
		task_id = str(uuid.uuid4())
		task = Task(
			task_id=task_id,
			description=description,
			agent=agent,
			parent_id=parent_id,
			metadata=metadata or {},
			steps=steps or [],
		)
		self._tasks[task_id] = task
		return task

	# ------------------------------------------------------------------
	# Lookup helpers
	# ------------------------------------------------------------------
	def get(self, task_id: str) -> Optional[Task]:
		return self._tasks.get(task_id)

	def list_children(self, parent_id: str) -> List[Task]:
		return [task for task in self._tasks.values() if task.parent_id == parent_id]

	# ------------------------------------------------------------------
	# Mutations
	# ------------------------------------------------------------------
	def start(self, task_id: str) -> None:
		task = self._require(task_id)
		task.mark(TaskStatus.RUNNING)

	def succeed(self, task_id: str, result: Optional[AgentResult] = None) -> None:
		task = self._require(task_id)
		task.result = result
		task.mark(TaskStatus.SUCCEEDED)

	def fail(self, task_id: str, error: str) -> None:
		task = self._require(task_id)
		task.append_log(error)
		task.result = (task.result or AgentResult(content="", error=error))
		task.mark(TaskStatus.FAILED)

	def set_steps(self, task_id: str, steps: List[TaskStep]) -> None:
		task = self._require(task_id)
		task.steps = steps
		task.updated_at = time.time()

	def log(self, task_id: str, message: str) -> None:
		self._require(task_id).append_log(message)

	# ------------------------------------------------------------------
	def _require(self, task_id: str) -> Task:
		task = self._tasks.get(task_id)
		if not task:
			raise KeyError(f"Task {task_id} not found")
		return task

