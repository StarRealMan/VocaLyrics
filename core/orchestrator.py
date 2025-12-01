"""Central loop that coordinates multi-agent execution."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Protocol, Tuple

from .context import ConversationContext
from .task import AgentResult, Task, TaskManager, TaskStep


class AgentProtocol(Protocol):
	"""Minimal interface orchestrator expects from every agent."""

	name: str

	def run(
		self,
		task: Task,
		context: ConversationContext,
		**kwargs: Any,
	) -> AgentResult:  # pragma: no cover - interface only
		...


class Orchestrator:
	"""Coordinates planning, execution, and response synthesis."""

	def __init__(
		self,
		context: ConversationContext,
		task_manager: Optional[TaskManager] = None,
		planner_agent_name: str = "planner",
		fallback_agent_name: str = "general",
		max_steps: int = 6,
		logger: Optional[logging.Logger] = None,
	) -> None:
		self.context = context
		self.task_manager = task_manager or TaskManager()
		self.planner_agent_name = planner_agent_name
		self.fallback_agent_name = fallback_agent_name
		self.max_steps = max_steps
		self.logger = logger or logging.getLogger("vocalyrics.orchestrator")
		self._agents: Dict[str, AgentProtocol] = {}
		self.finisher_agents = {"composer", "writer", "general"}
		self._step_outputs: Dict[str, AgentResult] = {}

	# ------------------------------------------------------------------
	# Agent registry
	# ------------------------------------------------------------------
	def register_agent(self, agent: AgentProtocol) -> None:
		self._agents[agent.name] = agent

	def register_agents(self, agents: Iterable[AgentProtocol]) -> None:
		for agent in agents:
			self.register_agent(agent)

	def get_agent(self, name: str) -> Optional[AgentProtocol]:
		return self._agents.get(name)

	# ------------------------------------------------------------------
	# Public API
	# ------------------------------------------------------------------
	def handle_user_request(
		self,
		user_input: str,
		attachments: Optional[Dict[str, Any]] = None,
		system_prompt: Optional[str] = None,
		trace_path: Optional[str] = None,
	) -> AgentResult:
		attachments = attachments or {}
		for key, value in attachments.items():
			self.context.set_attachment(key, value)
		self.context.add_user_message(user_input, attachments=list(attachments.keys()))

		trace_target = trace_path or attachments.get("trace_path") or self.context.get_attachment("trace_path")
		trace: Dict[str, Any] = {
			"user_request": user_input,
			"attachments": self._safe_jsonable(attachments),
			"planner": None,
			"steps": [],
			"finisher": None,
		}

		root_task = self.task_manager.create_task(description=user_input)
		plan, planner_result = self._create_plan(root_task, system_prompt=system_prompt)
		if planner_result is not None:
			trace["planner"] = self._build_trace_step(
				agent_name=self.planner_agent_name,
				goal="Plan user request",
				inputs={"system_prompt": system_prompt},
				result=planner_result,
				step_id="plan",
			)

		executed_results: List[Tuple[TaskStep, str, Dict[str, Any], AgentResult]] = []
		self._step_outputs = {}
		for step in plan[: self.max_steps]:
			agent_name, resolved_inputs, step_result = self._execute_step(root_task, step)
			executed_results.append((step, agent_name, resolved_inputs, step_result))
			trace_entry = self._build_trace_step(
				agent_name=agent_name,
				goal=step.goal,
				inputs=resolved_inputs,
				result=step_result,
				step_id=step.step_id,
			)
			trace["steps"].append(trace_entry)
			if not step_result.ok:
				break

		final_result = self._finalize_response(root_task, executed_results, trace)
		if final_result.ok:
			self.task_manager.succeed(root_task.task_id, final_result)
		else:
			self.task_manager.fail(root_task.task_id, final_result.error or "unknown error")

		trace["final_result"] = {
			"content": final_result.content,
			"error": final_result.error,
			"status": "succeeded" if final_result.ok else "failed",
		}
		trace["status"] = trace["final_result"]["status"]
		if trace_target:
			self._persist_trace(trace_target, trace)
		return final_result

	# ------------------------------------------------------------------
	# Planning
	# ------------------------------------------------------------------
	def _create_plan(
		self,
		root_task: Task,
		system_prompt: Optional[str] = None,
	) -> Tuple[List[TaskStep], Optional[AgentResult]]:
		planner = self.get_agent(self.planner_agent_name)
		if not planner:
			default_step = TaskStep(
				step_id="step-1",
				agent=self.fallback_agent_name,
				goal=root_task.description,
				inputs={"prompt": root_task.description},
			)
			self.task_manager.set_steps(root_task.task_id, [default_step])
			return [default_step], None

		plan_task = self.task_manager.create_task(
			description="Plan user request",
			agent=planner.name,
			parent_id=root_task.task_id,
			metadata={"system_prompt": system_prompt},
		)
		try:
			self.task_manager.start(plan_task.task_id)
			result = planner.run(task=plan_task, context=self.context, system_prompt=system_prompt)
			self.task_manager.succeed(plan_task.task_id, result)
			steps = self._parse_plan(result, default_goal=root_task.description)
			self.task_manager.set_steps(root_task.task_id, steps)
			return steps, result
		except Exception as exc:  # pragma: no cover - defensive path
			self.logger.exception("Planner failed: %s", exc)
			self.task_manager.fail(plan_task.task_id, str(exc))
			fallback = TaskStep(
				step_id="fallback-1",
				agent=self.fallback_agent_name,
				goal=root_task.description,
				inputs={"prompt": root_task.description},
			)
			self.task_manager.set_steps(root_task.task_id, [fallback])
			return [fallback], AgentResult(content="", error=str(exc))

	def _parse_plan(self, result: AgentResult, default_goal: str) -> List[TaskStep]:
		steps: List[TaskStep] = []
		try:
			payload = result.artifacts or []
			if payload:
				raw_steps = payload[0].payload  # expect first artifact to contain structured plan
			else:
				raw_steps = None
		except Exception:  # pragma: no cover - safety
			raw_steps = None

		if isinstance(raw_steps, list):
			for idx, step_dict in enumerate(raw_steps, 1):
				agent = step_dict.get("agent", self.fallback_agent_name)
				goal = step_dict.get("goal") or step_dict.get("description") or default_goal
				step_id = step_dict.get("id") or f"step-{idx}"
				inputs_obj = step_dict.get("inputs")
				inputs = inputs_obj if isinstance(inputs_obj, dict) else {}
				if agent == "general" and "prompt" not in inputs:
					inputs["prompt"] = goal
				steps.append(
					TaskStep(step_id=step_id, agent=agent, goal=goal, inputs=inputs)
				)

		if not steps:
			steps.append(
				TaskStep(
					step_id="step-1",
					agent=self.fallback_agent_name,
					goal=default_goal,
					inputs={"prompt": default_goal},
				)
			)
		return steps

	# ------------------------------------------------------------------
	# Step execution
	# ------------------------------------------------------------------
	def _execute_step(self, root_task: Task, step: TaskStep) -> Tuple[str, Dict[str, Any], AgentResult]:
		agent_name = step.agent
		agent = self.get_agent(agent_name)
		if not agent:
			self.logger.warning("Agent %s not registered, falling back", agent_name)
			agent_name = self.fallback_agent_name
			agent = self.get_agent(agent_name)
		if not agent:
			return agent_name, AgentResult(
				content="",
				error=f"No agent available for step {step.step_id}",
			)

		resolved_inputs = self._prepare_step_inputs(step.inputs)
	
		step_task = self.task_manager.create_task(
			description=step.goal,
			agent=agent.name,
			parent_id=root_task.task_id,
			metadata={"step_id": step.step_id, "inputs": resolved_inputs},
		)

		try:
			self.task_manager.start(step_task.task_id)
			result = agent.run(task=step_task, context=self.context, **resolved_inputs)
			self.task_manager.succeed(step_task.task_id, result)

			breakpoint()

			self.context.add_agent_message(agent.name, result.content, step_id=step.step_id)
			self._step_outputs[step.step_id] = result
			return agent.name, resolved_inputs, result
		except Exception as exc:
			self.logger.exception("Agent %s failed: %s", agent_name, exc)
			self.task_manager.fail(step_task.task_id, str(exc))
			failure = AgentResult(content="", error=str(exc))
			self._step_outputs[step.step_id] = failure
			return agent_name, resolved_inputs, failure

	# ------------------------------------------------------------------
	# Finalization
	# ------------------------------------------------------------------
	def _finalize_response(
		self,
		root_task: Task,
		executed_results: List[Tuple[TaskStep, str, Dict[str, Any], AgentResult]],
		trace: Dict[str, Any],
	) -> AgentResult:
		if not executed_results:
			return AgentResult(content="", error="No agent produced a result")

		last_step, last_agent, _, last_result = executed_results[-1]
		finisher = self._determine_finisher(executed_results)
		final = last_result
		if finisher:
			finisher_result, finisher_inputs = self._invoke_finisher(root_task, finisher, executed_results)
			if finisher_result.ok or not last_result.ok:
				final = finisher_result
			trace["finisher"] = self._build_trace_step(
				agent_name=finisher,
				goal="Finalize response",
				inputs=finisher_inputs,
				result=finisher_result,
				step_id="finisher",
			)

		if final.ok:
			self.context.add_agent_message(
				"orchestrator",
				final.content,
				task_id=root_task.task_id,
				finisher=finisher or last_agent,
			)
		return final

	def _determine_finisher(
		self,
		executed_results: List[Tuple[TaskStep, str, Dict[str, Any], AgentResult]],
	) -> Optional[str]:
		preferred = self.context.get_attachment("preferred_finisher")
		if preferred and preferred in self.finisher_agents and self.get_agent(preferred):
			return preferred

		last_step, last_agent, _, last_result = executed_results[-1]
		if last_result.ok and last_agent in self.finisher_agents:
			return None
		if self.fallback_agent_name in self.finisher_agents:
			return self.fallback_agent_name
		return None

	def _invoke_finisher(
		self,
		root_task: Task,
		finisher_name: str,
		executed_results: List[Tuple[TaskStep, str, Dict[str, Any], AgentResult]],
	) -> Tuple[AgentResult, Dict[str, Any]]:
		agent = self.get_agent(finisher_name)
		if not agent:
			return executed_results[-1][2], {}

		finisher_task = self.task_manager.create_task(
			description=f"Finalize response via {finisher_name}",
			agent=agent.name,
			parent_id=root_task.task_id,
			metadata={"role": "finisher"},
		)

		summary = self._summarize_results(root_task.description, executed_results)
		kwargs = self._build_finisher_kwargs(finisher_name, executed_results, summary)

		try:
			self.task_manager.start(finisher_task.task_id)
			result = agent.run(task=finisher_task, context=self.context, **kwargs)
			self.task_manager.succeed(finisher_task.task_id, result)
			self.context.add_agent_message(agent.name, result.content, role="finisher")
			return result, kwargs
		except Exception as exc:  # pragma: no cover - defensive
			self.logger.exception("Finisher %s failed: %s", finisher_name, exc)
			self.task_manager.fail(finisher_task.task_id, str(exc))
			return AgentResult(content="", error=str(exc)), kwargs

	def _summarize_results(
		self,
		user_request: str,
		executed_results: List[Tuple[TaskStep, str, AgentResult]],
	) -> str:
		lines: List[str] = []
		for step, agent_name, _, result in executed_results:
			snippet = result.content.strip().replace("\n", " ")
			if len(snippet) > 200:
				snippet = snippet[:197] + "..."
			status = "OK" if result.ok else "ERROR"
			lines.append(f"- {step.step_id} [{status}] {agent_name}: {snippet or '[no output]'}")
		return f"User request: {user_request}\n\nAgent findings:\n" + "\n".join(lines)

	def _build_finisher_kwargs(
		self,
		finisher_name: str,
		executed_results: List[Tuple[TaskStep, str, Dict[str, Any], AgentResult]],
		summary: str,
	) -> Dict[str, Any]:
		kwargs: Dict[str, Any] = {"brief": summary}
		doc_refs = self._collect_artifacts(executed_results, "documents")
		midi_artifacts = self._collect_artifacts(executed_results, "midi")

		if finisher_name == "composer":
			if doc_refs:
				kwargs["references"] = doc_refs
			if midi_artifacts:
				kwargs["midi_summary"] = midi_artifacts[-1]
			style_hint = self.context.get_attachment("style_hint")
			if style_hint:
				kwargs["style"] = style_hint
			seed_lyrics = self.context.get_attachment("seed_lyrics")
			if seed_lyrics:
				kwargs["seed_lyrics"] = seed_lyrics
		elif finisher_name == "writer":
			format_hint = self.context.get_attachment("format_hint")
			if format_hint:
				kwargs["format_hint"] = format_hint
		else:
			kwargs["brief"] = summary
		return kwargs

	def _collect_artifacts(
		self,
		executed_results: List[Tuple[TaskStep, str, Dict[str, Any], AgentResult]],
		kind: str,
	) -> List[Any]:
		payloads: List[Any] = []
		for _, _, _, result in executed_results:
			for artifact in result.artifacts:
				if artifact.kind == kind:
					payloads.append(artifact.payload)
		return payloads

	def _prepare_step_inputs(self, raw_inputs: Optional[Dict[str, Any]]) -> Dict[str, Any]:
		if not raw_inputs:
			return {}
		prepared: Dict[str, Any] = {}
		for key, value in raw_inputs.items():
			if isinstance(key, str) and key.endswith("_from"):
				target_field = key[: -5]
				resolved = self._resolve_reference(target_field, value)
				if target_field in prepared:
					prepared[target_field] = self._merge_reference_values(prepared[target_field], resolved)
				else:
					prepared[target_field] = resolved
			else:
				prepared[key] = value
		return prepared

	def _merge_reference_values(self, existing: Any, incoming: Any) -> Any:
		if existing is None:
			return incoming
		if incoming is None:
			return existing
		if isinstance(existing, list) and isinstance(incoming, list):
			return existing + incoming
		if isinstance(existing, list):
			existing = existing + ([incoming] if not isinstance(incoming, list) else incoming)
			return existing
		if isinstance(incoming, list):
			return [existing] + incoming
		return incoming

	def _resolve_reference(self, target_field: str, specification: Any) -> Any:
		if specification is None:
			return None
		if isinstance(specification, str):
			spec: Dict[str, Any] = {"step": specification}
		elif isinstance(specification, dict):
			spec = dict(specification)
		else:
			return None

		step_id = spec.get("step")
		if not step_id:
			return None
		source_result = self._step_outputs.get(step_id)
	
		if not source_result:
			return None

		conf = self._default_reference_config(target_field)
		artifact_kind = spec.get("artifact") or conf.get("artifact")
		field_name = spec.get("field") or conf.get("field") or "content"
		mode = spec.get("mode") or conf.get("mode") or "list"

		if artifact_kind:
			payloads = [
				artifact.payload
				for artifact in (source_result.artifacts or [])
				if artifact.kind == artifact_kind
			]
			if not payloads:
				return None
			if mode == "last":
				return payloads[-1]
			if mode == "first":
				return payloads[0]
			return payloads

		return getattr(source_result, field_name, None)

	def _default_reference_config(self, field_name: str) -> Dict[str, Any]:
		mapping: Dict[str, Dict[str, Any]] = {
			"references": {"artifact": "documents", "mode": "list"},
			"midi_summary": {"artifact": "midi", "mode": "last"},
			"seed_lyrics": {"field": "content", "mode": "last"},
			"brief": {"field": "content", "mode": "last"},
			"focus": {"field": "content", "mode": "last"},
			"prompt": {"field": "content", "mode": "last"},
		}
		return mapping.get(field_name, {"field": "content", "mode": "last"})

	def _build_trace_step(
		self,
		agent_name: str,
		goal: str,
		inputs: Optional[Dict[str, Any]],
		result: Optional[AgentResult],
		step_id: Optional[str],
	) -> Dict[str, Any]:
		entry: Dict[str, Any] = {
			"step_id": step_id,
			"agent": agent_name,
			"goal": goal,
			"inputs": self._safe_jsonable(inputs or {}),
			"status": "skipped" if result is None else ("succeeded" if result.ok else "failed"),
		}
		if result is not None:
			entry["output"] = result.content
			entry["error"] = result.error
			entry["artifacts"] = [
				{"kind": artifact.kind}
				for artifact in (result.artifacts or [])
			]
		return entry

	def _safe_jsonable(self, value: Any) -> Any:
		if isinstance(value, (str, int, float, bool)) or value is None:
			return value
		if isinstance(value, dict):
			return {str(k): self._safe_jsonable(v) for k, v in value.items()}
		if isinstance(value, (list, tuple, set)):
			return [self._safe_jsonable(v) for v in value]
		return str(value)

	def _persist_trace(self, path_str: str, trace: Dict[str, Any]) -> None:
		try:
			path = Path(path_str)
			path.parent.mkdir(parents=True, exist_ok=True)
			path.write_text(
				json.dumps(trace, ensure_ascii=False, indent=2),
				encoding="utf-8",
			)
		except Exception as exc:  # pragma: no cover - defensive
			self.logger.warning("Failed to write trace file %s: %s", path_str, exc)

