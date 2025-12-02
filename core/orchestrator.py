import json
import os
import time
import logging
from typing import Dict, Optional

from core.context import Context
from core.task import Task, TaskStatus
from agents.base import Agent


class Orchestrator:
    """
    核心调度器
    
    负责接收用户请求，调用 Planner 生成计划，并调度各个 Agent 执行任务。
    """
    
    def __init__(self, agents: Dict[str, Agent], planner_agent_name: str = "Planner"):
        self.agents = agents
        self.planner_name = planner_agent_name
        self.context = Context()
        self.logger = logging.getLogger("Orchestrator")
        
        if self.planner_name not in self.agents:
            raise ValueError(f"Planner agent '{self.planner_name}' not found in registered agents.")

    def register_agent(self, agent: Agent):
        """注册一个新的 Agent"""
        self.agents[agent.name] = agent

    def run(self, user_query: str, trace_dir: Optional[str] = None) -> str:
        """
        处理用户请求的主循环
        
        Args:
            user_query: 用户的自然语言请求
            trace_dir: 如果提供，将在此目录保存详细的执行追踪日志 (JSON)
        """
        # 1. 初始化上下文 (使用持久化的 self.context)
        self.context.clear_plan()
        self.context.add_user_message(user_query)
        final_response = ""
        
        trace_data = {
            "query": user_query,
            "timestamp": time.time(),
            "steps": []
        }
        
        # 2. 规划阶段 (Planning)
        self.logger.debug("Starting Planning Phase")
        planner = self.agents.get(self.planner_name)
        if not planner:
            return f"Error: Planner agent '{self.planner_name}' not found."
            
        # 创建一个伪任务给 Planner
        planning_task = Task(
            description="Analyze user query and generate execution plan.",
            assigned_agent=self.planner_name,
            input_params={"query": user_query}
        )
        
        try:
            # Planner 的 run 方法应该填充 context.plan
            plan = planner.run(self.context, planning_task)
            
            # 记录 Planner 的 Trace
            if trace_dir:
                trace_data["steps"].append({
                    "step": "Planning",
                    "agent": self.planner_name,
                    "input": planning_task.model_dump(),
                    "output": [t.model_dump() for t in self.context.plan], # 记录生成的计划
                    "context_snapshot": self.context.model_dump()
                })
            plan_summary = ", ".join(
                [f"{t.assigned_agent}: {t.description}" for t in (plan or [])]
            )
            self.logger.debug(
                f"Planner generated {len(plan or [])} tasks: {plan_summary}" if plan else "Planner generated 0 tasks."
            )
            
            assert plan[-1].assigned_agent in {"Lyricist", "Writer", "General"}, (
                "The final task in the plan must be assigned to either 'Lyricist', 'Writer', or 'General' agent to produce the final user-facing response."
            )

        except Exception as e:
            return f"Planning failed: {str(e)}"
            
        if not self.context.plan:
            return "Planner did not generate any tasks. Please try a different query."

        # 3. 执行阶段 (Execution)
        self.logger.debug(f"Starting Execution Phase ({len(self.context.plan)} tasks)")
        
        # 简单的顺序执行
        # 进阶版本可以构建 DAG 并并发执行无依赖的任务
        for num, task in enumerate(self.context.plan):
            if task.status == TaskStatus.COMPLETED:
                continue
                
            agent_name = task.assigned_agent
            agent = self.agents.get(agent_name)
            
            if not agent:
                error_msg = f"Agent '{agent_name}' not found for task: {task.description}"
                self.logger.error(error_msg)
                task.mark_failed(error_msg)
                continue
                
            self.logger.debug(f"Executing Task [{task.id[:8]}]: {task.description} (Agent: {agent_name})")
            task.mark_in_progress()
            
            try:
                # 执行任务
                result = agent.run(self.context, task)
                task.mark_completed(result)
                self.logger.debug(f"Task Completed. Result: {str(result)[:50]}...")

                if num == len(self.context.plan) - 1 and isinstance(result, str):
                    final_response = result
                
                # 记录 Execution Trace
                if trace_dir:
                    trace_data["steps"].append({
                        "step": "Execution",
                        "task_id": task.id,
                        "agent": agent_name,
                        "input": task.model_dump(),
                        "output": result,
                        "context_snapshot": self.context.model_dump() # 记录每一步后的 Context 状态
                    })
                    
            except Exception as e:
                error_msg = f"Execution failed: {str(e)}"
                self.logger.error(error_msg)
                task.mark_failed(error_msg)
                
                if trace_dir:
                    trace_data["steps"].append({
                        "step": "Execution_Failed",
                        "task_id": task.id,
                        "agent": agent_name,
                        "error": str(e)
                    })
        
        # 保存 Trace 文件
        if trace_dir:
            os.makedirs(trace_dir, exist_ok=True)
            trace_file = os.path.join(trace_dir, f"trace_{int(time.time())}.json")
            with open(trace_file, "w", encoding="utf-8") as f:
                json.dump(trace_data, f, ensure_ascii=False, indent=2, default=str)
            self.logger.debug(f"Trace saved to {trace_file}")

        # 4. 最终响应
        if final_response:
            self.context.add_assistant_message(final_response)
            return final_response
            
        no_response_msg = "Task execution finished, but no response was generated."
        self.context.add_assistant_message(no_response_msg)
        return no_response_msg
