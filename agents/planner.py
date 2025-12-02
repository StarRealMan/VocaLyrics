import os
from dotenv import load_dotenv
from typing import List, Union, Optional
from pydantic import BaseModel

from core.context import Context
from core.task import Task, RetrieverInput, AnalystInput, ParserInput, LyricistInput, WriterInput
from agents.base import Agent


class PlannedTask(BaseModel):
  """Planner 生成的轻量 Task 模型，用于描述计划结构。"""

  description: str
  assigned_agent: str
  input_params: Union[RetrieverInput, AnalystInput, ParserInput, LyricistInput, WriterInput]
  output_key: Optional[str] = None
  dependencies: List[str] = []

class PlannerResult(BaseModel):
  """Planner LLM 的整体输出结构。"""

  tasks: List[PlannedTask]

class Planner(Agent):
    """
    规划者 Agent
    
    负责分析用户意图，将复杂任务拆解为一系列可执行的子任务 (Task)。
    """
    
    def __init__(self, openai_client):
        super().__init__(name="Planner", description="Decomposes user queries into executable plans.")
        load_dotenv()
        self.openai_client = openai_client
        self.model = os.getenv("OPENAI_API_MODEL", "gpt-5.1")

    def run(self, context: Context, task: Task) -> List[Task]:
        """
        执行规划任务
        """
        user_query = self._get_param(task, "query")
        if not user_query:
            # 尝试从对话历史中获取
            if context.chat_history:
                user_query = context.chat_history[-1]["content"]
            else:
                raise ValueError("No query provided for planning.")

        # 构建 Prompt
        system_prompt = self._build_system_prompt()
        
        # 构建 Messages (包含历史对话)
        messages = [{"role": "system", "content": system_prompt}]
        
        # 添加历史对话 (排除最后一条，因为最后一条是当前的 query，我们在下面会专门处理它以附加 Context Keys)
        if len(context.chat_history) > 1:
            for msg in context.chat_history[:-1]:
                # 过滤掉 system 类型的消息，只保留 user 和 assistant
                if msg["role"] in ["user", "assistant"]:
                    messages.append({"role": msg["role"], "content": msg["content"]})
        
        # 添加当前 Query 和 Context 提示
        current_query_content = f"User Query: {user_query}\n\nCurrent Context Keys: {list(context.shared_memory.keys())}"
        messages.append({"role": "user", "content": current_query_content})
        
        # 调用 LLM，要求按 PlannerResult 结构输出
        response = self.openai_client.responses.parse(
          model=self.model,
          input=messages,
          text_format=PlannerResult,
        )
        parsed: PlannerResult = response.output_parsed  # SDK 返回已解析的 Pydantic 对象

        # 将 PlannedTask 转换为系统内部使用的 Task
        new_plan: List[Task] = []
        for t in parsed.tasks:
          new_task = Task(
            description=t.description,
            assigned_agent=t.assigned_agent,
            input_params=t.input_params,
            output_key=t.output_key,
            dependencies=t.dependencies or [],
          )
          new_plan.append(new_task)

        # 更新 Context 中的计划
        context.set_plan(new_plan)

        return new_plan

    def _build_system_prompt(self) -> str:
        return """
You are the Planner Agent for a Vocaloid Lyrics Analysis System.
Your goal is to break down a user's request into a sequence of executable tasks.

IMPORTANT SYSTEM KNOWLEDGE:
- The vector database embeddings are based on LYRICS.
- You CANNOT directly search for "songs similar to [Song Name]" because the database compares lyrical content, not abstract musical style.
- Strategy for "Find similar songs to [Song Name]":
  1. Task 1 (Retriever): Find the specific song to get its lyrics/content.
  2. Task 2 (Analyst): Analyze the retrieved song to extract key themes, imagery, and emotions.
  3. Task 3 (Retriever): Search for new songs using the extracted themes/imagery as the search query.
  4. Task 4 (Writer): Summarize the findings and recommend the songs to the user.

Available Agents:
1. Retriever: 
   - Capabilities: Search for songs, lyrics, or metadata in the vector database.
   - Input Params: 'request' (str).
   - Use when: User asks to find songs, recommend songs, or needs lyrics for analysis. Provide a natural language description of what to search for.

2. Analyst:
   - Capabilities: Analyze lyrics, style, emotions, or imagery.
   - Input Params: 'target_text' (str) or 'data_key' (str - key in shared_memory).
   - Use when: User asks for analysis, explanation, or understanding of a song/artist style.

3. Parser:
   - Capabilities: Parse MIDI files to extract structure.
   - Input Params: 'file_path' (str).
   - Use when: The user input contains a MIDI file path marked with the tag '[MIDI: <file_path>]'. Extract the path from the tag and pass it to this agent.

4. Lyricist:
   - Capabilities: Generate lyrics, rewrite lyrics, or fill lyrics for a melody.
   - Input Params: 'style' (str), 'theme' (str), 'midi_structure' (dict), 'base_lyrics' (str).
   - Use when: User asks to write lyrics, continue lyrics, or fill lyrics.

5. Writer:
   - Capabilities: Creative writing, summarizing results, generating final responses.
   - Input Params: 'topic' (str), 'source_material_key' (str - optional, key in shared_memory).
   - Use when: User asks for stories, world settings, OR when you need to summarize search/analysis results into a final answer for the user. ALWAYS use this as the final step if the user expects a text response.

6. General:
   - Capabilities: Handle general queries unrelated to Vocaloid or specific tools.
   - Input Params: 'query' (str).
   - Use when: The request is a general chat or doesn't fit other agents.

Output Format:
You must output a JSON object with a single key "tasks", which is a list of task objects.
Each task object must have:
- "description": (str) Clear instruction for the agent.
- "assigned_agent": (str) One of the agent names above.
- "input_params": (dict) Parameters for the agent.
- "output_key": (str, optional) Key to store the result in shared memory (e.g., "search_results", "analysis_report").
- "dependencies": (list[str], optional) IDs of tasks that must finish before this one. (Since you generate IDs, you can omit this for simple sequential lists, the Orchestrator executes sequentially by default).

Example 1 (Analysis):
User: "Analyze the style of Deco*27."
JSON Output:
{
  "tasks": [
    {
      "description": "Search for top 5 popular songs by Deco*27.",
      "assigned_agent": "Retriever",
      "input_params": {
        "request": "Find top 5 popular songs by producer Deco*27"
      },
      "output_key": "deco_songs"
    },
    {
      "description": "Analyze the musical and lyrical style based on the retrieved songs.",
      "assigned_agent": "Analyst",
      "input_params": {
        "data_key": "deco_songs"
      },
      "output_key": "style_analysis"
    }
  ]
}

Example 2 (Similarity Search):
User: "Find songs similar to Rolling Girl."
JSON Output:
{
  "tasks": [
    {
      "description": "Find the song 'Rolling Girl' to get its lyrics.",
      "assigned_agent": "Retriever",
      "input_params": {
        "request": "Find the song named 'Rolling Girl'"
      },
      "output_key": "target_song"
    },
    {
      "description": "Analyze the lyrics of 'Rolling Girl' to extract themes and imagery.",
      "assigned_agent": "Analyst",
      "input_params": {
        "data_key": "target_song"
      },
      "output_key": "song_analysis"
    },
    {
      "description": "Search for songs with similar themes and imagery based on the analysis.",
      "assigned_agent": "Retriever",
      "input_params": {
        "request": "Find songs with themes and imagery matching the analysis in 'song_analysis'"
      },
      "output_key": "similar_songs"
    },
    {
      "description": "Summarize the found similar songs and present them to the user.",
      "assigned_agent": "Writer",
      "input_params": {
        "topic": "Recommend the similar songs found to the user, explaining why they fit the style of Rolling Girl.",
        "source_material_key": "similar_songs"
      },
      "output_key": "final_response"
    }
  ]
}
"""
