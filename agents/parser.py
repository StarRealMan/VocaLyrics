import os
from typing import Any

from core.context import Context
from core.task import Task
from agents.base import Agent
from utils.midi import parse_midi


class Parser(Agent):
    """
    解析器 Agent
    
    负责解析 MIDI 文件，提取音乐结构信息（如音符、节奏、BPM等）。
    """
    
    def __init__(self):
        super().__init__(name="Parser", description="Parses MIDI files to extract musical structure and metadata.")

    def run(self, context: Context, task: Task) -> Any:
        """
        执行解析任务
        """
        params = task.input_params
        file_path = params.file_path
        
        if not file_path:
            raise ValueError("Parser requires a 'file_path' parameter.")
            
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"MIDI file not found at path: {file_path}")
            
        self.logger.debug(f"Parsing MIDI file: {file_path}...")
        
        try:
            # 调用 utils.midi.parse_midi
            midi_data = parse_midi(file_path)
            
            # 结果包含 meta 和 notes
            # 我们可以做一些简单的统计，方便 Writer 或 Lyricist 使用
            note_count = len(midi_data.get("notes", []))
            meta = midi_data.get("meta", {})
            
            summary = f"Parsed MIDI file with {note_count} notes. Meta: {meta}"
            self.logger.debug(f"Success: {summary}")
            
            # 保存完整数据到 Context
            self._save_to_memory(context, task, midi_data)
            
            return midi_data
            
        except Exception as e:
            raise ValueError(f"Failed to parse MIDI file: {str(e)}")
