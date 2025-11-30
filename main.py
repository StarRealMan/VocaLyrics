"""Command-line entrypoint for the VocaLyrics multi-agent chatbot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from core.bootstrap import create_default_orchestrator
from core.context import ConversationContext
from core.task import AgentResult


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="VocaLyrics multi-agent assistant")
	parser.add_argument("--query", type=str, help="Single-turn query to run", default=None)
	parser.add_argument(
		"--interactive",
		action="store_true",
		help="Launch interactive chat session",
	)
	parser.add_argument(
		"--trace-file",
		type=str,
		default=None,
		help="Write agent trace JSON to this path (interactive mode appends -turnN)",
	)
	return parser.parse_args()


def handle_query(orchestrator, query: str, trace_path: Optional[str] = None) -> AgentResult:
	return orchestrator.handle_user_request(query, trace_path=trace_path)


def interactive_loop(orchestrator, trace_template: Optional[str] = None) -> None:
	print("Entering interactive mode. Type 'exit' to quit.")
	turn_idx = 1
	context = orchestrator.context  # type: ignore[attr-defined]
	while True:
		try:
			user_input = input("You: ").strip()
		except (KeyboardInterrupt, EOFError):
			print("\nBye!")
			break
		if user_input.lower() in {"exit", "quit"}:
			print("Bye!")
			break
		handled, response = handle_inline_command(user_input, context)
		if handled:
			if response:
				print(response)
			continue
		trace_path = None
		if trace_template:
			trace_path = build_trace_path(trace_template, turn_idx)
			turn_idx += 1
		result = handle_query(orchestrator, user_input, trace_path=trace_path)
		if result.error:
			print(f"Agent error: {result.error}")
		else:
			print(f"VocaLyrics: {result.content}\n")


def main() -> None:
	args = parse_args()
	orchestrator, _ = create_default_orchestrator()
	
	if args.query:
		result = handle_query(orchestrator, args.query, trace_path=args.trace_file)
		if result.error:
			raise SystemExit(f"Agent error: {result.error}")
		print(result.content)
		if not args.interactive:
			return

	if args.interactive:
		interactive_loop(orchestrator, trace_template=args.trace_file)
	else:
		if not args.query:
			raise SystemExit("Provide --query or --interactive")


def build_trace_path(template: str, turn_idx: int) -> str:
	path = Path(template)
	parent = path.parent if path.parent != Path("") else Path(".")
	suffix = path.suffix or ".json"
	stem = path.stem if path.suffix else path.name
	return str(parent / f"{stem}-turn{turn_idx}{suffix}")


def handle_inline_command(user_input: str, context: ConversationContext) -> tuple[bool, Optional[str]]:
	if not user_input.startswith(":"):
		return False, None
	parts = user_input[1:].split(maxsplit=2)
	if not parts:
		return True, "Unknown command"
	cmd = parts[0].lower()
	if cmd == "attach" and len(parts) >= 2:
		target_token = parts[1].lower()
		value = parts[2] if len(parts) >= 3 else None
		if target_token == "midi":
			if not value:
				return True, "Usage: :attach midi /path/to/file.mid"
			resolved = Path(value).expanduser()
			if not resolved.exists():
				return True, f"[attach] File not found: {resolved}"
			context.set_attachment("midi_path", str(resolved))
			return True, f"[attach] MIDI path set to {resolved}"
		return True, f"[attach] Unsupported target '{target_token}'"
	return True, "Unknown command"


if __name__ == "__main__":
	main()

