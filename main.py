import argparse
import os
import sys
import logging
from dotenv import load_dotenv

# 确保项目根目录在 sys.path 中
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.orchestrator import Orchestrator
from agents.planner import Planner
from agents.retriever import Retriever
from agents.analyst import Analyst
from agents.writer import Writer
from agents.parser import Parser
from utils.logger import setup_logger

def main():
    parser = argparse.ArgumentParser(description="VocaLyrics Multi-Agent System")
    parser.add_argument("--query", type=str, default=None, help="User query to process.")
    parser.add_argument("--trace", action="store_true", help="Enable tracing to save execution logs to 'trace/' directory.")
    parser.add_argument("--midi", type=str, help="Path to a MIDI file to process.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    args = parser.parse_args()

    setup_logger(verbose=args.verbose)
    
    planner = Planner()
    retriever = Retriever()
    analyst = Analyst()
    writer = Writer()
    parser_agent = Parser()
    
    agents = {
        "Planner": planner,
        "Retriever": retriever,
        "Analyst": analyst,
        "Writer": writer,
        "Parser": parser_agent
    }

    orchestrator = Orchestrator(agents=agents)
    trace_dir = "trace" if args.trace else None

    if args.query:
        user_query = args.query + (f" [MIDI: {args.midi}]" if args.midi else "")
        print(f"\nUser Query: {user_query}\n")
        response = orchestrator.run(user_query, trace_dir=trace_dir)
        print(f"\nFinal Response: {response}")
        return

    print("=== VocaLyrics Interactive Mode ===")

    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["exit", "quit", "q"]:
                break
            if not user_input.strip():
                continue
            
            response = orchestrator.run(user_input, trace_dir=trace_dir)
            print(f"Assistant: {response}\n")

        except EOFError:
            print("\nExiting...")
            break
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
