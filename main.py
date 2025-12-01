import argparse
import os
import sys
from dotenv import load_dotenv

# 确保项目根目录在 sys.path 中
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.orchestrator import Orchestrator
from agents.planner import Planner
from agents.retriever import Retriever
from agents.analyst import Analyst
from agents.writer import Writer

def main():
    parser = argparse.ArgumentParser(description="VocaLyrics Multi-Agent System")
    parser.add_argument("--query", type=str, default="有没有和《ローリンガール》气质相似的歌？", help="User query to process.")
    parser.add_argument("--trace", action="store_true", help="Enable tracing to save execution logs to 'trace/' directory.")
    args = parser.parse_args()

    load_dotenv()
    
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found in environment variables.")
        print("Please create a .env file with your API key.")
        return

    planner = Planner()
    retriever = Retriever()
    analyst = Analyst()
    writer = Writer()
    
    agents = {
        "Planner": planner,
        "Retriever": retriever,
        "Analyst": analyst,
        "Writer": writer
    }

    orchestrator = Orchestrator(agents=agents)

    print(f"\nUser Query: {args.query}\n")
    trace_dir = "trace" if args.trace else None
    response = orchestrator.run(args.query, trace_dir=trace_dir)
    
    print(f"\nFinal Response: {response}")

if __name__ == "__main__":
    main()
