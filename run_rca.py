import json
import os
import sys
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage

# Add src to path so we can import from open_deep_research
sys.path.append(str(Path(__file__).parent / "src"))

from open_deep_research.data_tools import (
    list_tables_in_directory,
    get_schema,
    query_parquet_files,
)

def load_task(task_file: str = "task.json") -> str:
    """Load the task description from a JSON file."""
    try:
        with open(task_file, "r") as f:
            data = json.load(f)
            return data.get("task_description", "")
    except FileNotFoundError:
        print(f"Error: Task file '{task_file}' not found.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Task file '{task_file}' is not valid JSON.")
        sys.exit(1)

def save_output(output_file: str, messages: list):
    """Save the conversation history to a JSON file."""
    serializable_messages = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            serializable_messages.append({"type": "human", "content": msg.content})
        elif isinstance(msg, AIMessage):
            serializable_messages.append({
                "type": "ai", 
                "content": msg.content,
                "tool_calls": msg.tool_calls
            })
        # We can add other message types if needed, but these are the main ones
        else:
             serializable_messages.append({"type": msg.type, "content": msg.content})

    output_data = {
        "timestamp": datetime.now().isoformat(),
        "messages": serializable_messages
    }
    
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"Output saved to {output_file}")

def main():
    # Load environment variables
    load_dotenv()

    # 1. Load Task
    task_description = load_task("task.json")
    print("Task loaded successfully.")

    # 2. Setup Tools
    tools = [list_tables_in_directory, get_schema, query_parquet_files]

    # 3. Setup LLM
    # You can configure the model here. Using GPT-4o as a default high-capability model.
    # Ensure OPENAI_API_KEY is set in the environment.
    model = ChatOpenAI(model="gpt-4o", temperature=0)

    # 4. Create Agent
    agent_executor = create_react_agent(model, tools)

    # 5. Run Agent
    print("Starting analysis...")
    events = agent_executor.stream(
        {"messages": [HumanMessage(content=task_description)]},
        stream_mode="values"
    )

    messages = []
    for event in events:
        if "messages" in event:
            messages = event["messages"]
            last_message = messages[-1]
            if isinstance(last_message, AIMessage):
                if last_message.tool_calls:
                    print(f"Agent calling tools: {[tc['name'] for tc in last_message.tool_calls]}")
                else:
                    print(f"Agent: {last_message.content}")
            elif isinstance(last_message, HumanMessage):
                print(f"User: {last_message.content[:100]}...")

    # 6. Save Output
    output_path = Path("experiments/output.json")
    save_output(str(output_path), messages)

if __name__ == "__main__":
    main()
