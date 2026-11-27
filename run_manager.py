import asyncio
from dotenv import load_dotenv
import os
import yaml
from loguru import logger

from rag_system.agents.manager import ManagerAgent


def load_config():
    config_path = "config.yaml"
    if os.path.exists(config_path):
        with open(config_path) as f:
            return yaml.safe_load(f)
    return {}


async def interactive_mode(manager: ManagerAgent):
    print("\n" + "=" * 60)
    print("MANAGER AGENT - Interactive Mode")
    print("Type your questions below. Type 'quit' to exit.")
    print("=" * 60)

    while True:
        try:
            question = input("\nYour question: ").strip()

            if not question:
                continue

            if question.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break

            result = await manager.answer_question(question)

            print("\n" + "=" * 60)
            print("ANSWER:\n")
            print(result['answer'])
            print("\n" + "-" * 60)
            print(f"Specialist used: {result.get('specialist_used', 'N/A')}")
            if result.get('total_time_ms'):
                print(f"Total time: {result['total_time_ms']}ms")
            print("\nTop sources:")
            for i, source in enumerate(result.get('sources', [])[:3], 1):
                print(f"  {i}. {source['content'][:100]}...")
            print("=" * 60)

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            logger.error(f"Error: {e}")


def main():
    load_dotenv()
    config = load_config()

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "your_gemini_api_key_here":
        print("GEMINI_API_KEY not set in .env file.")
        print("Get your free key from: https://aistudio.google.com/app/apikey")
        print("Then add it to .env: GEMINI_API_KEY=your_actual_key")
        return

    mcp_host = os.getenv("MCP_SERVER_HOST", "localhost")
    if mcp_host == "0.0.0.0":
        mcp_host = "localhost"
    mcp_port = os.getenv("MCP_SERVER_PORT", "8000")
    mcp_url = os.getenv("MCP_SERVER_URL", f"http://{mcp_host}:{mcp_port}")
    model_name = os.getenv("GEMINI_MODEL", config.get("gemini", {}).get("model_name", "gemini-2.5-flash"))

    manager = ManagerAgent(
        agent_id="manager",
        mcp_url=mcp_url,
        api_key=api_key,
        model_name=model_name
    )

    asyncio.run(interactive_mode(manager))


if __name__ == "__main__":
    main()
