import asyncio
import os
from dotenv import load_dotenv
from loguru import logger

from rag_system.agents.manager import ManagerAgent
from rag_system.llm.base import LLMError
from rag_system.llm.factory import create_llm
from rag_system.utils.config import load_config


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

            if question.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break

            result = await manager.answer_question(question)

            print("\n" + "=" * 60)
            print("ANSWER:\n")
            print(result["answer"])
            print("\n" + "-" * 60)
            print(f"Specialist used: {result.get('specialist_used', 'N/A')}")
            if result.get("total_time_ms"):
                print(f"Total time: {result['total_time_ms']}ms")
            print("\nTop sources:")
            for i, source in enumerate(result.get("sources", [])[:3], 1):
                snippet = source.get("content") or source.get("snippet") or ""
                print(f"  {i}. {snippet[:100]}...")
            print("=" * 60)

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            logger.error(f"Error: {e}")


def main():
    load_dotenv()
    cfg = load_config()

    try:
        provider = cfg.get("llm", {}).get("provider", "gemini").lower()
        if provider == "gemini" and os.getenv("GEMINI_API_KEY", "").strip() == "your_gemini_api_key_here":
            raise LLMError("GEMINI_API_KEY still has placeholder value.")
        llm = create_llm(cfg)
    except LLMError as exc:
        print(f"LLM configuration error: {exc}")
        print("Check .env (e.g. GEMINI_API_KEY / OPENAI_API_KEY) and llm.provider in config.yaml.")
        return

    mcp_host = os.getenv("MCP_SERVER_HOST", "localhost")
    if mcp_host == "0.0.0.0":
        mcp_host = "localhost"
    mcp_port = os.getenv("MCP_SERVER_PORT", "8000")
    mcp_url = os.getenv("MCP_SERVER_URL", f"http://{mcp_host}:{mcp_port}")

    manager = ManagerAgent(
        agent_id="manager",
        mcp_url=mcp_url,
        llm=llm,
    )

    asyncio.run(interactive_mode(manager))


if __name__ == "__main__":
    main()
