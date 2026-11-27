"""
Manager Agent — the user-facing orchestrator.

Decides whether a question requires document retrieval, routes it to the
appropriate specialist via MCP, and synthesizes a final answer using Gemini.

Gemini API calls are wrapped with exponential-backoff retry (tenacity) to
handle transient 429 ResourceExhausted errors gracefully.
"""
from typing import List, Dict, Any, Optional
import time
import httpx
import google.generativeai as genai
import google.api_core.exceptions
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)
from loguru import logger

from rag_system.mcp_server.models import Message
from rag_system.utils.metrics import MetricsCollector, QueryMetric


SYNTHESIS_SYSTEM_PROMPT = """You are a meticulous technical analyst. Your job is to synthesize a clear, concise answer based only on the provided context and the user's question. You must:
1. Only use information from the provided context passages.
2. If the context does not contain enough information, explicitly state that.
3. At the end of your answer, include a "Sources" section listing the document name and chunk index for each passage you referenced, in the format: [Source: filename, chunk N]."""


class ManagerAgent:

    def __init__(
        self,
        agent_id: str = "manager",
        mcp_url: str = "http://localhost:8000",
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.5-flash",
    ):
        self.agent_id = agent_id
        self.mcp_url = mcp_url

        if not api_key:
            raise ValueError("GEMINI_API_KEY is required!")

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.model_name = model_name
        self.metrics = MetricsCollector()
        logger.info(f"Manager agent '{agent_id}' initialized with {model_name}")

    # ------------------------------------------------------------------
    # Gemini API call with retry/backoff
    # ------------------------------------------------------------------

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(google.api_core.exceptions.ResourceExhausted),
        reraise=True,
    )
    def _call_gemini(self, prompt: str) -> str:
        """Call Gemini with automatic retry on 429 rate-limit errors."""
        response = self.model.generate_content(prompt)
        return response.text

    # ------------------------------------------------------------------
    # Decision: does this question need retrieval?
    # ------------------------------------------------------------------

    def _should_use_retrieval(self, question: str, available_tools: List[str]) -> bool:
        """Ask Gemini whether to use the document_retriever tool."""
        if "document_retriever" not in available_tools:
            return False

        prompt = f"""You are an AI assistant deciding whether to use a document retrieval tool.
The user asked: "{question}"

Available tools: {available_tools}

Should the document_retriever tool be used to answer this question?
Reply with ONLY "yes" or "no"."""

        try:
            decision = self._call_gemini(prompt).strip().lower()
            use_tool = "yes" in decision
            logger.info(
                f"LLM tool decision for '{question[:50]}...': "
                f"{'USE TOOL' if use_tool else 'SKIP TOOL'}"
            )
            return use_tool
        except Exception as e:
            logger.warning(f"Tool decision failed ({e}), defaulting to use retrieval")
            return True

    # ------------------------------------------------------------------
    # MCP agent discovery and selection
    # ------------------------------------------------------------------

    async def discover_specialists(self) -> List[Dict]:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{self.mcp_url}/agents")
            response.raise_for_status()

            agents = response.json()
            specialists = [
                agent for agent in agents if agent["agent_id"] != self.agent_id
            ]

            logger.info(f"Found {len(specialists)} specialist(s)")
            return specialists

    def _select_specialist(
        self, question: str, specialists: List[Dict]
    ) -> Optional[Dict]:
        """Score specialists by keyword overlap with the question description."""
        if not specialists:
            return None

        question_lower = question.lower()
        best_match = None
        best_score = 0

        for spec in specialists:
            score = 0
            desc_lower = spec.get("description", "").lower()

            tools = spec.get("tools", [])
            if any(t["name"] == "document_retriever" for t in tools):
                score += 1

            keywords = desc_lower.split()
            for word in question_lower.split():
                if word in keywords:
                    score += 2

            if score > best_score:
                best_score = score
                best_match = spec

        if best_match:
            return best_match

        # fall back: any specialist with document_retriever
        for spec in specialists:
            tools = spec.get("tools", [])
            if any(t["name"] == "document_retriever" for t in tools):
                return spec

        return specialists[0]

    # ------------------------------------------------------------------
    # MCP round-trip
    # ------------------------------------------------------------------

    async def query_specialist(
        self, specialist: Dict, query: str, top_k: int = 5
    ) -> Dict:
        message = Message(
            sender=self.agent_id,
            recipient=specialist["agent_id"],
            tool="document_retriever",
            parameters={"query": query, "top_k": top_k},
        )

        t_start = time.time()

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{self.mcp_url}/send",
                json=message.dict(),
            )
            response.raise_for_status()
            result = response.json()

        latency_ms = (time.time() - t_start) * 1000
        logger.info(f"MCP tool call latency: {latency_ms:.1f}ms")

        if not result.get("success"):
            return {"error": result.get("error")}

        return result["result"]

    # ------------------------------------------------------------------
    # Answer synthesis
    # ------------------------------------------------------------------

    def _synthesize_answer(self, question: str, search_results: Dict) -> str:
        results = search_results.get("results", [])

        if not results:
            return "I couldn't find relevant information to answer your question."

        context_parts = []
        for i, result in enumerate(results[:5], 1):
            source_file = result.get("metadata", {}).get("filename", "unknown")
            chunk_idx = result.get("metadata", {}).get("chunk_index", "?")
            similarity = result.get("similarity", 0.0)
            context_parts.append(
                f"Passage {i} [Source: {source_file}, chunk {chunk_idx}] "
                f"(similarity: {similarity:.3f}):\n{result['content']}\n"
            )

        context = "\n".join(context_parts)

        prompt = f"""{SYNTHESIS_SYSTEM_PROMPT}

Context:
{context}

Question: {question}

Provide a clear, concise answer with source citations."""

        logger.info(f"Synthesis prompt length: {len(prompt)} chars")
        logger.debug(f"Full synthesis prompt:\n{prompt}")

        t_start = time.time()
        try:
            answer = self._call_gemini(prompt)
            latency_ms = (time.time() - t_start) * 1000
            logger.info(f"Gemini synthesis latency: {latency_ms:.1f}ms")
            return answer
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return f"Error generating answer: {str(e)}"

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def answer_question(self, question: str) -> Dict[str, Any]:
        logger.info(f"Processing question: {question}")
        total_start = time.time()

        try:
            specialists = await self.discover_specialists()

            if not specialists:
                return {
                    "question": question,
                    "answer": "No specialist agents are available.",
                    "error": "No specialists available",
                    "specialist_used": "None",
                    "sources": [],
                    "success": False,
                    "total_time_ms": round((time.time() - total_start) * 1000, 1),
                }

            all_tools = [
                t["name"] for s in specialists for t in s.get("tools", [])
            ]

            should_retrieve = self._should_use_retrieval(question, all_tools)
            logger.info(
                f"Manager decision: "
                f"{'retrieve documents' if should_retrieve else 'answer directly'}"
            )

            if not should_retrieve:
                try:
                    answer = self._call_gemini(question)
                    return {
                        "question": question,
                        "answer": answer,
                        "specialist_used": "None (direct LLM answer)",
                        "sources": [],
                        "success": True,
                        "total_time_ms": round((time.time() - total_start) * 1000, 1),
                    }
                except Exception as e:
                    logger.warning(f"Direct answer failed: {e}, falling back to retrieval")

            specialist = self._select_specialist(question, specialists)

            if not specialist:
                return {
                    "question": question,
                    "answer": "Could not find appropriate specialist.",
                    "error": "No suitable specialist",
                    "specialist_used": "None",
                    "sources": [],
                    "success": False,
                    "total_time_ms": round((time.time() - total_start) * 1000, 1),
                }

            search_results = await self.query_specialist(specialist, question)
            retrieval_ms = round((time.time() - total_start) * 1000, 1)

            if "error" in search_results:
                return {
                    "question": question,
                    "answer": f"Error querying specialist: {search_results['error']}",
                    "error": search_results["error"],
                    "specialist_used": specialist["name"],
                    "sources": [],
                    "success": False,
                    "total_time_ms": round((time.time() - total_start) * 1000, 1),
                }

            synth_start = time.time()
            answer = self._synthesize_answer(question, search_results)
            synth_ms = round((time.time() - synth_start) * 1000, 1)

            total_ms = round((time.time() - total_start) * 1000, 1)
            logger.info(f"Total question processing time: {total_ms}ms")

            self.metrics.record_query(
                QueryMetric(
                    question=question,
                    specialist_used=specialist["name"],
                    retrieval_time=retrieval_ms / 1000,
                    synthesis_time=synth_ms / 1000,
                    total_time=total_ms / 1000,
                    results_count=len(search_results.get("results", [])),
                    success=True,
                )
            )

            return {
                "question": question,
                "answer": answer,
                "specialist_used": specialist["name"],
                "sources": search_results.get("results", [])[:3],
                "success": True,
                "total_time_ms": total_ms,
            }

        except Exception as e:
            logger.error(f"Error processing question: {e}")
            return {
                "question": question,
                "answer": f"An error occurred: {str(e)}",
                "error": str(e),
                "specialist_used": "None",
                "sources": [],
                "success": False,
                "total_time_ms": round((time.time() - total_start) * 1000, 1),
            }
