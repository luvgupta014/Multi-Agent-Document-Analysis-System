"""
Manager Agent — the user-facing orchestrator.

Decides which specialist to route each question to using structured JSON
output from the LLM, then synthesizes a final answer.

Routing actions:
  document_retrieval — use the ChromaDB specialist
  sql_query          — use the SQL specialist
  web_search         — use the web search specialist
  direct             — answer without retrieval

Conversation memory keeps the last N turns in context (config: agents.memory_turns).
LLM calls are wrapped with retry inside each backend (Phase 3).
"""
from __future__ import annotations

import json
from typing import List, Dict, Any, Optional
import time
import httpx
from loguru import logger

from rag_system.mcp_server.models import Message
from rag_system.utils.config import load_config
from rag_system.utils.metrics import MetricsCollector, QueryMetric
from rag_system.utils.security import filter_retrieved_chunks
from rag_system.utils.tracing import Tracer, get_tracer
from rag_system.llm.base import BaseLLM
from rag_system.llm.factory import create_llm
from rag_system.agents.memory import ConversationMemory


SYNTHESIS_SYSTEM_PROMPT = """You are a meticulous technical analyst. Your job is to synthesize a clear, concise answer based only on the provided context and the user's question. You must:
1. Only use information from the provided context passages.
2. If the context does not contain enough information, explicitly state that.
3. At the end of your answer, include a "Sources" section listing the document name and chunk index for each passage you referenced, in the format: [Source: filename, chunk N]."""

ROUTING_PROMPT_TEMPLATE = """\
You are a routing assistant. Given the conversation history and the user's question,
decide which action to take. Reply with ONLY a JSON object — no markdown, no extra text.

Available actions:
  "document_retrieval" — search the internal knowledge base (documents, reports, pipelines, roadmaps)
  "sql_query"          — query the SQL database (structured/tabular data, model metrics, experiments)
  "web_search"         — search the web (current events, external information, general knowledge)
  "direct"             — answer directly without any tool (greetings, math, trivial questions)

{history}Question: {question}

Reply format: {{"action": "<one of the four actions above>", "reason": "<one sentence>"}}"""


class ManagerAgent:

    def __init__(
        self,
        agent_id: str = "manager",
        mcp_url: str = "http://localhost:8000",
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.5-flash",
        llm: Optional[BaseLLM] = None,
        memory_turns: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        prompt_injection_filter: Optional[bool] = None,
        injection_filter_mode: Optional[str] = None,
        tracer: Optional[Tracer] = None,
    ):
        cfg = load_config()
        retrieval_cfg = cfg.get("retrieval", {})
        security_cfg = cfg.get("security", {})
        tracing_cfg = cfg.get("tracing", {})
        agents_cfg = cfg.get("agents", {})

        turns = (
            int(memory_turns)
            if memory_turns is not None
            else int(agents_cfg.get("memory_turns", 5))
        )

        self.agent_id = agent_id
        self.mcp_url = mcp_url
        self.metrics = MetricsCollector()
        self.memory = ConversationMemory(max_turns=turns)
        self.default_top_k: int = int(retrieval_cfg.get("top_k", 5))
        self.similarity_threshold: float = (
            float(similarity_threshold)
            if similarity_threshold is not None
            else float(retrieval_cfg.get("similarity_threshold", 0.3))
        )
        self.prompt_injection_filter: bool = (
            bool(prompt_injection_filter)
            if prompt_injection_filter is not None
            else bool(security_cfg.get("prompt_injection_filter", True))
        )
        self.injection_filter_mode: str = (
            injection_filter_mode
            if injection_filter_mode is not None
            else str(security_cfg.get("filter_mode", "drop"))
        )
        self.tracer: Tracer = (
            tracer
            if tracer is not None
            else get_tracer(enabled=bool(tracing_cfg.get("enabled", True)))
        )

        if llm is not None:
            self._llm = llm
        elif api_key is not None:
            if not api_key:
                raise ValueError("GEMINI_API_KEY is required!")
            from rag_system.llm.gemini import GeminiLLM
            from rag_system.llm.base import LLMConfig
            self._llm = GeminiLLM(LLMConfig(
                provider="gemini",
                model_name=model_name,
                api_key=api_key,
            ))
        else:
            self._llm = create_llm(cfg)

        logger.info(f"Manager agent '{agent_id}' initialised with {self._llm}")

    # ------------------------------------------------------------------
    # LLM call
    # ------------------------------------------------------------------

    def _call_llm(self, prompt: str) -> str:
        return self._llm.generate(prompt)

    # ------------------------------------------------------------------
    # Structured JSON routing
    # ------------------------------------------------------------------

    def _route(self, question: str) -> Dict[str, str]:
        """
        Ask the LLM which action to take. Returns a dict with keys
        'action' and 'reason'. Falls back to 'document_retrieval' on error.
        """
        history_text = self.memory.format_for_prompt()
        prompt = ROUTING_PROMPT_TEMPLATE.format(
            history=history_text,
            question=question,
        )
        try:
            raw = self._call_llm(prompt).strip()
            # Strip markdown code fences if present
            raw = raw.strip("`")
            if raw.startswith("json"):
                raw = raw[4:].strip()
            decision = json.loads(raw)
            action = decision.get("action", "document_retrieval")
            valid = {"document_retrieval", "sql_query", "web_search", "direct"}
            if action not in valid:
                action = "document_retrieval"
            logger.info(f"Routing decision: action={action}, reason={decision.get('reason','')}")
            return {"action": action, "reason": decision.get("reason", "")}
        except Exception as exc:
            logger.warning(f"Routing failed ({exc}), defaulting to document_retrieval")
            return {"action": "document_retrieval", "reason": "routing fallback"}

    # ------------------------------------------------------------------
    # Legacy compatibility — kept for existing tests
    # ------------------------------------------------------------------

    def _should_use_retrieval(self, question: str, available_tools: List[str]) -> bool:
        """Return True if the question needs document retrieval."""
        if "document_retriever" not in available_tools:
            return False
        decision = self._route(question)
        return decision["action"] == "document_retrieval"

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
    # MCP round-trip (generic — works for any tool)
    # ------------------------------------------------------------------

    async def query_specialist(
        self, specialist: Dict, query: str, top_k: int = 5, tool: str = "document_retriever"
    ) -> Dict:
        params: Dict[str, Any] = {"query": query}
        if tool == "document_retriever":
            params["top_k"] = top_k
        message = Message(
            sender=self.agent_id,
            recipient=specialist["agent_id"],
            tool=tool,
            parameters=params,
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
    # Answer synthesis — document retrieval path
    # ------------------------------------------------------------------

    def _synthesize_answer(self, question: str, search_results: Dict, history: str = "") -> str:
        results = search_results.get("results", [])

        # SQL result path — format table rows as context
        if search_results.get("columns") and not results:
            rows = search_results.get("rows", [])
            if rows:
                results = [{"content": str(rows), "metadata": {}, "similarity": 1.0}]

        if not results:
            return "I couldn't find relevant information to answer your question."

        # Enforce retrieval score threshold when similarity scores are present
        scored = [r["similarity"] for r in results if "similarity" in r]
        if scored and self.similarity_threshold > 0 and max(scored) < self.similarity_threshold:
            logger.info(
                f"Max similarity {max(scored):.3f} below threshold {self.similarity_threshold}"
            )
            return "Not found in documents: no retrieved passage met the similarity threshold."

        # Strip or drop chunks containing indirect prompt-injection attempts
        results = filter_retrieved_chunks(
            results,
            enabled=self.prompt_injection_filter,
            mode=self.injection_filter_mode,
        )
        if not results:
            return "I couldn't find safe, relevant information to answer your question."

        context_parts = []
        for i, result in enumerate(results[:5], 1):
            source_file = result.get("metadata", {}).get("filename", "unknown")
            chunk_idx = result.get("metadata", {}).get("chunk_index", "?")
            similarity = result.get("similarity", 0.0)
            text = result.get("content") or result.get("snippet") or ""
            context_parts.append(
                f"Passage {i} [Source: {source_file}, chunk {chunk_idx}] "
                f"(similarity: {similarity:.3f}):\n{text}\n"
            )

        context = "\n".join(context_parts)

        prompt = f"""{SYNTHESIS_SYSTEM_PROMPT}

{history}Context:
{context}

Question: {question}

Provide a clear, concise answer with source citations."""

        logger.info(f"Synthesis prompt length: {len(prompt)} chars")
        logger.debug(f"Full synthesis prompt:\n{prompt}")

        t_start = time.time()
        try:
            answer = self._call_llm(prompt)
            latency_ms = (time.time() - t_start) * 1000
            logger.info(f"LLM synthesis latency: {latency_ms:.1f}ms")
            return answer
        except Exception as e:
            logger.error(f"LLM API error: {e}")
            return f"Error generating answer: {str(e)}"

    def _generate_sql(self, question: str, schema: str = "") -> str:
        """Translate a natural-language question into a single SQLite SELECT statement."""
        prompt = (
            "Convert the question below into a single read-only SQLite SELECT query.\n"
            "Reply with ONLY the raw SQL query — no markdown fences, no explanation.\n\n"
            f"{schema}\n\n"
            f"Question: {question}\n"
            "SQL:"
        )
        raw = self._call_llm(prompt).strip()
        raw = raw.strip("`").strip()
        if raw.lower().startswith("sql"):
            raw = raw[3:].strip()
        return raw.split(";")[0].strip()

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def answer_question(self, question: str, top_k: Optional[int] = None) -> Dict[str, Any]:
        logger.info(f"Processing question: {question}")
        total_start = time.time()
        trace = self.tracer.start_trace("answer_question", input={"question": question})

        def _err(msg: str, error: str) -> Dict:
            self.memory.add("user", question)
            self.memory.add("assistant", msg)
            trace.update(output={"error": error}, metadata={"success": False})
            return {
                "question": question,
                "answer": msg,
                "error": error,
                "specialist_used": "None",
                "sources": [],
                "success": False,
                "total_time_ms": round((time.time() - total_start) * 1000, 1),
            }

        try:
            # 1. Routing decision
            route_span = trace.span("route", input={"question": question})
            routing = self._route(question)
            action = routing["action"]
            route_span.end(output=routing)
            logger.info(f"Manager decision: {action}")

            # 2. Direct answer (no specialist)
            if action == "direct":
                try:
                    history_prefix = self.memory.format_for_prompt()
                    answer = self._call_llm(f"{history_prefix}Question: {question}")
                    self.memory.add("user", question)
                    self.memory.add("assistant", answer)
                    trace.update(output={"answer": answer}, metadata={"action": action})
                    return {
                        "question": question,
                        "answer": answer,
                        "specialist_used": "None (direct LLM answer)",
                        "routing_action": action,
                        "sources": [],
                        "success": True,
                        "total_time_ms": round((time.time() - total_start) * 1000, 1),
                    }
                except Exception as e:
                    logger.warning(f"Direct answer failed: {e}, falling back to retrieval")
                    action = "document_retrieval"

            # 3. Discover specialists
            specialists = await self.discover_specialists()
            if not specialists:
                return _err("No specialist agents are available.", "No specialists available")

            # 4. Select the right specialist for the action
            tool_map = {
                "document_retrieval": "document_retriever",
                "sql_query": "sql_query",
                "web_search": "web_search",
            }
            target_tool = tool_map.get(action, "document_retriever")

            # Find a specialist that has the required tool
            specialist = None
            for s in specialists:
                if any(t["name"] == target_tool for t in s.get("tools", [])):
                    specialist = s
                    break

            # Fall back to document_retriever if no match
            if specialist is None:
                action = "document_retrieval"
                target_tool = "document_retriever"
                specialist = self._select_specialist(question, specialists)

            if not specialist:
                return _err("Could not find appropriate specialist.", "No suitable specialist")

            # 5. Query the specialist
            specialist_query = question
            if target_tool == "sql_query":
                sql_tool = next(
                    (t for t in specialist.get("tools", []) if t.get("name") == "sql_query"),
                    {},
                )
                schema_hint = sql_tool.get("description", "")
                specialist_query = self._generate_sql(question, schema=schema_hint)
                logger.info(f"Generated SQL: {specialist_query}")

            ret_span = trace.span(
                "retrieve",
                input={"question": question, "specialist": specialist["agent_id"], "tool": target_tool},
            )
            search_results = await self.query_specialist(
                specialist,
                specialist_query,
                top_k=top_k or self.default_top_k,
                tool=target_tool,
            )
            retrieval_ms = round((time.time() - total_start) * 1000, 1)
            ret_span.end(
                output={"results_count": len(search_results.get("results", []))},
                metadata={"latency_ms": retrieval_ms},
            )

            if "error" in search_results:
                return _err(
                    f"Error querying specialist: {search_results['error']}",
                    search_results["error"],
                )

            # 6. Synthesise
            synth_start = time.time()
            synth_span = trace.span("synthesize", input={"question": question})
            history_prefix = self.memory.format_for_prompt()
            answer = self._synthesize_answer(question, search_results, history=history_prefix)
            synth_ms = round((time.time() - synth_start) * 1000, 1)
            synth_span.end(output={"answer": answer}, metadata={"latency_ms": synth_ms})

            self.memory.add("user", question)
            self.memory.add("assistant", answer[:200])  # store summary to keep prompt short

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
            trace.update(
                output={"answer": answer},
                metadata={
                    "specialist_used": specialist["name"],
                    "action": action,
                    "total_time_ms": total_ms,
                },
            )

            return {
                "question": question,
                "answer": answer,
                "specialist_used": specialist["name"],
                "routing_action": action,
                "sources": search_results.get("results", [])[:3],
                "success": True,
                "total_time_ms": total_ms,
            }

        except Exception as e:
            logger.error(f"Error processing question: {e}")
            return _err(f"An error occurred: {str(e)}", str(e))
        finally:
            self.tracer.flush()
