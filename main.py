"""
skill-writer-agent — saves completed workflows as reusable learned skills AND
exposes callable capabilities so the planner can explicitly manage skills.

Passive mode: auto-saves any workflow_completed event (when skill_learning_enabled).
Active capabilities (always available):
  write_skill    — explicitly save a plan/description as a named skill
  search_skills  — FTS search over saved skills
  list_skills    — list all saved skills with metadata
  delete_skill   — remove a skill by id
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import uuid
from pathlib import Path

import httpx
import websockets
from websockets.exceptions import ConnectionClosed

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("skill-writer")

AGENT_NAME         = "skill-writer-agent"
HEARTBEAT_INTERVAL = 15
_ID_FILE           = Path(".agent_id")

CAPABILITIES = [
    {
        "name": "write_skill",
        "description": (
            "Save a workflow plan as a reusable named skill. "
            "Pass the goal/task description and optionally a list of steps. "
            "If steps are omitted, the most recently completed workflow matching "
            "the goal will be fetched and saved. "
            "Returns {skill_id, goal, step_count}."
        ),
        "tags": ["skill", "write", "save", "learn", "workflow"],
        "input_schema": {
            "type": "object",
            "properties": {
                "goal": {
                    "type": "string",
                    "description": "Short description of what this skill does.",
                },
                "title": {
                    "type": "string",
                    "description": "Human-readable name for the skill (optional).",
                },
                "steps": {
                    "type": "array",
                    "description": (
                        "Ordered list of workflow steps to save. Each step should have "
                        "name, capability, and input_data. Optional if task_id is given."
                    ),
                    "items": {"type": "object"},
                },
                "task_id": {
                    "type": "string",
                    "description": (
                        "ID of a previously completed workflow to save as a skill. "
                        "If provided, steps are fetched automatically."
                    ),
                },
            },
            "required": ["goal"],
        },
    },
    {
        "name": "search_skills",
        "description": (
            "Full-text search over saved learned skills by goal/description. "
            "Returns up to `limit` matching skills with their plans. "
            "Useful for the planner to check if a skill already exists before "
            "planning from scratch."
        ),
        "tags": ["skill", "search", "find", "lookup"],
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Keywords to search for in skill goals.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum results to return (default: 5).",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "list_skills",
        "description": (
            "List all saved learned skills sorted by creation date (newest first). "
            "Returns id, goal, step_count, use_count, created_at for each skill."
        ),
        "tags": ["skill", "list", "catalog"],
        "input_schema": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of skills to return (default: 50).",
                    "default": 50,
                },
            },
        },
    },
    {
        "name": "delete_skill",
        "description": (
            "Delete a learned skill by its ID. "
            "Returns {deleted: true} on success or an error message."
        ),
        "tags": ["skill", "delete", "remove"],
        "input_schema": {
            "type": "object",
            "properties": {
                "skill_id": {
                    "type": "string",
                    "description": "The UUID of the skill to delete.",
                },
            },
            "required": ["skill_id"],
        },
    },
]


def _stable_agent_id() -> str:
    if _ID_FILE.exists():
        return _ID_FILE.read_text().strip()
    agent_id = str(uuid.uuid4())
    _ID_FILE.write_text(agent_id)
    return agent_id


class SkillWriterAgent:
    def __init__(self, orchestrator_url: str) -> None:
        self._base      = orchestrator_url.rstrip("/")
        self._agent_id  = _stable_agent_id()
        self._enabled   = False
        self._min_steps = 3
        self._http: httpx.AsyncClient | None = None

    # ── Registration ──────────────────────────────────────────────────────────

    async def _register(self) -> None:
        r = await self._http.post(
            f"{self._base}/api/v1/agents/register",
            json={
                "id":          self._agent_id,
                "name":        AGENT_NAME,
                "description": (
                    "Saves completed workflows as reusable learned skills and provides "
                    "capabilities to write, search, list, and delete skills. Also "
                    "passively auto-saves workflows when skill_learning_enabled=true."
                ),
                "capabilities": CAPABILITIES,
                "version":     "1.0.0",
            },
            timeout=10,
        )
        r.raise_for_status()
        self._apply_settings(r.json().get("common_settings", {}))

    # ── Settings ──────────────────────────────────────────────────────────────

    def _apply_settings(self, common: dict) -> None:
        raw = str(common.get("skill_learning_enabled", "false")).lower().strip()
        self._enabled = raw in ("true", "1", "yes")
        try:
            self._min_steps = max(1, int(common.get("skill_learning_min_steps", 3)))
        except (ValueError, TypeError):
            self._min_steps = 3
        logger.info("Skill learning: enabled=%s  min_steps=%d", self._enabled, self._min_steps)

    # ── Capability handlers ───────────────────────────────────────────────────

    async def _cap_write_skill(self, inp: dict) -> dict:
        goal    = inp.get("goal", "").strip()
        title   = inp.get("title", goal)
        steps   = inp.get("steps")
        task_id = inp.get("task_id", "")

        if not goal:
            return {"error": "goal is required"}

        if steps:
            plan = {
                "task_id":     task_id or "",
                "title":       title,
                "description": goal,
                "goal":        goal,
                "steps": [
                    {
                        "step_id":     s.get("step_id", str(uuid.uuid4())),
                        "order":       s.get("order", i),
                        "name":        s.get("name", ""),
                        "description": s.get("description", ""),
                        "capability":  s.get("capability", ""),
                        "input_data":  s.get("input_data", {}),
                    }
                    for i, s in enumerate(steps)
                ],
            }
        elif task_id:
            try:
                r = await self._http.get(
                    f"{self._base}/api/v1/workflows/{task_id}", timeout=10
                )
                r.raise_for_status()
                detail = r.json()
            except Exception as exc:
                return {"error": f"Could not fetch workflow {task_id}: {exc}"}
            plan = {
                "task_id":     task_id,
                "title":       detail.get("title", title),
                "description": detail.get("description", goal),
                "goal":        goal,
                "steps": [
                    {
                        "step_id":     s.get("step_id", ""),
                        "order":       s.get("order", i),
                        "name":        s.get("name", ""),
                        "description": s.get("description", ""),
                        "capability":  s.get("capability", ""),
                        "input_data":  s.get("input_data", {}),
                    }
                    for i, s in enumerate(
                        sorted(detail.get("steps", []), key=lambda x: x.get("order", 0))
                    )
                ],
            }
        else:
            return {"error": "Provide either 'steps' (list) or 'task_id' (completed workflow UUID)"}

        try:
            r = await self._http.post(
                f"{self._base}/api/v1/skills/learned",
                json={"task_id": plan["task_id"], "goal": goal, "plan": plan},
                timeout=10,
            )
            r.raise_for_status()
            skill_id = r.json().get("skill_id", "")
        except Exception as exc:
            return {"error": f"Failed to save skill: {exc}"}

        logger.info("Skill written: id=%s steps=%d goal=%r", skill_id, len(plan["steps"]), goal[:80])
        return {"skill_id": skill_id, "goal": goal, "step_count": len(plan["steps"])}

    async def _cap_search_skills(self, inp: dict) -> dict:
        query = inp.get("query", "").strip()
        limit = int(inp.get("limit", 5))
        if not query:
            return {"error": "query is required"}
        try:
            r = await self._http.get(
                f"{self._base}/api/v1/skills/learned/search",
                params={"q": query, "limit": limit},
                timeout=10,
            )
            r.raise_for_status()
            results = r.json().get("results", [])
        except Exception as exc:
            return {"error": f"Search failed: {exc}"}
        return {"results": results, "count": len(results)}

    async def _cap_list_skills(self, inp: dict) -> dict:
        limit = int(inp.get("limit", 50))
        try:
            r = await self._http.get(
                f"{self._base}/api/v1/skills/learned",
                params={"limit": limit},
                timeout=10,
            )
            r.raise_for_status()
            skills = r.json().get("skills", [])
        except Exception as exc:
            return {"error": f"List failed: {exc}"}
        summary = [
            {
                "skill_id":   s.get("id", ""),
                "goal":       s.get("goal", ""),
                "step_count": s.get("step_count", 0),
                "use_count":  s.get("use_count", 0),
                "created_at": s.get("created_at", ""),
            }
            for s in skills
        ]
        return {"skills": summary, "count": len(summary)}

    async def _cap_delete_skill(self, inp: dict) -> dict:
        skill_id = inp.get("skill_id", "").strip()
        if not skill_id:
            return {"error": "skill_id is required"}
        try:
            r = await self._http.delete(
                f"{self._base}/api/v1/skills/learned/{skill_id}", timeout=10
            )
            if r.status_code == 404:
                return {"error": f"Skill {skill_id} not found"}
            r.raise_for_status()
        except Exception as exc:
            return {"error": f"Delete failed: {exc}"}
        logger.info("Skill deleted: id=%s", skill_id)
        return {"deleted": True, "skill_id": skill_id}

    _CAPABILITY_MAP = {
        "write_skill":   _cap_write_skill,
        "search_skills": _cap_search_skills,
        "list_skills":   _cap_list_skills,
        "delete_skill":  _cap_delete_skill,
    }

    # ── Message dispatch ──────────────────────────────────────────────────────

    async def _dispatch(self, raw: str, ws) -> None:
        msg      = json.loads(raw)
        msg_type = msg.get("type", "")

        if msg_type == "settings_push":
            self._apply_settings(msg.get("payload", {}).get("settings", {}))
            return

        if msg_type == "heartbeat_ack":
            return

        if msg_type == "task_request":
            asyncio.create_task(self._handle_task_request(msg, ws))
            return

        if msg_type == "workflow_event":
            payload = msg.get("payload", {})
            if payload.get("event") == "workflow_completed" and self._enabled:
                task_id = payload.get("task_id", "")
                goal    = payload.get("goal", "")
                if task_id:
                    asyncio.create_task(self._auto_save(task_id, goal))

    async def _handle_task_request(self, msg: dict, ws) -> None:
        payload    = msg.get("payload", {})
        capability = payload.get("capability", "")
        input_data = payload.get("input_data", {})
        task_id    = payload.get("task_id", "")

        handler = self._CAPABILITY_MAP.get(capability)
        if handler is None:
            result = {"error": f"Unknown capability: {capability}"}
            status = "failed"
        else:
            try:
                result = await handler(self, input_data)
                status = "failed" if "error" in result else "completed"
            except Exception as exc:
                logger.exception("Error in capability %s", capability)
                result = {"error": str(exc)}
                status = "failed"

        try:
            await ws.send(json.dumps({
                "type":      "task_response",
                "sender_id": self._agent_id,
                "payload": {
                    "task_id":    task_id,
                    "capability": capability,
                    "status":     status,
                    "result":     result,
                },
            }))
        except Exception as exc:
            logger.warning("Failed to send task_response: %s", exc)

    # ── Passive auto-save (workflow_completed events) ─────────────────────────

    async def _auto_save(self, task_id: str, goal: str) -> None:
        try:
            r = await self._http.get(
                f"{self._base}/api/v1/workflows/{task_id}", timeout=10
            )
            r.raise_for_status()
            detail = r.json()
        except Exception as exc:
            logger.warning("Auto-save: could not fetch workflow %s: %s", task_id, exc)
            return

        steps = detail.get("steps", [])
        if len(steps) < self._min_steps:
            logger.debug("Auto-save skipped: %d steps < min %d", len(steps), self._min_steps)
            return

        plan = {
            "task_id":     task_id,
            "title":       detail.get("title", ""),
            "description": detail.get("description", ""),
            "goal":        goal or detail.get("goal", ""),
            "steps": [
                {
                    "step_id":     s.get("step_id", ""),
                    "order":       s.get("order", 0),
                    "name":        s.get("name", ""),
                    "description": s.get("description", ""),
                    "capability":  s.get("capability", ""),
                    "input_data":  s.get("input_data", {}),
                }
                for s in sorted(steps, key=lambda x: x.get("order", 0))
            ],
        }

        try:
            r2 = await self._http.post(
                f"{self._base}/api/v1/skills/learned",
                json={"task_id": task_id, "goal": plan["goal"], "plan": plan},
                timeout=10,
            )
            r2.raise_for_status()
            logger.info(
                "Auto-saved skill: task=%s steps=%d goal=%r",
                task_id, len(steps), plan["goal"][:80],
            )
        except Exception as exc:
            logger.error("Auto-save failed for %s: %s", task_id, exc)

    # ── WebSocket loop ────────────────────────────────────────────────────────

    async def _heartbeat(self, ws) -> None:
        while True:
            await asyncio.sleep(HEARTBEAT_INTERVAL)
            await ws.send(json.dumps({
                "type": "heartbeat",
                "sender_id": self._agent_id,
                "payload": {},
            }))

    async def _run_session(self, ws) -> None:
        logger.info("WebSocket session active")
        hb_task   = asyncio.create_task(self._heartbeat(ws))
        recv_task = asyncio.create_task(self._recv_loop(ws))
        try:
            done, pending = await asyncio.wait(
                [hb_task, recv_task], return_when=asyncio.FIRST_EXCEPTION
            )
            for t in pending:
                t.cancel()
            for t in done:
                if not t.cancelled() and t.exception():
                    raise t.exception()
        finally:
            hb_task.cancel()
            recv_task.cancel()

    async def _recv_loop(self, ws) -> None:
        async for raw in ws:
            await self._dispatch(raw, ws)

    async def _connect_loop(self) -> None:
        ws_url  = self._base.replace("http", "ws") + f"/ws/{self._agent_id}"
        backoff = 1.0
        while True:
            try:
                async with websockets.connect(ws_url) as ws:
                    backoff = 1.0
                    await self._run_session(ws)
            except ConnectionClosed as exc:
                code = exc.rcvd.code if exc.rcvd else None
                if code == 4004:
                    logger.warning("Unknown agent_id (4004) — re-registering…")
                    try:
                        await self._register()
                    except Exception as reg_exc:
                        logger.error("Re-registration failed: %s", reg_exc)
                else:
                    logger.warning("WS closed (code=%s) — retry in %.0fs", code, backoff)
            except Exception as exc:
                logger.warning("WS error (%s) — retry in %.0fs", exc, backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60.0)

    async def run(self) -> None:
        async with httpx.AsyncClient() as http:
            self._http = http
            await self._register()
            logger.info(
                "skill-writer-agent registered (id=%s, capabilities=%d, auto-learn=%s)",
                self._agent_id, len(CAPABILITIES), self._enabled,
            )
            await self._connect_loop()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--orchestrator-url",
        default=os.environ.get("ORCHESTRATOR_URL", "http://localhost:8000"),
    )
    args = parser.parse_args()
    asyncio.run(SkillWriterAgent(args.orchestrator_url).run())


if __name__ == "__main__":
    main()
