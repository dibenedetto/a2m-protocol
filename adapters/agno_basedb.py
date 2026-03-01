"""
A2M -- AgentToMemory Protocol
Agno adapter: A2MAgnoBaseDb

Complete implementation of agno.db.base.BaseDb backed by the A2M REST API.
Every database feature (sessions, memory, knowledge, evals, traces, spans,
cultural knowledge, learnings, schema versioning) maps to A2M entries.

A2M storage layout
------------------
  __schema__/{table}          type=working,    tags=[agno-schema]
  session/{stype}/{sid}       type=working,    tags=[agno-session, stype:*, user:*, component:*]
  memory/{mid}                type=semantic,   tags=[agno-memory, user:*, <topics>]
  knowledge/{id}              type=semantic,   tags=[agno-knowledge, linked:*]
  eval/{run_id}               type=procedural, tags=[agno-eval, agent:*, model:*, etype:*]
  trace/{trace_id}            type=working,    tags=[agno-trace, run:*, session:*, user:*, agent:*, team:*, workflow:*, status:*]
  span/{span_id}              type=working,    tags=[agno-span, trace:*, parent:*]
  culture/{id}                type=semantic,   tags=[agno-culture, agent:*, team:*]
  learning/{id}               type=procedural, tags=[agno-learning, ltype:*, user:*, agent:*, team:*, session:*]

Embeddings are always caller-supplied (A2M never generates them).
Supply `embed_fn` at construction to enable semantic search over memories
and cultural-knowledge entries.

Install requirement:
    pip install agno

Usage:
    from adapters.agno_basedb import A2MAgnoBaseDb
    from client.client import A2MClient
    from agno.memory import MemoryManager

    client = A2MClient("http://localhost:8765", namespace="myapp/agent-0")
    db     = A2MAgnoBaseDb(client=client)
    manager = MemoryManager(db=db)
"""

from __future__ import annotations

import dataclasses
from datetime import date, datetime
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from uuid import uuid4

try:
    from agno.db.base import BaseDb, SessionType
    from agno.db.schemas import UserMemory
    from agno.db.schemas.culture import CulturalKnowledge
    from agno.db.schemas.evals import EvalFilterType, EvalRunRecord, EvalType
    from agno.db.schemas.knowledge import KnowledgeRow
    from agno.session.agent import AgentSession
    from agno.session.team import TeamSession
    from agno.session.workflow import WorkflowSession
except ImportError as exc:
    raise ImportError(
        "Agno is required for A2MAgnoBaseDb. Install with: pip install agno"
    ) from exc

# Tracing schemas are optional (they live under agno.tracing which may not
# always be installed).  We import them at runtime so isinstance() checks work,
# but fall back gracefully if the sub-package is absent.
try:
    from agno.tracing.schemas import Span, Trace
    _HAS_TRACING = True
except ImportError:
    _HAS_TRACING = False
    Span = Any   # type: ignore[assignment,misc]
    Trace = Any  # type: ignore[assignment,misc]

if TYPE_CHECKING:
    pass  # kept for future annotations

from client.client import A2MClient


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def _to_dict(obj: Any) -> dict:
    """Serialise any agno model or plain dict to a JSON-safe dict."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    if isinstance(obj, dict):
        return obj
    return vars(obj)


def _dc_from_dict(cls, d: dict):
    """Reconstruct a dataclass from a stored dict, ignoring unknown keys."""
    known = {f.name for f in dataclasses.fields(cls)}
    return cls(**{k: v for k, v in d.items() if k in known})


_SESSION_CLASS: dict[SessionType, type] = {
    SessionType.AGENT:    AgentSession,
    SessionType.TEAM:     TeamSession,
    SessionType.WORKFLOW: WorkflowSession,
}


# ---------------------------------------------------------------------------
# Memory serialisation (unchanged from original adapter)
# ---------------------------------------------------------------------------

def _to_a2m_value(memory: UserMemory) -> dict:
    return {
        "memory":     memory.memory,
        "memory_id":  memory.memory_id,
        "user_id":    memory.user_id,
        "agent_id":   memory.agent_id,
        "team_id":    memory.team_id,
        "topics":     memory.topics or [],
        "input":      memory.input,
        "feedback":   memory.feedback,
        "created_at": memory.created_at,
        "updated_at": memory.updated_at,
    }


def _from_a2m_entry(entry: dict) -> UserMemory:
    v = entry.get("value", {})
    if isinstance(v, str):
        return UserMemory(memory=v, memory_id=entry.get("key"))
    return UserMemory(
        memory=v.get("memory", ""),
        memory_id=v.get("memory_id") or entry.get("key"),
        user_id=v.get("user_id"),
        agent_id=v.get("agent_id"),
        team_id=v.get("team_id"),
        topics=v.get("topics") or [],
        input=v.get("input"),
        feedback=v.get("feedback"),
        created_at=v.get("created_at"),
        updated_at=v.get("updated_at"),
    )


# ---------------------------------------------------------------------------
# Main adapter
# ---------------------------------------------------------------------------

class A2MAgnoBaseDb(BaseDb):
    """
    Complete agno.db.base.BaseDb implementation backed by the A2M REST API.

    All 47 abstract methods are implemented.  Supply an `embed_fn` to enable
    vector search over memories and cultural-knowledge entries.
    """

    def __init__(
        self,
        client:   A2MClient,
        embed_fn: Optional[Callable[[str], list[float]]] = None,
    ) -> None:
        super().__init__()
        self.client   = client
        self.embed_fn = embed_fn

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _write(
        self,
        key:       str,
        type:      str,
        tags:      list[str],
        value:     Any,
        embedding: Optional[list[float]] = None,
    ) -> None:
        self.client.write(
            key=key,
            type=type,
            value=value,
            embedding=embedding,
            meta={"source_framework": "agno", "tags": tags},
        )

    def _all_entries(self, type: str, tags: list[str]) -> list[dict]:
        """Fetch every matching entry, paginating in batches of 500."""
        results, offset = [], 0
        while True:
            resp  = self.client.list(type=type, tags=tags, limit=500, offset=offset)
            batch = resp["entries"]
            results.extend(batch)
            if len(batch) < 500:
                break
            offset += 500
        return results

    def _page(
        self,
        type:   str,
        tags:   list[str],
        limit:  Optional[int],
        page:   Optional[int],
    ) -> list[dict]:
        lim    = min(limit or 50, 500)
        offset = ((page or 1) - 1) * lim
        return self.client.list(type=type, tags=tags, limit=lim, offset=offset)["entries"]

    # -----------------------------------------------------------------------
    # Schema versioning
    # -----------------------------------------------------------------------

    # Map Agno logical table names to the A2M (type, tag) used for that
    # category.  When Agno asks "does agno_sessions exist?", we probe the
    # A2M server with a targeted list query for that category.
    _TABLE_MAP: dict[str, tuple[str, str]] = {
        "agno_sessions":          ("working",    "agno-session"),
        "agno_culture":           ("semantic",   "agno-culture"),
        "agno_memories":          ("semantic",   "agno-memory"),
        "agno_metrics":           ("working",    "agno-session"),
        "agno_eval_runs":         ("procedural", "agno-eval"),
        "agno_knowledge":         ("semantic",   "agno-knowledge"),
        "agno_traces":            ("working",    "agno-trace"),
        "agno_spans":             ("working",    "agno-span"),
        "agno_schema_versions":   ("working",    "agno-schema"),
        "agno_learnings":         ("procedural", "agno-learning"),
        "agno_components":        ("working",    "agno-component"),
        "agno_component_configs": ("working",    "agno-component-config"),
        "agno_component_links":   ("working",    "agno-component-link"),
        "agno_schedules":         ("working",    "agno-schedule"),
        "agno_schedule_runs":     ("working",    "agno-schedule-run"),
        "agno_approvals":         ("working",    "agno-approval"),
    }

    def table_exists(self, table_name: str) -> bool:
        """
        Check that the A2M server can serve entries for the given Agno table.

        A2M maps every Agno table to a (type, tag) pair in its single
        relational ``entries`` table.  This method issues a lightweight
        list query for the matching category; if the server responds, the
        "table" is considered to exist.  An unknown ``table_name`` falls
        back to a health check on the relational backend.
        """
        try:
            mapping = self._TABLE_MAP.get(table_name)
            if mapping:
                a2m_type, tag = mapping
                self.client.list(type=a2m_type, tags=[tag], limit=1)
            else:
                info = self.client.health()
                return info.get("relational", {}).get("ok", False)
            return True
        except Exception:
            return False

    def get_latest_schema_version(self, table_name: str) -> Optional[str]:
        entry = self.client.read(f"__schema__/{table_name}")
        return entry["value"]["version"] if entry else None

    def upsert_schema_version(self, table_name: str, version: str) -> None:
        self._write(
            key=f"__schema__/{table_name}",
            type="working",
            tags=["agno-schema"],
            value={"table": table_name, "version": version},
        )

    # -----------------------------------------------------------------------
    # Memory
    # -----------------------------------------------------------------------

    def _user_tags(self, user_id: Optional[str]) -> list[str]:
        tags = ["agno-memory"]
        if user_id:
            tags.append(f"user:{user_id}")
        return tags

    def _write_memory(self, memory: UserMemory) -> None:
        mid       = memory.memory_id or memory.memory
        embedding = self.embed_fn(memory.memory) if self.embed_fn else None
        tags      = self._user_tags(memory.user_id)
        if memory.topics:
            tags.extend(memory.topics)
        self._write(
            key=f"memory/{mid}",
            type="semantic",
            tags=tags,
            value=_to_a2m_value(memory),
            embedding=embedding,
        )

    def upsert_user_memory(
        self,
        memory:      UserMemory,
        deserialize: Optional[bool] = True,
    ) -> Optional[Union[UserMemory, Dict[str, Any]]]:
        self._write_memory(memory)
        return memory if deserialize else _to_a2m_value(memory)

    def upsert_memories(
        self,
        memories:            List[UserMemory],
        deserialize:         Optional[bool] = True,
        preserve_updated_at: bool           = False,
    ) -> List[Union[UserMemory, Dict[str, Any]]]:
        for m in memories:
            self._write_memory(m)
        return memories if deserialize else [_to_a2m_value(m) for m in memories]

    def get_user_memories(
        self,
        user_id:        Optional[str]       = None,
        agent_id:       Optional[str]       = None,
        team_id:        Optional[str]       = None,
        topics:         Optional[List[str]] = None,
        search_content: Optional[str]       = None,
        limit:          Optional[int]       = None,
        page:           Optional[int]       = None,
        sort_by:        Optional[str]       = None,
        sort_order:     Optional[str]       = None,
        deserialize:    Optional[bool]      = True,
    ) -> Union[List[UserMemory], Tuple[List[Dict[str, Any]], int]]:
        tags    = self._user_tags(user_id)
        entries = self._page(type="semantic", tags=tags, limit=limit, page=page)
        memories = [_from_a2m_entry(e) for e in entries]
        if agent_id:
            memories = [m for m in memories if m.agent_id == agent_id]
        if team_id:
            memories = [m for m in memories if m.team_id == team_id]
        if topics:
            ts = set(topics)
            memories = [m for m in memories if ts & set(m.topics or [])]
        return memories

    def get_user_memory(
        self,
        memory_id:   str,
        deserialize: Optional[bool] = True,
        user_id:     Optional[str]  = None,
    ) -> Optional[Union[UserMemory, Dict[str, Any]]]:
        entry = self.client.read(f"memory/{memory_id}")
        if entry is None:
            return None
        m = _from_a2m_entry(entry)
        return m if deserialize else _to_a2m_value(m)

    def delete_user_memory(
        self,
        memory_id: str,
        user_id:   Optional[str] = None,
    ) -> None:
        try:
            self.client.delete(f"memory/{memory_id}")
        except Exception:
            pass

    def delete_user_memories(
        self,
        memory_ids: List[str],
        user_id:    Optional[str] = None,
    ) -> None:
        for mid in memory_ids:
            self.delete_user_memory(mid, user_id=user_id)

    def clear_memories(self) -> None:
        self.client.delete_bulk(type="semantic", tags=["agno-memory"])

    def get_all_memory_topics(self, user_id: Optional[str] = None) -> List[str]:
        topics: set[str] = set()
        for entry in self._all_entries(type="semantic", tags=self._user_tags(user_id)):
            topics.update(_from_a2m_entry(entry).topics or [])
        return sorted(topics)

    def get_user_memory_stats(
        self,
        limit:   Optional[int] = None,
        page:    Optional[int] = None,
        user_id: Optional[str] = None,
    ) -> Tuple[List[Dict[str, Any]], int]:
        memories = self.get_user_memories(user_id=user_id, limit=limit, page=page)
        stats = [
            {
                "memory_id": m.memory_id,
                "memory":    m.memory,
                "topics":    m.topics or [],
                "user_id":   m.user_id,
            }
            for m in memories
        ]
        return stats, len(stats)

    # -----------------------------------------------------------------------
    # Semantic search (not in BaseDb; called by MemoryManager)
    # -----------------------------------------------------------------------

    def search_user_memories(
        self,
        embedding: list[float],
        user_id:   Optional[str] = None,
        limit:     int           = 5,
    ) -> List[UserMemory]:
        results = self.client.query(
            embedding=embedding,
            type="semantic",
            tags=self._user_tags(user_id),
            top_k=limit,
        )
        return [_from_a2m_entry(r["entry"]) for r in results]

    # -----------------------------------------------------------------------
    # Sessions
    # -----------------------------------------------------------------------

    def _session_key(self, session_id: str, session_type: SessionType) -> str:
        return f"session/{session_type.value}/{session_id}"

    def _session_tags(
        self,
        session_type: SessionType,
        user_id:      Optional[str] = None,
        component_id: Optional[str] = None,
    ) -> list[str]:
        tags = ["agno-session", f"stype:{session_type.value}"]
        if user_id:      tags.append(f"user:{user_id}")
        if component_id: tags.append(f"component:{component_id}")
        return tags

    def _deserialize_session(
        self,
        value:        dict,
        session_type: SessionType,
        deserialize:  Optional[bool],
    ) -> Any:
        if not deserialize:
            return value
        cls = _SESSION_CLASS.get(session_type)
        return _dc_from_dict(cls, value) if cls else value

    def upsert_session(
        self,
        session:     Any,
        deserialize: Optional[bool] = True,
    ) -> Any:
        if isinstance(session, AgentSession):
            stype = SessionType.AGENT
        elif isinstance(session, TeamSession):
            stype = SessionType.TEAM
        else:
            stype = SessionType.WORKFLOW
        tags = self._session_tags(stype, getattr(session, "user_id", None))
        name = (
            getattr(session, "session_name", None)
            or (session.session_data or {}).get("session_name") if hasattr(session, "session_data") else None
        )
        if name:
            tags.append(f"name:{name}")
        d = _to_dict(session)
        self._write(
            key=self._session_key(session.session_id, stype),
            type="working",
            tags=tags,
            value=d,
        )
        return session if deserialize else d

    def upsert_sessions(
        self,
        sessions:            List[Any],
        deserialize:         Optional[bool] = True,
        preserve_updated_at: bool           = False,
    ) -> List[Any]:
        return [self.upsert_session(s, deserialize=deserialize) for s in sessions]

    def get_session(
        self,
        session_id:   str,
        session_type: SessionType,
        user_id:      Optional[str]  = None,
        deserialize:  Optional[bool] = True,
    ) -> Any:
        entry = self.client.read(self._session_key(session_id, session_type))
        if entry is None:
            return None
        return self._deserialize_session(entry["value"], session_type, deserialize)

    def get_sessions(
        self,
        session_type:    SessionType,
        user_id:         Optional[str] = None,
        component_id:    Optional[str] = None,
        session_name:    Optional[str] = None,
        start_timestamp: Optional[int] = None,
        end_timestamp:   Optional[int] = None,
        limit:           Optional[int] = None,
        page:            Optional[int] = None,
        sort_by:         Optional[str] = None,
        sort_order:      Optional[str] = None,
        deserialize:     Optional[bool] = True,
    ) -> Any:
        tags    = self._session_tags(session_type, user_id, component_id)
        entries = self._page(type="working", tags=tags, limit=limit, page=page)
        results = []
        for e in entries:
            v = e["value"]
            # client-side: session_name filter
            if session_name:
                stored_name = (v.get("session_data") or {}).get("session_name")
                if stored_name != session_name:
                    continue
            # client-side: timestamp filters on created_at
            if start_timestamp or end_timestamp:
                created = v.get("created_at")
                if isinstance(created, (int, float)):
                    if start_timestamp and created < start_timestamp:
                        continue
                    if end_timestamp and created > end_timestamp:
                        continue
            results.append(self._deserialize_session(v, session_type, deserialize))
        return results

    def delete_session(
        self,
        session_id:   str,
        user_id:      Optional[str] = None,
    ) -> bool:
        # We need the session_type to form the key; try all types.
        for stype in SessionType:
            try:
                self.client.delete(self._session_key(session_id, stype))
                return True
            except Exception:
                pass
        return False

    def delete_sessions(
        self,
        session_ids: List[str],
        user_id:     Optional[str] = None,
    ) -> None:
        for sid in session_ids:
            self.delete_session(sid, user_id=user_id)

    def rename_session(
        self,
        session_id:   str,
        session_type: SessionType,
        session_name: str,
        user_id:      Optional[str]  = None,
        deserialize:  Optional[bool] = True,
    ) -> Any:
        key   = self._session_key(session_id, session_type)
        entry = self.client.read(key)
        if entry is None:
            return None
        v = entry["value"]
        if v.get("session_data") is None:
            v["session_data"] = {}
        v["session_data"]["session_name"] = session_name
        tags = self._session_tags(session_type, user_id)
        tags.append(f"name:{session_name}")
        self._write(key=key, type="working", tags=tags, value=v)
        return self._deserialize_session(v, session_type, deserialize)

    # -----------------------------------------------------------------------
    # Metrics
    # -----------------------------------------------------------------------

    def get_metrics(
        self,
        starting_date: Optional[date] = None,
        ending_date:   Optional[date] = None,
    ) -> Tuple[List[Dict[str, Any]], Optional[int]]:
        """
        Aggregate basic usage metrics from stored sessions and traces.
        Returns (list_of_metric_dicts, total).
        """
        metrics: list[dict] = []
        for stype in SessionType:
            entries = self._all_entries(type="working", tags=["agno-session", f"stype:{stype.value}"])
            for e in entries:
                v = e["value"]
                ts = v.get("created_at")
                if starting_date and isinstance(ts, str) and ts[:10] < str(starting_date):
                    continue
                if ending_date and isinstance(ts, str) and ts[:10] > str(ending_date):
                    continue
                metrics.append({
                    "session_id":   v.get("session_id"),
                    "session_type": stype.value,
                    "created_at":   ts,
                })
        return metrics, len(metrics)

    def calculate_metrics(self) -> Optional[Any]:
        """Trigger metric aggregation; not applicable for A2M -- returns None."""
        return None

    # -----------------------------------------------------------------------
    # Knowledge
    # -----------------------------------------------------------------------

    def _knowledge_key(self, id: str) -> str:
        return f"knowledge/{id}"

    def upsert_knowledge_content(self, knowledge_row: KnowledgeRow) -> Any:
        tags = ["agno-knowledge"]
        if knowledge_row.linked_to:
            tags.append(f"linked:{knowledge_row.linked_to}")
        self._write(
            key=self._knowledge_key(knowledge_row.id),
            type="semantic",
            tags=tags,
            value=_to_dict(knowledge_row),
        )
        return knowledge_row

    def get_knowledge_content(self, id: str) -> Optional[KnowledgeRow]:
        entry = self.client.read(self._knowledge_key(id))
        if entry is None:
            return None
        return KnowledgeRow.model_validate(entry["value"])

    def get_knowledge_contents(
        self,
        limit:      Optional[int] = None,
        page:       Optional[int] = None,
        sort_by:    Optional[str] = None,
        sort_order: Optional[str] = None,
        linked_to:  Optional[str] = None,
    ) -> Tuple[List[KnowledgeRow], int]:
        tags = ["agno-knowledge"]
        if linked_to:
            tags.append(f"linked:{linked_to}")
        entries = self._page(type="semantic", tags=tags, limit=limit, page=page)
        rows = [KnowledgeRow.model_validate(e["value"]) for e in entries]
        return rows, len(rows)

    def delete_knowledge_content(self, id: str) -> None:
        try:
            self.client.delete(self._knowledge_key(id))
        except Exception:
            pass

    # -----------------------------------------------------------------------
    # Evals
    # -----------------------------------------------------------------------

    def _eval_key(self, run_id: str) -> str:
        return f"eval/{run_id}"

    def _eval_tags(self, eval_run: EvalRunRecord) -> list[str]:
        tags = ["agno-eval"]
        if eval_run.agent_id:    tags.append(f"agent:{eval_run.agent_id}")
        if eval_run.model_id:    tags.append(f"model:{eval_run.model_id}")
        if eval_run.team_id:     tags.append(f"team:{eval_run.team_id}")
        if eval_run.workflow_id: tags.append(f"workflow:{eval_run.workflow_id}")
        if eval_run.eval_type:   tags.append(f"etype:{eval_run.eval_type}")
        return tags

    def create_eval_run(self, eval_run: EvalRunRecord) -> Optional[EvalRunRecord]:
        if not eval_run.run_id:
            eval_run.run_id = str(uuid4())
        self._write(
            key=self._eval_key(eval_run.run_id),
            type="procedural",
            tags=self._eval_tags(eval_run),
            value=_to_dict(eval_run),
        )
        return eval_run

    def get_eval_run(
        self,
        eval_run_id: str,
        deserialize: Optional[bool] = True,
    ) -> Optional[Union[EvalRunRecord, Dict[str, Any]]]:
        entry = self.client.read(self._eval_key(eval_run_id))
        if entry is None:
            return None
        if deserialize:
            return EvalRunRecord.model_validate(entry["value"])
        return entry["value"]

    def get_eval_runs(
        self,
        limit:       Optional[int]            = None,
        page:        Optional[int]            = None,
        sort_by:     Optional[str]            = None,
        sort_order:  Optional[str]            = None,
        agent_id:    Optional[str]            = None,
        team_id:     Optional[str]            = None,
        workflow_id: Optional[str]            = None,
        model_id:    Optional[str]            = None,
        filter_type: Optional[EvalFilterType] = None,
        eval_type:   Optional[List[EvalType]] = None,
        deserialize: Optional[bool]           = True,
    ) -> Union[List[EvalRunRecord], Tuple[List[Dict[str, Any]], int]]:
        tags = ["agno-eval"]
        if agent_id:    tags.append(f"agent:{agent_id}")
        if model_id:    tags.append(f"model:{model_id}")
        if team_id:     tags.append(f"team:{team_id}")
        if workflow_id: tags.append(f"workflow:{workflow_id}")
        entries = self._page(type="procedural", tags=tags, limit=limit, page=page)
        # client-side eval_type filter (list of EvalType values)
        if eval_type:
            etypes = {e.value if hasattr(e, "value") else e for e in eval_type}
            entries = [
                e for e in entries
                if str(e["value"].get("eval_type")) in etypes
            ]
        if deserialize:
            return [EvalRunRecord.model_validate(e["value"]) for e in entries]
        rows = [e["value"] for e in entries]
        return rows, len(rows)

    def rename_eval_run(
        self,
        eval_run_id: str,
        name:        str,
        deserialize: Optional[bool] = True,
    ) -> Optional[Union[EvalRunRecord, Dict[str, Any]]]:
        entry = self.client.read(self._eval_key(eval_run_id))
        if entry is None:
            return None
        v       = entry["value"]
        v["name"] = name
        er      = EvalRunRecord.model_validate(v)
        self._write(
            key=self._eval_key(eval_run_id),
            type="procedural",
            tags=self._eval_tags(er),
            value=v,
        )
        return er if deserialize else v

    def delete_eval_runs(self, eval_run_ids: List[str]) -> None:
        for rid in eval_run_ids:
            try:
                self.client.delete(self._eval_key(rid))
            except Exception:
                pass

    # -----------------------------------------------------------------------
    # Traces
    # -----------------------------------------------------------------------

    def _trace_key(self, trace_id: str) -> str:
        return f"trace/{trace_id}"

    def _trace_tags(self, trace: Any) -> list[str]:
        tags = ["agno-trace"]
        for attr, prefix in [
            ("run_id",      "run"),
            ("session_id",  "session"),
            ("user_id",     "user"),
            ("agent_id",    "agent"),
            ("team_id",     "team"),
            ("workflow_id", "workflow"),
            ("status",      "status"),
        ]:
            v = getattr(trace, attr, None)
            if v:
                tags.append(f"{prefix}:{v}")
        return tags

    def upsert_trace(self, trace: Any) -> None:
        tid = getattr(trace, "trace_id", None) or str(uuid4())
        self._write(
            key=self._trace_key(tid),
            type="working",
            tags=self._trace_tags(trace),
            value=_to_dict(trace),
        )

    def get_trace(
        self,
        trace_id:   Optional[str] = None,
        run_id:     Optional[str] = None,
        session_id: Optional[str] = None,
        user_id:    Optional[str] = None,
        agent_id:   Optional[str] = None,
    ) -> Any:
        if trace_id:
            entry = self.client.read(self._trace_key(trace_id))
            return entry["value"] if entry else None
        # search by secondary field
        tags = ["agno-trace"]
        if run_id:     tags.append(f"run:{run_id}")
        if session_id: tags.append(f"session:{session_id}")
        if user_id:    tags.append(f"user:{user_id}")
        if agent_id:   tags.append(f"agent:{agent_id}")
        entries = self.client.list(type="working", tags=tags, limit=1)["entries"]
        return entries[0]["value"] if entries else None

    def get_traces(
        self,
        run_id:     Optional[str]      = None,
        session_id: Optional[str]      = None,
        user_id:    Optional[str]      = None,
        agent_id:   Optional[str]      = None,
        team_id:    Optional[str]      = None,
        workflow_id: Optional[str]     = None,
        status:     Optional[str]      = None,
        start_time: Optional[datetime] = None,
        end_time:   Optional[datetime] = None,
        limit:      Optional[int]      = 20,
        page:       Optional[int]      = 1,
    ) -> Tuple[List[Any], int]:
        tags = ["agno-trace"]
        if run_id:     tags.append(f"run:{run_id}")
        if session_id: tags.append(f"session:{session_id}")
        if user_id:    tags.append(f"user:{user_id}")
        if agent_id:   tags.append(f"agent:{agent_id}")
        if team_id:    tags.append(f"team:{team_id}")
        if workflow_id: tags.append(f"workflow:{workflow_id}")
        if status:     tags.append(f"status:{status}")
        entries = self._page(type="working", tags=tags, limit=limit, page=page)
        traces  = [e["value"] for e in entries]
        # client-side datetime range filter
        if start_time or end_time:
            def _in_range(v: dict) -> bool:
                ts = v.get("created_at") or v.get("start_time")
                if not isinstance(ts, str):
                    return True
                try:
                    dt = datetime.fromisoformat(ts)
                except ValueError:
                    return True
                if start_time and dt < start_time:
                    return False
                if end_time and dt > end_time:
                    return False
                return True
            traces = [t for t in traces if _in_range(t)]
        return traces, len(traces)

    def get_trace_stats(
        self,
        user_id:    Optional[str]      = None,
        agent_id:   Optional[str]      = None,
        team_id:    Optional[str]      = None,
        workflow_id: Optional[str]     = None,
        start_time: Optional[datetime] = None,
        end_time:   Optional[datetime] = None,
        limit:      Optional[int]      = 20,
        page:       Optional[int]      = 1,
    ) -> Tuple[List[Dict[str, Any]], int]:
        traces, total = self.get_traces(
            user_id=user_id,
            agent_id=agent_id,
            team_id=team_id,
            workflow_id=workflow_id,
            start_time=start_time,
            end_time=end_time,
            limit=limit,
            page=page,
        )
        stats = [
            {
                "trace_id":    t.get("trace_id"),
                "status":      t.get("status"),
                "duration_ms": t.get("duration_ms"),
                "error_count": t.get("error_count"),
                "total_spans": t.get("total_spans"),
                "agent_id":    t.get("agent_id"),
                "created_at":  t.get("created_at"),
            }
            for t in traces
        ]
        return stats, total

    # -----------------------------------------------------------------------
    # Spans
    # -----------------------------------------------------------------------

    def _span_key(self, span_id: str) -> str:
        return f"span/{span_id}"

    def _span_tags(self, span: Any) -> list[str]:
        tags = ["agno-span"]
        tid = getattr(span, "trace_id",      None)
        pid = getattr(span, "parent_span_id", None)
        if tid: tags.append(f"trace:{tid}")
        if pid: tags.append(f"parent:{pid}")
        return tags

    def create_span(self, span: Any) -> None:
        sid = getattr(span, "span_id", None) or str(uuid4())
        self._write(
            key=self._span_key(sid),
            type="working",
            tags=self._span_tags(span),
            value=_to_dict(span),
        )

    def create_spans(self, spans: List[Any]) -> None:
        for span in spans:
            self.create_span(span)

    def get_span(self, span_id: str) -> Any:
        entry = self.client.read(self._span_key(span_id))
        return entry["value"] if entry else None

    def get_spans(
        self,
        trace_id:       Optional[str] = None,
        parent_span_id: Optional[str] = None,
        limit:          Optional[int] = 1000,
    ) -> List[Any]:
        tags = ["agno-span"]
        if trace_id:       tags.append(f"trace:{trace_id}")
        if parent_span_id: tags.append(f"parent:{parent_span_id}")
        entries = self._page(type="working", tags=tags, limit=limit, page=1)
        return [e["value"] for e in entries]

    # -----------------------------------------------------------------------
    # Cultural knowledge
    # -----------------------------------------------------------------------

    def _culture_key(self, id: str) -> str:
        return f"culture/{id}"

    def _culture_tags(
        self,
        ck: CulturalKnowledge,
    ) -> list[str]:
        tags = ["agno-culture"]
        if ck.agent_id: tags.append(f"agent:{ck.agent_id}")
        if ck.team_id:  tags.append(f"team:{ck.team_id}")
        return tags

    def upsert_cultural_knowledge(
        self, cultural_knowledge: CulturalKnowledge
    ) -> Optional[CulturalKnowledge]:
        embedding = (
            self.embed_fn(cultural_knowledge.content)
            if self.embed_fn and cultural_knowledge.content
            else None
        )
        self._write(
            key=self._culture_key(cultural_knowledge.id),
            type="semantic",
            tags=self._culture_tags(cultural_knowledge),
            value=_to_dict(cultural_knowledge),
            embedding=embedding,
        )
        return cultural_knowledge

    def get_cultural_knowledge(self, id: str) -> Optional[CulturalKnowledge]:
        entry = self.client.read(self._culture_key(id))
        if entry is None:
            return None
        return _dc_from_dict(CulturalKnowledge, entry["value"])

    def get_all_cultural_knowledge(
        self,
        name:       Optional[str] = None,
        limit:      Optional[int] = None,
        page:       Optional[int] = None,
        sort_by:    Optional[str] = None,
        sort_order: Optional[str] = None,
        agent_id:   Optional[str] = None,
        team_id:    Optional[str] = None,
    ) -> Optional[List[CulturalKnowledge]]:
        tags = ["agno-culture"]
        if agent_id: tags.append(f"agent:{agent_id}")
        if team_id:  tags.append(f"team:{team_id}")
        entries = self._page(type="semantic", tags=tags, limit=limit, page=page)
        results = [_dc_from_dict(CulturalKnowledge, e["value"]) for e in entries]
        if name:
            results = [ck for ck in results if ck.name == name]
        return results

    def delete_cultural_knowledge(self, id: str) -> None:
        try:
            self.client.delete(self._culture_key(id))
        except Exception:
            pass

    def clear_cultural_knowledge(self) -> None:
        self.client.delete_bulk(type="semantic", tags=["agno-culture"])

    # -----------------------------------------------------------------------
    # Learnings
    # -----------------------------------------------------------------------

    def _learning_tags(
        self,
        learning_type: str,
        user_id:    Optional[str] = None,
        agent_id:   Optional[str] = None,
        team_id:    Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> list[str]:
        tags = ["agno-learning", f"ltype:{learning_type}"]
        if user_id:    tags.append(f"user:{user_id}")
        if agent_id:   tags.append(f"agent:{agent_id}")
        if team_id:    tags.append(f"team:{team_id}")
        if session_id: tags.append(f"session:{session_id}")
        return tags

    def upsert_learning(
        self,
        id:           str,
        learning_type: str,
        content:      Dict[str, Any],
        user_id:      Optional[str]            = None,
        agent_id:     Optional[str]            = None,
        team_id:      Optional[str]            = None,
        session_id:   Optional[str]            = None,
        namespace:    Optional[str]            = None,
        entity_id:    Optional[str]            = None,
        entity_type:  Optional[str]            = None,
        metadata:     Optional[Dict[str, Any]] = None,
    ) -> None:
        self._write(
            key=f"learning/{id}",
            type="procedural",
            tags=self._learning_tags(learning_type, user_id, agent_id, team_id, session_id),
            value={
                "id":            id,
                "learning_type": learning_type,
                "content":       content,
                "user_id":       user_id,
                "agent_id":      agent_id,
                "team_id":       team_id,
                "session_id":    session_id,
                "namespace":     namespace,
                "entity_id":     entity_id,
                "entity_type":   entity_type,
                "metadata":      metadata or {},
            },
        )

    def get_learning(
        self,
        learning_type: str,
        user_id:      Optional[str] = None,
        agent_id:     Optional[str] = None,
        team_id:      Optional[str] = None,
        session_id:   Optional[str] = None,
        namespace:    Optional[str] = None,
        entity_id:    Optional[str] = None,
        entity_type:  Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        tags    = self._learning_tags(learning_type, user_id, agent_id, team_id, session_id)
        entries = self.client.list(type="procedural", tags=tags, limit=1)["entries"]
        if not entries:
            return None
        v = entries[0]["value"]
        # apply extra client-side filters that can't be expressed as tags
        if namespace   and v.get("namespace")   != namespace:   return None
        if entity_id   and v.get("entity_id")   != entity_id:   return None
        if entity_type and v.get("entity_type") != entity_type: return None
        return v

    def get_learnings(
        self,
        learning_type: Optional[str] = None,
        user_id:      Optional[str]  = None,
        agent_id:     Optional[str]  = None,
        team_id:      Optional[str]  = None,
        session_id:   Optional[str]  = None,
        namespace:    Optional[str]  = None,
        entity_id:    Optional[str]  = None,
        entity_type:  Optional[str]  = None,
        limit:        Optional[int]  = None,
    ) -> List[Dict[str, Any]]:
        tags = ["agno-learning"]
        if learning_type: tags.append(f"ltype:{learning_type}")
        if user_id:       tags.append(f"user:{user_id}")
        if agent_id:      tags.append(f"agent:{agent_id}")
        if team_id:       tags.append(f"team:{team_id}")
        if session_id:    tags.append(f"session:{session_id}")
        entries = self._page(type="procedural", tags=tags, limit=limit, page=None)
        results = [e["value"] for e in entries]
        if namespace:   results = [r for r in results if r.get("namespace")   == namespace]
        if entity_id:   results = [r for r in results if r.get("entity_id")   == entity_id]
        if entity_type: results = [r for r in results if r.get("entity_type") == entity_type]
        return results

    def delete_learning(self, id: str) -> bool:
        try:
            self.client.delete(f"learning/{id}")
            return True
        except Exception:
            return False
