"""
Local in-memory replacement for Zep Cloud.

Activated when ZEP_API_KEY is not set (or zep-cloud is not installed).
Text chunks are stored in-memory; graph.search() uses claude -p to find
relevant excerpts and returns them as facts.
"""

from __future__ import annotations

import json
import re
import subprocess
import threading
import uuid as _uuid_mod
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Compatibility types (mirror the public zep_cloud API surface MiroFish uses)
# ---------------------------------------------------------------------------

try:
    from zep_cloud import NotFoundError  # use the real type so except-clauses in app code match
except ImportError:
    class NotFoundError(Exception):  # type: ignore[no-redef]
        """Fallback when zep_cloud is not installed."""


class BatchAddItem:
    def __init__(
        self,
        type: str = "graph_episode",
        graph_id: str = "",
        data: str = "",
        data_type: str = "text",
        source_description: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.type = type
        self.graph_id = graph_id
        self.data = data
        self.data_type = data_type
        self.source_description = source_description
        self.metadata = metadata or {}


class EntityEdgeSourceTarget:
    def __init__(self, source: str = "", target: str = ""):
        self.source = source
        self.target = target


# ---------------------------------------------------------------------------
# Simple attribute bag used for all API result objects
# ---------------------------------------------------------------------------

class _Obj:
    def __init__(self, **kw: Any):
        for k, v in kw.items():
            setattr(self, k, v)


# ---------------------------------------------------------------------------
# Node and Edge objects
# ---------------------------------------------------------------------------

class _Node:
    def __init__(
        self,
        uuid_: str,
        name: str,
        labels: List[str],
        summary: str,
        attributes: Dict[str, Any],
    ):
        self.uuid_ = uuid_
        self.name = name
        self.labels = labels
        self.summary = summary
        self.attributes = attributes


class _Edge:
    def __init__(
        self,
        uuid_: str,
        name: str,
        fact: str,
        source_node_uuid: str,
        target_node_uuid: str,
        attributes: Dict[str, Any],
    ):
        self.uuid_ = uuid_
        self.name = name
        self.fact = fact
        self.source_node_uuid = source_node_uuid
        self.target_node_uuid = target_node_uuid
        self.attributes = attributes


class _Fact:
    def __init__(self, fact: str):
        self.fact = fact


# ---------------------------------------------------------------------------
# with_raw_response result — .data list + .headers dict
# ---------------------------------------------------------------------------

class _RawResponseWrapper:
    def __init__(self, data: List[Any], next_cursor: Optional[str] = None):
        self.data = data
        self.headers: Dict[str, str] = {}
        if next_cursor is not None:
            self.headers["zep-next-cursor"] = next_cursor


# ---------------------------------------------------------------------------
# Global in-memory store (thread-safe)
# ---------------------------------------------------------------------------

_graphs: Dict[str, Dict[str, Any]] = {}
_batches: Dict[str, Dict[str, Any]] = {}
_LOCK = threading.Lock()

# Per-graph:
# {
#   "name": str,
#   "description": str,
#   "chunks": [str],
#   "nodes": {uuid_: _Node},
#   "edges": {uuid_: _Edge},
# }

# Per-batch:
# {
#   "metadata": dict,
#   "items": [{"data": str, "graph_id": str, "metadata": dict,
#               "episode_uuid": str, "sequence_index": int}],
#   "status": "draft" | "succeeded",
# }


# ---------------------------------------------------------------------------
# Claude helper
# ---------------------------------------------------------------------------

def _call_claude(prompt: str, timeout: int = 120) -> str:
    result = subprocess.run(
        [
            "claude", "-p", prompt,
            "--output-format", "text",
            "--disallowed-tools", "Bash,Edit,Write,Read,Glob,Grep,WebSearch,WebFetch",
        ],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return (result.stdout or result.stderr or "").strip()


def _strip_fences(text: str) -> str:
    text = re.sub(r"^```(?:json)?\s*\n?", "", text, flags=re.IGNORECASE)
    return re.sub(r"\n?```\s*$", "", text).strip()


# ---------------------------------------------------------------------------
# Search helper — asks claude to rank chunks by relevance
# ---------------------------------------------------------------------------

def _search_chunks(chunks: List[str], query: str, limit: int) -> List[str]:
    if not chunks:
        return []
    sample = chunks[:30]
    numbered = "\n\n".join(f"[{i + 1}] {c[:400]}" for i, c in enumerate(sample))
    prompt = (
        f"You are a search assistant. Given these numbered text excerpts and a query, "
        f"return the indices (1-based) of the {min(limit, len(sample))} most relevant excerpts "
        f"as a JSON array of integers. Output only the JSON array, e.g. [2, 5, 1].\n\n"
        f"Query: {query}\n\nExcerpts:\n{numbered}"
    )
    try:
        raw = _call_claude(prompt, timeout=60)
        indices = json.loads(_strip_fences(raw))
        if isinstance(indices, list):
            results = []
            for i in indices:
                if isinstance(i, int) and 1 <= i <= len(sample):
                    results.append(chunks[i - 1])
                if len(results) >= limit:
                    break
            return results
    except Exception:
        pass
    return chunks[:limit]


# ---------------------------------------------------------------------------
# Sub-APIs
# ---------------------------------------------------------------------------

class _EpisodeAPI:
    def get(self, *, uuid_: str) -> _Obj:
        return _Obj(uuid_=uuid_, status="processed", processed=True)


class _NodeWithRawResponse:
    def get_by_graph_id(
        self,
        graph_id: str,
        *,
        limit: int = 100,
        cursor: Optional[str] = None,
    ) -> _RawResponseWrapper:
        with _LOCK:
            graph = _graphs.get(graph_id) or {}
            all_nodes = list((graph.get("nodes") or {}).values())
        start = int(cursor) if cursor and str(cursor).isdigit() else 0
        page = all_nodes[start : start + limit]
        next_cursor = str(start + limit) if start + limit < len(all_nodes) else None
        return _RawResponseWrapper(page, next_cursor)


class _NodeAPI:
    def __init__(self):
        self.with_raw_response = _NodeWithRawResponse()

    def get(self, *, uuid_: str) -> _Node:
        with _LOCK:
            for graph in _graphs.values():
                nodes = graph.get("nodes") or {}
                if uuid_ in nodes:
                    return nodes[uuid_]
        raise NotFoundError(f"Node {uuid_!r} not found")

    def get_edges(self, *, node_uuid: str) -> List[_Edge]:
        with _LOCK:
            results = []
            for graph in _graphs.values():
                for edge in (graph.get("edges") or {}).values():
                    if (
                        edge.source_node_uuid == node_uuid
                        or edge.target_node_uuid == node_uuid
                    ):
                        results.append(edge)
        return results


class _EdgeWithRawResponse:
    def get_by_graph_id(
        self,
        graph_id: str,
        *,
        limit: int = 100,
        cursor: Optional[str] = None,
    ) -> _RawResponseWrapper:
        with _LOCK:
            graph = _graphs.get(graph_id) or {}
            all_edges = list((graph.get("edges") or {}).values())
        start = int(cursor) if cursor and str(cursor).isdigit() else 0
        page = all_edges[start : start + limit]
        next_cursor = str(start + limit) if start + limit < len(all_edges) else None
        return _RawResponseWrapper(page, next_cursor)


class _EdgeAPI:
    def __init__(self):
        self.with_raw_response = _EdgeWithRawResponse()


class _GraphAPI:
    def __init__(self):
        self.episode = _EpisodeAPI()
        self.node = _NodeAPI()
        self.edge = _EdgeAPI()

    def create(
        self,
        *,
        graph_id: str,
        name: str = "",
        description: str = "",
        **kw: Any,
    ) -> _Obj:
        with _LOCK:
            if graph_id not in _graphs:
                _graphs[graph_id] = {
                    "name": name,
                    "description": description,
                    "chunks": [],
                    "nodes": {},
                    "edges": {},
                }
        return _Obj(graph_id=graph_id, name=name)

    def get(self, graph_id: str) -> _Obj:
        with _LOCK:
            if graph_id not in _graphs:
                raise NotFoundError(f"Graph {graph_id!r} not found")
            g = _graphs[graph_id]
        return _Obj(graph_id=graph_id, name=g["name"])

    def delete(self, *, graph_id: str) -> None:
        with _LOCK:
            _graphs.pop(graph_id, None)

    def add(
        self,
        *,
        graph_id: str = "",
        type: str = "text",  # noqa: A002
        data: str = "",
        **kw: Any,
    ) -> _Obj:
        ep_uuid = str(_uuid_mod.uuid4())
        with _LOCK:
            if graph_id not in _graphs:
                _graphs[graph_id] = {"name": graph_id, "description": "", "chunks": [], "nodes": {}, "edges": {}}
            _graphs[graph_id]["chunks"].append(data)
        return _Obj(uuid=ep_uuid, uuid_=ep_uuid, status="processed", processed=True)

    def set_ontology(
        self,
        *,
        graph_ids: Optional[List[str]] = None,
        entities: Any = None,
        edges: Any = None,
        **kw: Any,
    ) -> None:
        # No-op locally; we don't enforce ontology constraints
        pass

    def search(
        self,
        *,
        graph_id: str = "",
        query: str = "",
        search_filters: Any = None,
        limit: int = 5,
        **kw: Any,
    ) -> _Obj:
        with _LOCK:
            graph = _graphs.get(graph_id) or {}
            chunks = list(graph.get("chunks") or [])
        scope = kw.get("scope", "edges")
        relevant = _search_chunks(chunks, query, limit)
        # Return results in whichever collection the caller reads
        if scope == "nodes":
            nodes = [_Node(uuid_=str(_uuid_mod.uuid4()), name="", labels=["Entity"],
                           summary=chunk[:200], attributes={}) for chunk in relevant]
            return _Obj(edges=[], nodes=nodes, facts=[])
        else:
            edges = [_Edge(uuid_=str(_uuid_mod.uuid4()), name="RELATED_TO",
                           fact=chunk[:400], source_node_uuid="", target_node_uuid="",
                           attributes={}) for chunk in relevant]
            return _Obj(edges=edges, nodes=[], facts=[])


# ---------------------------------------------------------------------------
# Batch API
# ---------------------------------------------------------------------------

class _BatchItemResult:
    def __init__(
        self,
        episode_uuid: str,
        sequence_index: int,
        status: str = "succeeded",
        source_uuid: Optional[str] = None,
        error: Optional[str] = None,
    ):
        self.episode_uuid = episode_uuid
        self.sequence_index = sequence_index
        self.status = status
        self.source_uuid = source_uuid or episode_uuid
        self.error = error


class _BatchAPI:
    def create(self, *, metadata: Optional[Dict[str, Any]] = None, **kw: Any) -> _Obj:
        batch_id = f"local-batch-{_uuid_mod.uuid4().hex[:12]}"
        with _LOCK:
            _batches[batch_id] = {
                "metadata": metadata or {},
                "items": [],
                "status": "draft",
            }
        return _Obj(batch_id=batch_id)

    def add(
        self,
        *,
        batch_id: str,
        items: List[Any],
        **kw: Any,
    ) -> List[_BatchItemResult]:
        results = []
        with _LOCK:
            batch = _batches.get(batch_id)
            if batch is None:
                raise NotFoundError(f"Batch {batch_id!r} not found")
            for item in items:
                ep_uuid = str(_uuid_mod.uuid4())
                seq = len(batch["items"])
                batch["items"].append({
                    "data": getattr(item, "data", ""),
                    "graph_id": getattr(item, "graph_id", "")
                    or (batch["metadata"].get("graph_id", "")),
                    "metadata": getattr(item, "metadata", {}),
                    "episode_uuid": ep_uuid,
                    "sequence_index": seq,
                })
                results.append(
                    _BatchItemResult(
                        episode_uuid=ep_uuid,
                        sequence_index=seq,
                        status="succeeded",
                        source_uuid=ep_uuid,
                    )
                )
        return results

    def process(self, *, batch_id: str, **kw: Any) -> _Obj:
        """Store all batch text chunks into the target graph."""
        with _LOCK:
            batch = _batches.get(batch_id)
            if batch is None:
                raise NotFoundError(f"Batch {batch_id!r} not found")
            graph_id_default = batch["metadata"].get("graph_id", "")
            for item in batch["items"]:
                graph_id = item.get("graph_id") or graph_id_default
                data = item.get("data", "")
                if graph_id and graph_id in _graphs and data:
                    _graphs[graph_id]["chunks"].append(data)
            batch["status"] = "succeeded"
        return _Obj(batch_id=batch_id, status="succeeded")

    def get(self, *, batch_id: str, **kw: Any) -> _Obj:
        with _LOCK:
            batch = _batches.get(batch_id)
        if batch is None:
            raise NotFoundError(f"Batch {batch_id!r} not found")
        item_count = len(batch["items"])
        return _Obj(
            batch_id=batch_id,
            status=batch["status"],
            progress=_Obj(
                percent_complete=100 if batch["status"] == "succeeded" else 0,
                succeeded_items=item_count if batch["status"] == "succeeded" else 0,
            ),
        )

    def list(
        self,
        *,
        limit: int = 100,
        cursor: Optional[int] = None,
        **kw: Any,
    ) -> _Obj:
        with _LOCK:
            all_batches = [
                _Obj(batch_id=bid, metadata=b["metadata"], status=b["status"])
                for bid, b in _batches.items()
            ]
        start = cursor or 0
        page = all_batches[start : start + limit]
        next_cursor = start + limit if start + limit < len(all_batches) else None
        return _Obj(batches=page, next_cursor=next_cursor)

    def list_items(
        self,
        *,
        batch_id: str,
        limit: int = 100,
        cursor: Optional[int] = None,
        **kw: Any,
    ) -> _Obj:
        with _LOCK:
            batch = _batches.get(batch_id)
        if batch is None:
            raise NotFoundError(f"Batch {batch_id!r} not found")
        all_items = [
            _BatchItemResult(
                episode_uuid=item["episode_uuid"],
                sequence_index=item["sequence_index"],
                status="succeeded",
                source_uuid=item["episode_uuid"],
                error=None,
            )
            for item in batch["items"]
        ]
        start = cursor or 0
        page = all_items[start : start + limit]
        next_cursor = start + limit if start + limit < len(all_items) else None
        return _Obj(items=page, next_cursor=next_cursor)


# ---------------------------------------------------------------------------
# Public client
# ---------------------------------------------------------------------------

class LocalZepClient:
    """Drop-in replacement for zep_cloud.client.Zep when no API key is set."""

    def __init__(self):
        self.graph = _GraphAPI()
        self.batch = _BatchAPI()
