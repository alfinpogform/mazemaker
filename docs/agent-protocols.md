# Agent Protocols

Mazemaker agent protocols define how distributed agents coordinate across multiple memory systems. Protocols enable:

- **Multi-system memory federation** — share facts across Mazemaker, Supermemory, Perseus Vault, and custom providers
- **Cross-agent synchronization** — memories written to one system automatically surface in others
- **Conflict resolution** — newer facts supersede stale ones across all backends
- **Protocol versioning** — forward compatibility as systems evolve

---

## Table of contents

1. [Core concepts](#core-concepts)
2. [Protocol layers](#protocol-layers)
3. [Provider interface](#provider-interface)
4. [Synchronization protocol](#synchronization-protocol)
5. [Conflict resolution](#conflict-resolution)
6. [Example integrations](#example-integrations)

---

## Core concepts

### Agents and Systems

An **agent** is a stateful entity that remembers, reasons, and acts. An **agent system** is the memory backend it uses (Mazemaker, Supermemory, Perseus Vault, etc.).

### Agent Protocol

The **Agent Protocol** is a set of conventions for:

1. How agents **write** facts to their system
2. How facts **propagate** to federated systems
3. How **conflicts** are detected and resolved
4. How **versioning** and **causality** are preserved

### Key principles

- **Isolation**: Each system stores its own facts and operates independently
- **Causality**: Facts carry event-time and source information
- **Convergence**: Federated systems eventually agree on canonical state
- **Autonomy**: Agents decide what to share and with whom

---

## Protocol layers

```
┌──────────────────────────────────────────────────┐
│ Agent Application Layer                          │
│ (Your agents, skills, workflows)                │
└──────────────────┬───────────────────────────────┘
                   │
┌──────────────────▼───────────────────────────────┐
│ Integration Layer (this spec)                    │
│ - Multi-system recall                           │
│ - Selective propagation                         │
│ - Conflict detection                            │
└──────────────────┬───────────────────────────────┘
                   │
┌──────────────────▼───────────────────────────────┐
│ Provider Interface                               │
│ - MemoryProvider (Mazemaker, Perseus, etc.)     │
│ - Custom provider impl (Supermemory, etc.)      │
└──────────────────┬───────────────────────────────┘
                   │
┌──────────────────▼───────────────────────────────┐
│ Storage Layer                                    │
│ - SQLite / Postgres + pgvector                  │
│ - Vault HTTP endpoint                           │
│ - Supermemory API                               │
└──────────────────────────────────────────────────┘
```

---

## Provider interface

Every memory provider (Mazemaker, Perseus Vault, Supermemory, etc.) implements:

```python
class MemoryProvider(ABC):
    """Base interface for federated memory systems."""

    @property
    def name(self) -> str:
        """Provider name (e.g. 'mazemaker', 'supermemory', 'perseus-vault')."""
        pass

    def is_available(self) -> bool:
        """True if configured and reachable."""
        pass

    def initialize(self, session_id: str, **kwargs) -> None:
        """Start provider, resolve credentials, connect to backend."""
        pass

    def shutdown(self) -> None:
        """Graceful shutdown."""
        pass

    # ─────────────────────────────────────────────────

    def remember(self, content: str, label: str = "", 
                 category: str = "", **metadata) -> Dict[str, Any]:
        """Write a fact. Returns {"id": id, "stored": True, "superseded": []}."""
        pass

    def recall(self, query: str, k: int = 10, **filters) -> Dict[str, Any]:
        """Search facts. Returns {"results": [...], "elapsed_ms": int}."""
        pass

    def think(self, memory_id: int, k: int = 20, depth: int = 3) -> Dict[str, Any]:
        """Explore related memories via graph traversal."""
        pass

    def graph_stats(self) -> Dict[str, Any]:
        """Returns {"totals": {...}, "node": {...}}."""
        pass

    # ─────────────────────────────────────────────────

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """MCP tool schemas (remember, recall, think, etc.)."""
        pass

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any]) -> str:
        """Execute tool. Returns JSON string."""
        pass

    # ─────────────────────────────────────────────────

    def queue_prefetch(self, query: str, **kwargs) -> None:
        """Warm up a recall in background (optional)."""
        pass

    def prefetch(self, query: str, **kwargs) -> str:
        """Return prefetched recall block (or empty string)."""
        pass

    def on_memory_write(self, action: str, target: str, content: str,
                        metadata: Optional[Dict[str, Any]] = None) -> None:
        """Notified when another provider writes. Use for propagation."""
        pass
```

---

## Synchronization protocol

### Write propagation (Push)

When an agent writes to System A, the fact should propagate to Systems B and C:

```
Agent (via tool: remember)
    ↓
System A (e.g., Mazemaker)
    ├→ Validate & store locally
    ├→ Emit "write" event
    └→ Notify other providers via on_memory_write()
        ↓
    System B (e.g., Supermemory)
        ├→ Detect: same fact already here? Merge or ignore.
        └→ Store with source attribution
    System C (e.g., Perseus Vault)
        ├→ Detect: newer version exists? Supersede.
        └→ Store with causality chain
```

### Metadata envelope

Every fact crossing system boundaries carries:

```json
{
  "id": 12345,
  "content": "user prefers Italian cuisine",
  "label": "preference:cuisine",
  "category": "preference",
  "created_at": "2026-08-18T10:30:00Z",
  "event_time": "2026-08-18T10:30:00Z",
  "source_system": "mazemaker",
  "source_agent": "claude-code-session-abc123",
  "version": 1,
  "salience": 0.8,
  "deprecated": false,
  "supersedes": [11999, 12001],
  "superseded_by": null
}
```

### Three propagation modes

1. **Eager (default)**: Push immediately on write. For canonical shared facts.
2. **Lazy**: Store locally; sync on next recall query. For ephemeral/contextual facts.
3. **Manual**: Agent explicitly calls `sync()`. For sensitive or high-volume cases.

```python
# Eager propagation (system's decide via policy)
memory_provider.remember(
    "user prefers Italian cuisine",
    label="preference:cuisine",
    propagate="eager"  # or "lazy" or "manual"
)
```

---

## Conflict resolution

### Detection

Conflicts arise when two agents write contradictory facts:

```
Agent A: "user prefers Italian cuisine" (time: 10:30)
Agent B: "user prefers French cuisine"   (time: 10:35)
```

### Resolution strategies

1. **Last-write-wins (LWW)**: Newer timestamp supersedes. Simple, deterministic.
2. **Source priority**: Some agents are "trusted" (e.g., user directly, not inferred). Trusted sources win.
3. **Causality**: If there's an explicit `supersedes` chain, follow it.
4. **User arbitration**: Agent pauses and asks "which memory is correct?"

### Implementation

Every fact carries **event_time** (when it was created) and **vector_clock** (causality):

```json
{
  "id": 12345,
  "content": "user prefers French cuisine",
  "event_time": "2026-08-18T10:35:00Z",
  "vector_clock": {"system_a": 5, "system_b": 3},
  "supersedes": [12340],
  "superseded_by": null
}
```

On conflict:
1. Compare event_times. Newer wins.
2. If same time, compare vector clocks (causality). Non-concurrent wins.
3. If concurrent, apply source priority.

---

## Example integrations

### Supermemory Provider

Supermemory is a multi-agent shared memory platform. Here's how to integrate it:

```python
# plugins/supermemory/__init__.py

from agent.memory_provider import MemoryProvider

class SupermemoryProvider(MemoryProvider):
    """Federated Supermemory backend."""

    def __init__(self):
        self._client = None
        self._space_key = ""
        self._enabled = False

    @property
    def name(self) -> str:
        return "supermemory"

    def is_available(self) -> bool:
        """Check if supermemory credentials are set."""
        api_key = os.environ.get("SUPERMEMORY_API_KEY", "").strip()
        return bool(api_key)

    def initialize(self, session_id: str, **kwargs) -> None:
        """Connect to Supermemory via its REST API."""
        try:
            from supermemory import Supermemory
            api_key = os.environ.get("SUPERMEMORY_API_KEY")
            self._client = Supermemory(api_key=api_key)
            self._space_key = os.environ.get("SUPERMEMORY_SPACE", "default")
            self._enabled = True
        except Exception as e:
            logger.warning(f"supermemory: init failed: {e}")
            self._enabled = False

    def remember(self, content: str, label: str = "", 
                 category: str = "", **metadata) -> Dict[str, Any]:
        """Write to Supermemory and return metadata."""
        if not self._enabled:
            return {"stored": False, "error": "supermemory not available"}
        try:
            result = self._client.add_memory(
                content=content,
                space=self._space_key,
                tags=[label, category],
                metadata=metadata
            )
            return {
                "id": result.get("id"),
                "stored": True,
                "fused_into": None,
                "superseded": []
            }
        except Exception as e:
            return {"stored": False, "error": str(e)}

    def recall(self, query: str, k: int = 10, **filters) -> Dict[str, Any]:
        """Search Supermemory."""
        if not self._enabled:
            return {"results": [], "elapsed_ms": 0}
        try:
            results = self._client.search(
                query=query,
                space=self._space_key,
                limit=k
            )
            return {
                "results": [
                    {
                        "id": r.get("id"),
                        "label": r.get("label", ""),
                        "content": r.get("content", ""),
                        "score": r.get("score", 0.0)
                    }
                    for r in results
                ],
                "elapsed_ms": 0
            }
        except Exception as e:
            logger.error(f"supermemory recall failed: {e}")
            return {"results": [], "elapsed_ms": 0}

    def on_memory_write(self, action: str, target: str, content: str,
                        metadata: Optional[Dict[str, Any]] = None) -> None:
        """Propagate writes from Mazemaker to Supermemory."""
        if not self._enabled or action != "add":
            return
        try:
            self._client.add_memory(
                content=content,
                space=self._space_key,
                metadata={**metadata, "source": target}
            )
        except Exception as e:
            logger.debug(f"supermemory propagation failed: {e}")

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "supermemory_remember",
                "description": "Store a memory in Supermemory (shared across agents)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "description": "Memory content"},
                        "label": {"type": "string", "description": "Label/tag"},
                    },
                    "required": ["content"],
                }
            },
            {
                "name": "supermemory_recall",
                "description": "Search Supermemory",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "limit": {"type": "integer", "description": "Max results (default: 10)"},
                    },
                    "required": ["query"],
                }
            }
        ]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any]) -> str:
        if tool_name == "supermemory_remember":
            result = self.remember(
                content=args.get("content", ""),
                label=args.get("label", "")
            )
            return json.dumps(result)
        elif tool_name == "supermemory_recall":
            result = self.recall(
                query=args.get("query", ""),
                k=args.get("limit", 10)
            )
            return json.dumps(result)
        return json.dumps({"error": f"unknown tool {tool_name}"})

def register(ctx) -> None:
    ctx.register_memory_provider(SupermemoryProvider())
```

---

## Enabling protocols

### For a single agent system

```yaml
# ~/.hermes/config.yaml
memory:
  provider: neural  # Mazemaker
  neural:
    db_path: ~/.mazemaker/engine/memory.db
```

### For multi-system federation

```yaml
# ~/.hermes/config.yaml
memory:
  providers:
    - name: mazemaker
      type: neural
      config:
        db_path: ~/.mazemaker/engine/memory.db
        propagate: eager
    - name: supermemory
      type: supermemory
      config:
        api_key: ${SUPERMEMORY_API_KEY}
        space: shared
        propagate: lazy
    - name: vault
      type: perseus-vault
      config:
        url: https://vault.perseus.observer/message
        token: ${PERSEUS_VAULT_MCP_TOKEN}
        propagate: manual
  
  # Conflict resolution strategy
  conflict_resolution:
    strategy: last_write_wins  # or "source_priority", "user_arbitration"
    trusted_sources:
      - "user-input"
      - "claude-code"
```

---

## Going deeper

- **Embedding consistency** — [`architecture.md#embedding-backends`](architecture.md#embedding-backends)
- **Dream consolidation** — [`dream-engine.md`](dream-engine.md)
- **MCP tools reference** — [`mcp-tools.md`](mcp-tools.md)
- **Configuration all knobs** — [`configuration.md`](configuration.md)
