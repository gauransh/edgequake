# The sandbox demo query path

`POST /api/v1/query/sandbox-demo` answers one question about one frozen graph
of governed sandbox runs, and it cannot answer it without graph-core.

```
frozen sandbox evidence            (itsm-sandbox-policy-engine)
  -> snapshot.json + a trained model pack
  -> EdgeQuake embeds one document per governed resource, retrieves top-K
  -> the EMBEDDED graph-core runtime reranks them IN THIS PROCESS
  -> EdgeQuake returns the graph-core-approved order + evidence + latency
```

The demo exists only when EdgeQuake is started with `EDGEQUAKE_SANDBOX_PACK_DIR`
pointing at an absolute path. Unset, the routes do not exist. Set, the pack must
load or the process refuses to start.

## Two ways to run it

**`edgequake-sandbox-demo`** — the demo and nothing else: no `DATABASE_URL`,
no migrations, no worker pool. The demo's index is one frozen `snapshot.json`
in memory and its reranker is linked in, so none of that is on the hot path.
Fewer moving parts between the presenter and the query is the point.

```bash
EDGEQUAKE_SANDBOX_PACK_DIR=/abs/path/to/pack \
EDGEQUAKE_EMBEDDING_PROVIDER=ollama \
EDGEQUAKE_EMBEDDING_MODEL=nomic-embed-text \
EDGEQUAKE_EMBEDDING_DIMENSION=768 \
  cargo run --bin edgequake-sandbox-demo
# -> http://127.0.0.1:8080/demo/sandbox
```

**The full `edgequake` server** — set the same `EDGEQUAKE_SANDBOX_PACK_DIR`
and the two routes appear alongside everything else. Needs the ordinary
`DATABASE_URL` and the rest of the stack.

## Routes

| Method | Path | |
|---|---|---|
| `GET` | `/demo/sandbox` | the same-origin one-page UI |
| `GET` | `/api/v1/query/sandbox-demo/manifest` | identity of the loaded generation |
| `POST` | `/api/v1/query/sandbox-demo` | the query |

```json
{
  "workspace_id": "demo-workspace",
  "query": "Which governed resources show the riskiest access pattern?",
  "anchor_external_id": "agent:claude-code-demo",
  "candidate_k": 20,
  "top_k": 5
}
```

The anchor is **explicit and frozen**. It is never inferred from the question,
and it is a graph-core external id, never an EdgeQuake display name.

## Why graph-core is a library here, not a service

EdgeQuake's ordinary rerank integration is an HTTP call to graph-core, and that
route is *advisory*: when no model is active it returns a degraded response and
EdgeQuake's original order stands, because a user's query must never fail on
evidence that is only meant to help.

This path makes the opposite promise. The claim being demonstrated is that the
final ordering is *necessarily* graph-core's, so:

* the runtime is `itsm_graph_core::serving::embedded_runtime`, linked into this
  binary — no graph-core HTTP, S3, S3 Vectors, or DynamoDB call happens after
  startup;
* a missing, corrupt, or misaligned pack aborts startup;
* an anchor or candidate outside the model's id space fails the query;
* `graph_core_applied` is always `true` in a success response, because no other
  response can be produced;
* the page has one control — run the query — and no way to turn the reranker off.

## Startup gates

`SandboxDemoQueryEngine::load` fails, and EdgeQuake does not start, when:

1. graph-core refuses the bundle (see `itsm-graph-core/docs/embedded-serving.md`
   for its nine checks over `model.safetensors` / `manifest.json` /
   `mapping.json` / `activation.json`);
2. `snapshot.json` does not hash to `activation.json`'s `snapshot_sha256`;
3. the snapshot names a different workspace or revision than the activation;
4. any relationship has a dangling endpoint or any entity id repeats;
5. **any** governed resource in the snapshot is outside the model's id space —
   a partial overlap means the snapshot and the pack are not a matched pair;
6. the embedding provider fails, or returns the wrong count or mixed dimensions.

Loading also embeds every resource document, so the first query on stage is warm.

## Retrieval documents

A governed resource has almost no text: most labels are redacted hashes
(`directory 090805d6`) and every description is the same exporter boilerplate.
So `sandbox_demo::documents` builds each document from what the *graph* says —
permitted/refused counts, distinct runs, action keywords, agent profiles,
scenarios, outcomes.

* **Deterministic** — only sorted collections and integers, so warm runs return
  byte-identical text and therefore identical orders.
* **Safe** — no entity metadata, exporter description, audit payload, or file
  content reaches the document. Labels are already through the exporter's
  allowlist redaction.

The risk vocabulary ("refused", "contested", "high-risk") appears only when the
graph actually carries denial edges. That is what lets a question about risk
retrieve the resources policy really refused rather than the ones that merely
appear often.

## Scoring

| component | source | weight (frozen in `activation.json`) |
|---|---|---|
| semantic | EdgeQuake's configured embedding provider, cosine mapped to `[0,1]` | 0.60 |
| structural | graph-core anchor↔candidate cosine over the trained entity matrix | 0.40 |
| graph | — | 0.00 |
| link | not part of this path | 0.00 |

Weights are per-bundle, never per query. `final_score = 0.6·semantic + 0.4·structural`.

`semantic_score` is `(cosine + 1) / 2` because the blend contract is unit-range;
the raw cosine is also returned as `semantic_cosine`.

## Evidence

Each result carries `[anchor, run, relation, resource]`, resolved from the
in-memory graph with this preference order so the most governance-relevant path
available is the one shown, deterministically:

1. a run the anchor initiated that was **refused** this resource,
2. a run the anchor initiated that was permitted it,
3. any run that was refused it,
4. any run that was permitted it.

`evidence.anchor_runs_touching` and `anchor_run_share_pct` are the numbers worth
putting on screen: they are the difference between "the model ranked this first"
and "the model ranked this first, and here are the anchor's runs that touched it".

## Scope

This is a focused query adapter, not a storage migration. Nothing here touches
EdgeQuake's persistent backends, Local/Global/Hybrid/Mix modes, or the ordinary
`/api/v1/query` path. The same `EmbeddedGraphCoreRuntime` interface is what would
later slot into Local mode between vector candidate retrieval and graph
expansion — the seam that already exists in `engine_impl::modes::local`.
