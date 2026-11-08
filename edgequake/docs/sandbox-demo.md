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
final ordering is *necessarily* produced by graph-core's blend (under the default
weights that blend is semantic-only — see [Scoring](#scoring)), so:

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

| component | source | default weight | `bundle` weight (frozen in `activation.json`) |
|---|---|---|---|
| semantic | EdgeQuake's configured embedding provider, cosine mapped to `[0,1]` | 1.00 | 0.60 |
| structural | graph-core anchor↔candidate cosine over the trained entity matrix | 0.00 | 0.40 |
| graph | — | 0.00 | 0.00 |
| link | not part of this path | 0.00 | 0.00 |

Weights are per-process, never per query. The default is
`final_score = 1.0·semantic + 0.0·structural`: on a pre-registered gold
relevance set, semantic-only ranking beat the bundle's `0.6 / 0.4` blend by
+0.113 nDCG@5 on holdout (95% CI `[+0.030, +0.214]`), and no structural arm tried
beat semantic-only (crystalos-ai/itsm-graph-core #226/#227). graph-core still
loads, passes every startup gate, and reranks every query; `structural_score` is
still computed and returned for transparency, and `graph_core_applied` is still
`true`. Only its blend weight is zero.

`EDGEQUAKE_SANDBOX_BLEND_WEIGHTS` selects the weights:

| value | weights | logged `weights_source` |
|---|---|---|
| unset / blank | `semantic=1.0, structural=0.0` | `default` |
| `bundle` | the bundle's frozen weights — the runtime's own rerank, byte-identical to the pre-default response (the rollback switch) | `bundle` |
| `semantic=<w>,structural=<w>` | those, each `>= 0`, summing to ~1 | `env` |

The response's `weights` and the manifest's `weights` are the ones applied.
`EDGEQUAKE_SANDBOX_STRUCTURAL_SCORER` (`cosine` default) picks the structural
evidence and is independent of the weights.

`semantic_score` is `(cosine + 1) / 2` because the blend contract is unit-range;
the raw cosine is also returned as `semantic_cosine`.

### Opt-in intent routing

`EDGEQUAKE_SANDBOX_INTENT_ROUTING` is `off` by default. Unset (or `off`), every
query takes the path above and the response is byte-identical to before the knob
existed — there is no `intent` field (pinned by a golden test captured at
56060ca).

With `EDGEQUAKE_SANDBOX_INTENT_ROUTING=next_touch`, a query that matches the
next-touch lexicon is scored as

```text
structural' = zscore(structural over the candidate set) · std(semantic over the candidate set)
final_score = semantic + λ · structural'
```

with population standard deviations; a constant structural (or semantic) column
gives `structural' = 0`. `structural` comes from
`EDGEQUAKE_SANDBOX_INTENT_SCORER` (`agent_frequency` default | `popularity` |
`cosine` | `knn` | `blend:<name>=<w>,...`, same grammar as the structural scorer)
and `λ` from `EDGEQUAKE_SANDBOX_INTENT_LAMBDA` (default `1.0`, finite, `>= 0`).
Such a response carries `"intent": "next_touch"`, `structural_scorer` names the
intent scorer, `structural_score` is the scorer's raw value, and `weights` is
`semantic=1, structural=λ`. Ties break on external id. Every other query keeps
the default path exactly and carries `"intent": null`.

The lexicon is fixed in code (`sandbox_demo::intent`), not tuned on results.
Lowercase, split on whitespace, trim punctuation from token ends; a query matches
when a cue and an action occur within 4 tokens of each other, either order:

| cues | actions |
|---|---|
| `will`, `next`, `likely`, `usually`, `typically`, `tends to`, `tend to`, `going to` | `touch(es)`, `use(s)`, `access(es)`, `open(s)`, `run(s)`, `need(s)` |

Whole tokens only (`reuse` is not `use`, `willing` is not `will`). Matches:
"What will claude-code-demo touch next?", "Which file does it open next?".
Does not match: "Which resources show the riskiest access pattern?", "Who will
approve this?". Known looseness: `run` is also a noun, so "the next run" matches.

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
