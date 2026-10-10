# Reproduction kit — the two zero-GPU read-outs

`repro.py` regenerates the two read-outs in the writeup that need **no GPU and no model**:

- **§3.4 geometry** — median pairwise cosine, share `>0.9` / `>0.99`, median top1−top5 gap.
- **§3.3 main finding** — that the symbol an issue names can be *absent* from the shipped graph at
  `base_commit`, and what the graph tool answers when it is asked about that symbol.

Everything it needs is public: the competition's `graphs/` + `embeddings/` files and, for the tool
replay, the official harness wheel (`swegemma`, installable from the competition page).

## Data

Download these from the competition dataset (they are the competition's own file names):

| file | used by |
|---|---|
| `<repo_short>_<base_commit>.npz` | geometry |
| `<repo_short>_<base_commit>.json` | the tool replay |
| `tasks.jsonl` | resolving a task's `base_commit` and its issue text |

`base_commit` for a task = the `base_commit` column of that task's row in `tasks.jsonl`.
If you keep graphs and embeddings in separate directories, point `EMB_DIR` at the embeddings one.

## Commands, with the values we measured (2026-10-10)

```
python repro.py geometry rich 577def22dd83b999c5937cb2bf2ed49600943325 <embeddings_dir>
→ rich  nodes=1944  pairs=1888596
  median pairwise cosine : 0.9342
  share > 0.9            : 0.6165
  share > 0.99           : 0.1141
  median (top1 - top5)   : 4.24e-04

python repro.py query rich aca9467ec6a482b3bed1ba7d2e3de72271c5f817 <graphs_dir> tasks.jsonl rich_3061
→ issue rich_3061 names: 'Text.extend_tabs'
  graph nodes containing 'Text.extend_tabs': 0  []
  search_similar_code('Text.extend_tabs') -> status=ok count=10
    similarity values: [0.5294, 0.5294, 0.5294, 0.5294, 0.5294, 0.5294, 0.5294, 0.5294, 0.5294, 0.5294]
    ALL VALUES EQUAL — a miss that is indistinguishable from a hit on the face of the response.
```

The same file also reproduces the other two rows of §3.4 (`httpx_4acf5c2c…`: 740 nodes, 0.9412,
1.6e-4; `fastapi_016ab760…`: 4,287 nodes, 0.8344, 3.4e-4) — one more `geometry` call each.

## What this kit does **not** reproduce

The behavioural arms (§3.1, §3.2) need a GPU box serving a model and the official runner; they are
specified by the protocol in §2 of the writeup (which files go under `data/`, what the two arms
change, what the read-out is) rather than by a script. The one piece of that protocol that is easy
to check without a GPU is the *prompt-level* effect: with the data present the harness adds a
595-character `## Code Intelligence Tools` section to the system prompt, and with it absent the
prompt is unchanged — that difference is what the file-existence gate controls.
