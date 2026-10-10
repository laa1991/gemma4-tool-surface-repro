"""Self-contained reproduction of the zero-GPU read-outs in our writeup.

Everything here runs from the competition's *public* dataset files plus the official harness
wheel. Nothing in this file depends on our machine.

What it reproduces
------------------
1.  Geometry (§3.4): median pairwise cosine, share >0.9 / >0.99, and the top1-top5 cosine gap
    for a repository's node embeddings.
2.  The main finding (§3.3): that the symbol named by an issue can be absent from the shipped
    graph at base_commit, and what the graph tool answers when it is asked about it.

Inputs (download these from the competition dataset; file names are the competition's own):
    <repo_short>_<base_commit>.npz    node embeddings        -> for read-out 1
    <repo_short>_<base_commit>.json   code graph             -> for read-out 2
    tasks.jsonl                       all tasks              -> for read-out 2 (issue text)

Usage
-----
    python repro.py geometry  <repo_short> <base_commit> <embeddings_dir>
    python repro.py query     <repo_short> <base_commit> <graph_dir> <tasks.jsonl> <task_id>
    python repro.py query     <repo_short> <base_commit> <graph_dir> <tasks.jsonl> <task_id> <symbol>

Example (the numbers quoted in §3.3 / §3.4):
    python repro.py geometry rich 577def22dd83b999c5937cb2bf2ed49600943325 .
    python repro.py query    rich aca9467ec6a482b3bed1ba7d2e3de72271c5f817 . tasks.jsonl \\
                             rich_3061 Text.extend_tabs

If the official harness (`swegemma`) is installed, `query` replays the tool call itself and prints
the response the agent would see; otherwise it prints the graph-level check only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def _npz_path(directory: str, repo: str, sha: str) -> Path:
    return Path(directory) / f'{repo}_{sha}.npz'


def geometry(directory: str, repo: str, sha: str) -> int:
    import numpy as np  # noqa: PLC0415

    path = _npz_path(directory, repo, sha)
    if not path.exists():
        print(f'missing {path}  (download it from the competition dataset)')
        return 2
    with np.load(path) as z:
        keys = list(z.files)
        vec = z[keys[0]] if len(keys) == 1 else np.stack([z[k] for k in keys])
        vec = np.asarray(vec, dtype=np.float64)
    norm = vec / (np.linalg.norm(vec, axis=1, keepdims=True) + 1e-12)
    n = norm.shape[0]
    sims = norm @ norm.T
    iu = np.triu_indices(n, k=1)
    flat = sims[iu]

    # top1 - top5 per query row
    np.fill_diagonal(sims, -np.inf)
    top = -np.sort(-sims, axis=1)[:, :5]
    gaps = top[:, 0] - top[:, 4]

    print(f'{repo}  nodes={n}  pairs={flat.size}')
    print(f'  median pairwise cosine : {np.median(flat):.4f}')
    print(f'  share > 0.9            : {(flat > 0.9).mean():.4f}')
    print(f'  share > 0.99           : {(flat > 0.99).mean():.4f}')
    print(f'  median (top1 - top5)   : {np.median(gaps):.2e}')
    print('  (normalisation: rows are L2-normalised here; the shipped tools normalise internally)')
    return 0


def query(graph_dir: str, tasks_jsonl: str, repo: str, sha: str, task_id: str, symbol: str | None) -> int:
    graph_path = Path(graph_dir) / f'{repo}_{sha}.json'
    if not graph_path.exists():
        print(f'missing {graph_path}  (download it from the competition dataset)')
        return 2
    graph = json.load(open(graph_path, encoding='utf-8'))
    ids = [n['id'] for n in graph['nodes']]

    if symbol is None:
        try:
            rows = [json.loads(line) for line in open(tasks_jsonl, encoding='utf-8-sig')]
            task = next(r for r in rows if r['instance_id'] == task_id)
            text = task.get('problem_statement') or ''
        except Exception as exc:  # noqa: BLE001
            print(f'could not read the issue text ({exc}); pass a symbol explicitly')
            return 2
        # the identifier the issue text names: the backticked dotted name it mentions
        import re  # noqa: PLC0415

        cands = re.findall(r'`([A-Za-z_][\w.]*\.[A-Za-z_]\w*)`', text) or re.findall(
            r'`([A-Za-z_]\w+)`', text)
        symbol = cands[0] if cands else ''
        print(f'issue {task_id} names: {symbol!r}  (head: {" ".join(text.split())[:110]}…)')

    hits = [i for i in ids if symbol in i]
    print(f'  graph nodes containing {symbol!r}: {len(hits)}  {hits[:3]}')

    try:
        from swegemma.tools import graph as G  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        print(f'  (official harness not importable: {exc})')
        print('  install it from the competition page to replay the tool response')
        return 0

    class _Ctx:
        def __init__(self) -> None:
            self.task = {'repo': repo, 'base_commit': sha}
            self.repo = repo
            self.graph_dir = graph_dir
            # the competition ships graph and embedding files flat; if you unzipped them into two
            # directories, point EMB_DIR at the embeddings one.
            import os  # noqa: PLC0415

            self.embeddings_dir = os.environ.get('EMB_DIR', graph_dir)
            self.max_stdout_chars = 5000

        def check_budget(self, **_kw):
            return None

    ctx = _Ctx()
    resp = json.loads(G.search_similar_code(ctx, symbol, 10))
    sims = [r.get('similarity') for r in (resp.get('results') or [])]
    print(f'  search_similar_code({symbol!r}) -> status={resp.get("status")} '
          f'count={resp.get("count")}')
    print(f'    similarity values: {sims}')
    if sims and len(set(sims)) == 1:
        print('    ALL VALUES EQUAL — this is the failure the writeup describes: a miss that is')
        print('    indistinguishable from a hit on the face of the response.')
    return 0


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    mode = argv[1]
    if mode == 'geometry' and len(argv) == 5:
        return geometry(argv[4], argv[2], argv[3])
    if mode == 'query' and len(argv) in (7, 8):
        return query(argv[4], argv[5], argv[2], argv[3], argv[6],
                     argv[7] if len(argv) == 8 else None)
    print(__doc__)
    return 2


if __name__ == '__main__':
    sys.exit(main(sys.argv))
