# helper: emit the union of the 5 custom folds, one PanTS id per line
from pathlib import Path
d = Path("src/data")
ids = set()
for i in range(1, 6):
    for ln in (d / f"fold_{i}_ids.txt").read_text().splitlines():
        ln = ln.strip()
        if ln:
            ids.add(ln)
for x in sorted(ids):
    print(x)
