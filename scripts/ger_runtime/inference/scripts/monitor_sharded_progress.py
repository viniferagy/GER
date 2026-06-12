import argparse
import json
import time
from pathlib import Path

from tqdm import tqdm


def count_nonempty_lines(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def read_shard_progress(shard_dir: Path):
    progress_file = shard_dir / "progress.json"
    if not progress_file.exists():
        return None
    try:
        with progress_file.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True, help="Dataset result directory containing shard_* subdirectories.")
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--desc", default="sharded inference")
    parser.add_argument("--stop-file", required=True)
    parser.add_argument("--interval", type=float, default=2.0)
    args = parser.parse_args()

    result_dir = Path(args.dir)
    stop_file = Path(args.stop_file)
    shard_dirs = [result_dir / f"shard_{idx}" for idx in range(args.num_shards)]

    bar = tqdm(total=None, desc=args.desc, unit="ex")
    try:
        while True:
            progress_items = [read_shard_progress(shard_dir) for shard_dir in shard_dirs]
            known_totals = [int(item["total"]) for item in progress_items if item is not None and "total" in item]
            total = sum(known_totals) if len(known_totals) == args.num_shards else None
            if total is not None and bar.total != total:
                bar.total = total

            completed = 0
            stages = []
            for shard_dir, item in zip(shard_dirs, progress_items):
                line_count = count_nonempty_lines(shard_dir / "predictions.jsonl")
                progress_count = int(item.get("completed", 0)) if item is not None else 0
                completed += max(line_count, progress_count)
                stages.append(item.get("stage", "starting") if item is not None else "starting")
            if completed >= bar.n:
                bar.update(completed - bar.n)
            else:
                bar.n = completed
            if stages:
                compact_stages = ",".join(f"{idx}:{stage}" for idx, stage in enumerate(stages))
                bar.set_postfix_str(compact_stages)
            bar.refresh()

            if stop_file.exists():
                break
            time.sleep(args.interval)
    finally:
        completed = sum(count_nonempty_lines(shard_dir / "predictions.jsonl") for shard_dir in shard_dirs)
        if completed >= bar.n:
            bar.update(completed - bar.n)
        else:
            bar.n = completed
        bar.refresh()
        bar.close()


if __name__ == "__main__":
    main()
