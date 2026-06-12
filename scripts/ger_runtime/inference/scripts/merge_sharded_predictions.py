import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True, help="Dataset result directory containing shard_*/predictions.jsonl")
    parser.add_argument("--num_shards", type=int, required=True)
    parser.add_argument("--output", default="predictions.jsonl")
    return parser.parse_args()


def main():
    args = parse_args()
    result_dir = Path(args.dir)
    merged = {}

    for shard_id in range(args.num_shards):
        shard_file = result_dir / f"shard_{shard_id}" / "predictions.jsonl"
        if not shard_file.exists():
            raise FileNotFoundError(f"Missing shard predictions: {shard_file}")

        with shard_file.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                if "__original_index" not in item:
                    raise KeyError(f"{shard_file}:{line_no} does not contain __original_index")
                original_index = int(item["__original_index"])
                if original_index in merged:
                    raise ValueError(f"Duplicate original index {original_index} in {shard_file}:{line_no}")
                merged[original_index] = item

    if not merged:
        raise ValueError(f"No predictions found under {result_dir}")

    expected_indices = set(range(max(merged) + 1))
    missing = sorted(expected_indices - set(merged))
    if missing:
        preview = ", ".join(map(str, missing[:20]))
        raise ValueError(f"Missing {len(missing)} prediction indices, first missing: {preview}")

    output_file = result_dir / args.output
    with output_file.open("w", encoding="utf-8") as f:
        for original_index in sorted(merged):
            item = merged[original_index]
            item.pop("__original_index", None)
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Merged {len(merged)} predictions into {output_file}")


if __name__ == "__main__":
    main()
