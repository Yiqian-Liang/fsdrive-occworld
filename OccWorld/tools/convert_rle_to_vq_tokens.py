#!/usr/bin/env python3
"""Convert OCC RLE JSONL to `<VQ_*>` token sequence JSONL.

Each line of the input JSONL is expected to contain a structure like:
{
    "conversations": [
        {"from": "human", "value": "..."},
        {"from": "gpt",   "value": "[[id,count], ...]"}
    ],
    "images": ["path/to/image.jpg"]
}
where the assistant message holds a JSON string of RLE pairs. This script
expands the RLE into a whitespace separated string of `<VQ_*>` tokens and
writes the result to the output JSONL.
"""
import argparse
import json
from pathlib import Path
from typing import List, Tuple


def rle_to_vq_tokens(rle_pairs: List[Tuple[int, int]], *, offset: int = 0, prefix: str = "VQ") -> str:
    """Expand RLE ``[[id,count], ...]`` to a `<VQ_id>` token string.

    Args:
        rle_pairs: List of ``[id, count]`` pairs.
        offset: Optional integer added to each ``id`` before formatting.
        prefix: Token prefix, default ``"VQ"``.
    """
    tokens: List[str] = []
    for id_val, count in rle_pairs:
        tok_id = int(id_val) + offset
        tokens.extend([f"<{prefix}_{tok_id}>"] * int(count))
    return " ".join(tokens)


def process_item(item: dict, *, offset: int, prefix: str) -> dict:
    """Convert the assistant RLE message in one JSON item to `<VQ_*>` tokens."""
    conversations = item.get("conversations", [])
    if len(conversations) != 2:
        return None
    human, gpt = conversations
    if human.get("from") != "human" or gpt.get("from") != "gpt":
        return None

    try:
        rle_pairs = json.loads(gpt["value"])
    except json.JSONDecodeError:
        return None

    token_str = rle_to_vq_tokens(rle_pairs, offset=offset, prefix=prefix)
    item["conversations"] = [human, {"from": "gpt", "value": token_str}]
    return item


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert OCC RLE jsonl to `<VQ_*>` token jsonl")
    parser.add_argument("--input", required=True, help="Input RLE JSONL file")
    parser.add_argument("--output", required=True, help="Output JSONL with `<VQ_*>` token strings")
    parser.add_argument("--offset", type=int, default=0, help="Offset added to ids before formatting")
    parser.add_argument("--prefix", default="VQ", help="Token prefix, default 'VQ'")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)

    with in_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            item = process_item(item, offset=args.offset, prefix=args.prefix)
            if item is None:
                continue
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"[DONE] Converted {in_path} -> {out_path}")


if __name__ == "__main__":
    main()
