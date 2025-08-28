# tools/filter_index_by_images.py
import os, json, argparse
from pathlib import Path
from typing import List  # 添加这一行，兼容Python 3.8

CAMS_ORDER = ["CAM_FRONT_LEFT","CAM_FRONT","CAM_FRONT_RIGHT",
              "CAM_BACK_LEFT","CAM_BACK","CAM_BACK_RIGHT"]

def resolve_abs(rel_path: str, old_prefix: str, roots: List[str]) -> str:  # 修改为 List[str]
    """把 JSONL 里的相对路径映射到本机的若干候选根目录；返回第一个存在的绝对路径；否则返回空串。"""
    if not rel_path.startswith(old_prefix):
        return ""
    tail = rel_path[len(old_prefix):].lstrip("/")
    for r in roots:
        p = Path(r) / tail
        if p.exists():
            return str(p)
    return ""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index",  required=True, help="index_nuscenes.with_tokens.jsonl")
    ap.add_argument("--out",    required=True, help="输出过滤后的 JSONL")
    ap.add_argument("--old_prefix", default="data/nuscenes/samples",
                    help="JSONL 里 cams 路径的共同前缀")
    ap.add_argument("--roots",  required=True,
                    help="用逗号分隔的本机 samples 根，例如："
                         '"/nas/.../data/nuscenes/samples,/nas/.../data/nuscenes-v1.0-trainval/samples"')
    ap.add_argument("--min_cams", type=int, default=3, help="至少几路相机存在才保留该样本")
    ap.add_argument("--rewrite_to_abs", action="store_true",
                    help="把 cams 字段改写成绝对路径（便于后续直接用）")
    args = ap.parse_args()

    roots = [x for x in args.roots.split(",") if x]
    Path(Path(args.out).parent).mkdir(parents=True, exist_ok=True)

    total, ok, miss_by_cam = 0, 0, {c:0 for c in CAMS_ORDER}
    with open(args.out, "w") as fw:
        for line in open(args.index):
            total += 1
            item = json.loads(line)
            found = {}
            for cam in CAMS_ORDER:
                rel = item["cams"][cam]
                abs_p = resolve_abs(rel, args.old_prefix, roots)
                if abs_p:
                    found[cam] = abs_p
                else:
                    miss_by_cam[cam] += 1
            if len(found) >= args.min_cams:
                if args.rewrite_to_abs:
                    item["cams"] = {k: found.get(k, "") for k in CAMS_ORDER}
                fw.write(json.dumps(item, ensure_ascii=False) + "\n")
                ok += 1

    print(f"[STAT] total={total}  kept={ok}  skipped={total-ok}  (min_cams={args.min_cams})")
    print("[MISS by cam]", miss_by_cam)
    print(f"[OK] wrote -> {args.out}")

if __name__ == "__main__":
    main()