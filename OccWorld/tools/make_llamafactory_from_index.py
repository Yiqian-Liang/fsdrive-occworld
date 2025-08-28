# -*- coding: utf-8 -*-
"""
把 index_nuscenes.with_tokens.jsonl 转成 LLaMA-Factory 多模态 JSON：
- 2x3 拼接 6 路相机（支持缺图跳过或占位补齐）
- token(50x50) 输出为 RLE 或扁平 2500 ids
- 兼容 Python 3.8（不使用 list[str] 新语法）
"""
import os, json, argparse, numpy as np
from pathlib import Path
from typing import List, Dict
from PIL import Image

CAMS_ORDER = [
    "CAM_FRONT_LEFT","CAM_FRONT","CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT","CAM_BACK","CAM_BACK_RIGHT"
]

def rle_encode(flat_ids: np.ndarray):
    if flat_ids.size == 0: return []
    runs, cur, cnt = [], int(flat_ids[0]), 1
    for x in flat_ids[1:]:
        x = int(x)
        if x == cur: cnt += 1
        else: runs.append([cur, cnt]); cur, cnt = x, 1
    runs.append([cur, cnt])
    return runs

def stitch_6cams(six_paths: Dict[str,str], out_path: str,
                 tile=(2,3), resize=(640,360), placeholder=None):
    W, H = resize
    canvas = Image.new("RGB", (tile[1]*W, tile[0]*H), (0,0,0))
    for i, cam in enumerate(CAMS_ORDER):
        r, c = divmod(i, tile[1])
        p = six_paths.get(cam)
        if p and Path(p).exists():
            im = Image.open(p).convert("RGB").resize((W,H))
        else:
            if placeholder and Path(placeholder).exists():
                im = Image.open(placeholder).convert("RGB").resize((W,H))
            else:
                im = Image.new("RGB", (W,H), (0,0,0))
        canvas.paste(im, (c*W, r*H))
    out_dir = os.path.dirname(out_path)
    if out_dir: os.makedirs(out_dir, exist_ok=True)
    canvas.save(out_path, quality=95)

def resolve_one(rel_path: str, old_prefix: str, roots: List[str]) -> str:
    """
    把 JSONL 里的相机相对路径(以 old_prefix 开头)映射到多个真实根之一。
    若都不存在，返回拼在第一个 root 下的候选路径（用于报缺）。
    """
    old = old_prefix.rstrip("/") + "/"
    if not rel_path.startswith(old):
        # 容错：有些索引里可能就是绝对路径
        return rel_path
    tail = rel_path[len(old):]  # 例如 CAM_FRONT/xxx.jpg
    for r in roots:
        cand = str(Path(r) / tail)
        if os.path.exists(cand):
            return cand
    return str(Path(roots[0]) / tail)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True, help="index_nuscenes.with_tokens.jsonl")
    ap.add_argument("--root",  default=".", help="把 JSONL 里的相对路径(occ_token_npz 等)拼到这个根目录")
    ap.add_argument("--out_json", required=True, help="输出 LLaMA-Factory JSON")
    ap.add_argument("--images_root", default="vis/stitched", help="拼接图保存根（相对 --root）")
    ap.add_argument("--token_format", choices=["rle","flat"], default="rle")
    ap.add_argument("--offset", type=int, default=1, help="token 是否 1-based（1=先减1）")
    ap.add_argument("--tokens_root", default="/nas/vlm_driving/OccWorld/tokens_official", help="tokens_official 的绝对路径（修正报错）")

    # 相机路径映射
    ap.add_argument("--cams_old_prefix", default="data/nuscenes/samples",
                    help="JSONL 中相机路径的前缀")
    ap.add_argument("--cams_roots", default="/nas/vlm_driving/OccWorld/data/nuscenes/samples,"
                                            "/nas/vlm_driving/OccWorld/data/nuscenes-v1.0-trainval/samples,"
                                            "/nas/vlm_driving/OccWorld/data/nuscenes-mini/samples",
                    help="逗号分隔的真实根，按序兜底")

    # 缺图策略
    ap.add_argument("--min_cams", type=int, default=6, help="至少存在几路相机才保留样本")
    ap.add_argument("--skip_missing", action="store_true", help="不足 min_cams 就跳过")
    ap.add_argument("--use_placeholder", action="store_true", help="不足也保留，用占位图补齐")
    ap.add_argument("--placeholder_img", default="", help="占位图路径（配合 --use_placeholder）")

    # 拼图分辨率
    ap.add_argument("--tile_w", type=int, default=640)
    ap.add_argument("--tile_h", type=int, default=360)

    args = ap.parse_args()
    cams_roots = [x.strip() for x in args.cams_roots.split(",") if x.strip()]
    root = Path(args.root)

    out_samples = []
    total = ok = skipped = 0
    missing_dump = []
    miss_counter = {c:0 for c in CAMS_ORDER}

    with open(args.index, "r") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            j = json.loads(line); total += 1

            # 1) 解析 6 路相机真实路径
            imgs = {}
            ok_cnt = 0
            for cam in CAMS_ORDER:
                rel = j["cams"][cam]  # 例如 data/nuscenes/samples/CAM_FRONT/xxx.jpg
                abs_p = resolve_one(rel, args.cams_old_prefix, cams_roots)
                imgs[cam] = abs_p
                if os.path.exists(abs_p):
                    ok_cnt += 1
                else:
                    miss_counter[cam] += 1

            # 2) 保留/跳过
            if ok_cnt < args.min_cams and not args.use_placeholder:
                if args.skip_missing:
                    skipped += 1
                    missing_dump.append({
                        "scene": j["scene"], "sample": j["sample_token"], "cams": imgs
                    })
                    continue

            # 3) 输出 stitched 图片
            out_img_rel = f'{args.images_root}/{j["scene"]}/{j["sample_token"]}.jpg'
            out_img_abs = root / out_img_rel
            if not out_img_abs.exists():
                stitch_6cams(
                    imgs, str(out_img_abs),
                    tile=(2,3),
                    resize=(args.tile_w, args.tile_h),
                    placeholder=(args.placeholder_img if args.use_placeholder else None)
                )

            # 4) 读 token → 文本答案
            # 修正路径：替换 JSONL 中的 /tokens_official 为实际绝对路径
            token_path = j["occ_token_npz"].replace("/tokens_official", args.tokens_root)
            tk = np.load(token_path)["token"][0]  # (50,50)
            if args.offset == 1:
                tk = np.maximum(tk.astype(np.int64) - 1, 0)
            flat = tk.reshape(-1)

            if args.token_format == "rle":
                answer = json.dumps(rle_encode(flat), ensure_ascii=False)
                prompt = "<image>\nPredict occupancy tokens (size 50x50).\nReturn RLE pairs [[id,count],...], 0-based."
            else:
                answer = " ".join(str(int(x)) for x in flat.tolist())
                prompt = "<image>\nPredict occupancy tokens (size 50x50).\nReturn 2500 integers, 0-based."

            out_samples.append({
                "conversations": [
                    {"from": "human", "value": prompt},
                    {"from": "gpt",   "value": answer}
                ],
                "images": [out_img_rel]
            })
            ok += 1

    # 5) 写出
    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as fo:
        json.dump(out_samples, fo, ensure_ascii=False, indent=2)

    print(f"[OK] wrote {ok} samples -> {args.out_json}")
    print(f"[STAT] total={total} ok={ok} skipped={skipped}  (min_cams={args.min_cams}, skip_missing={args.skip_missing})")
    print("[MISS by cam]", miss_counter)
    if skipped:
        miss = os.path.splitext(args.out_json)[0] + ".missing.json"
        with open(miss, "w", encoding="utf-8") as fo:
            json.dump(missing_dump, fo, ensure_ascii=False, indent=2)
        print(f"[HINT] Missing cam list -> {miss}")
    print("[TIP] 如果 ok 太少：\n"
          "  1) 调小 --min_cams（如 4 或 3）\n"
          "  2) 补充 --cams_roots，把你机器上所有可能的数据根都加进去\n"
          "  3) 仅用前向 3 路也行：把 CAMS_ORDER 改成前三路，并把 --min_cams 设为 3")
if __name__ == "__main__":
    main()