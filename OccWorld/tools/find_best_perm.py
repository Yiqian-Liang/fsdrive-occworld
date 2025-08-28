# tools/find_best_perm.py
import os, json, numpy as np, re, sys
from itertools import product

def expand_pairs(pairs):
    arr=[]
    for i,c in pairs: arr.extend([int(i)]*int(c))
    return np.array(arr, np.int64)

def parse_pairs_from_conv(v: str):
    # 从 [[id,count], ...] 文本提取 pair
    pairs = [tuple(map(int, m.groups()))
             for m in re.finditer(r'\[\s*(\d+)\s*,\s*(\d+)\s*\]', v)]
    return pairs

def load_one_from_lf(item):
    conv = item["conversations"][-1] if item["conversations"][-1]["from"]=="gpt" else item["conversations"][0]
    pairs = parse_pairs_from_conv(conv["value"])
    tk = expand_pairs(pairs)         # 长度应该是 2500
    if tk.size != 2500:
        raise SystemExit(f"RLE expand size={tk.size}, expected 2500")
    return tk

def grid_ops(grid):
    g = grid.reshape(50,50)
    for fill in ("C","F"):
        a = grid.reshape(50,50, order=fill)
        for t in [lambda x:x, np.transpose]:
            b = t(a)
            for vh in [(1,1),(1,-1),(-1,1),(-1,-1)]:  # 翻转
                yield (fill, t.__name__, vh), b[::vh[0], ::vh[1]]

if __name__=="__main__":
    lf = sys.argv[1]
    root = sys.argv[2]  # /nas/vlm_driving/OccWorld
    items = json.load(open(lf))
    item = items[0]     # 先抽第一条
    img  = item["images"][0] if isinstance(item["images"], list) else item["images"]
    bn   = os.path.basename(img).replace(".jpg","")
    scene= img.split("/")[-2]  # scene-xxxx
    npz  = f"{root}/tokens_official/{scene}/{bn}/token.npz"
    tk_json_1d = load_one_from_lf(item)
    tk_npz = np.load(npz)["token"][0].astype(np.int64) - 1  # 1-based -> 0-based

    best = (0.0, None)
    for (fill, tname, (vh, vv)), G in grid_ops(tk_json_1d):
        match = (G == tk_npz).mean()
        best = max(best, (match, (fill, tname, vh, vv)))
        print(f"order={fill:1}  op={tname:9}  flip=({vh:+d},{vv:+d})  match={match:.4f}")
    print("\nBEST:", best)
