# tools/rewrite_lf_json_canonical.py
import json, re, argparse
import numpy as np

def rle_pairs_to_grid(pairs, hw=(50,50)):
    flat=[]
    for cls,cnt in pairs:
        flat.extend([int(cls)]*int(cnt))
    arr=np.asarray(flat, np.int64)
    if arr.size != hw[0]*hw[1]:
        raise ValueError(f"RLE size {arr.size} != {hw[0]*hw[1]}")
    return arr.reshape(hw)

def rle_text_to_grid(s, hw=(50,50)):
    toks=re.findall(r'(\d+)\s*x\s*(\d+)', s)
    if not toks: raise ValueError("invalid RLE text")
    flat=[]
    for a,b in toks:
        flat.extend([int(a)]*int(b))
    arr=np.asarray(flat, np.int64)
    if arr.size != hw[0]*hw[1]:
        raise ValueError(f"RLE size {arr.size} != {hw[0]*hw[1]}")
    return arr.reshape(hw)

def csv_to_grid(s, hw=(50,50)):
    nums=[int(x) for x in s.replace('\n',' ').replace('\t',' ').split(',') if x.strip()!='']
    arr=np.asarray(nums, np.int64)
    if arr.size != hw[0]*hw[1]:
        raise ValueError(f"CSV size {arr.size} != {hw[0]*hw[1]}")
    return arr.reshape(hw)

def parse_grid_from_item(item, hw=(50,50), offset=1):
    # 优先直接字段（若你之前已经加过 target_rle / target_csv）
    if 'target_rle' in item:
        raw=item['target_rle']; pairs=raw if isinstance(raw,list) else json.loads(raw)
        grid=rle_pairs_to_grid(pairs, hw)
    elif 'target_csv' in item:
        grid=csv_to_grid(item['target_csv'], hw)
    else:
        conv=item.get('conversations',[])
        if not conv: raise ValueError("no conversations")
        msg=conv[-1]['value'] if conv[-1].get('from')=='gpt' else conv[0]['value']
        msg=msg.strip()
        grid=None
        # try JSON list-of-lists
        try:
            parsed=json.loads(msg)
            if isinstance(parsed,list) and parsed and isinstance(parsed[0],list):
                grid=rle_pairs_to_grid(parsed, hw)
        except Exception:
            pass
        if grid is None and 'x' in msg:
            grid=rle_text_to_grid(msg, hw)
        if grid is None and ',' in msg and 'x' not in msg:
            grid=csv_to_grid(msg, hw)
        if grid is None:
            raise ValueError("cannot parse tokens")
    # 把 1-based 变 0-based（你的数据基本是 1..512）
    if offset==1:
        grid=np.maximum(grid-1, 0)
    return grid

def grid_to_rle_pairs(grid):
    # 以 C-order(行优先，无转置) 扫描，压成 [val,count] 列表
    flat=grid.reshape(-1)
    out=[]
    if flat.size==0: return out
    cur=int(flat[0]); cnt=1
    for v in flat[1:]:
        v=int(v)
        if v==cur: cnt+=1
        else:
            out.append([cur, cnt])
            cur=v; cnt=1
    out.append([cur, cnt])
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--in_json", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--hw", type=int, nargs=2, default=[50,50])
    args=ap.parse_args()
    items=json.load(open(args.in_json))
    new_items=[]
    bad=0
    for it in items:
        try:
            grid=parse_grid_from_item(it, hw=tuple(args.hw), offset=1)  # 读入→0-based
            # 规范化：统一按 C-order(行优先) 写回 RLE
            pairs=grid_to_rle_pairs(grid.astype(np.int64))
            it2=dict(it)  # 保留原字段
            it2["target_rle"]=pairs
            # 可选：写 scene，便于后面按 scene 切分
            if "scene" not in it2:
                m=re.search(r'(scene-\d+)', (it2["images"][0] if isinstance(it2["images"],list) else it2["images"]))
                if m: it2["scene"]=m.group(1)
            new_items.append(it2)
        except Exception as e:
            bad+=1
    json.dump(new_items, open(args.out_json,"w"))
    print(f"done: {len(new_items)} items, skipped {bad}")

if __name__=="__main__":
    main()
