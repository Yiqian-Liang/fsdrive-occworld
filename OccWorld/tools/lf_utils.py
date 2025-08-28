# tools/lf_utils.py (debug version)
import json, re, numpy as np

def _apply_perm(grid, order='F', do_transpose=True, flip_h=+1, flip_w=+1):
    # 默认 F + transpose + +1 + +1, 根据用户 match 测试
    if grid.ndim == 1: raise ValueError("internal: _apply_perm expects 2D array")
    if do_transpose:
        grid = grid.T
    if flip_h == -1:
        grid = grid[::-1, :].copy()  # .copy() 防负步长
    if flip_w == -1:
        grid = grid[:, ::-1].copy()
    return grid

def rle_pairs_to_grid(pairs, hw=(50,50), order='F', do_transpose=True, flip_h=+1, flip_w=+1):
    flat = []
    for cid, cnt in pairs:
        flat.extend([int(cid)] * int(cnt))
    arr = np.asarray(flat, np.int64)
    if arr.size != hw[0]*hw[1]:
        raise ValueError(f"RLE pairs decode to {arr.size}, expect {hw[0]*hw[1]}")
    grid = np.reshape(arr, hw, order=order)
    grid = _apply_perm(grid, order, do_transpose, flip_h, flip_w)
    return grid

def rle_text_to_grid(s: str, hw=(50,50), order='F', do_transpose=True, flip_h=+1, flip_w=+1):
    import re as _re
    toks = _re.findall(r'(\d+)\s*x\s*(\d+)', s)
    if not toks:
        raise ValueError("empty/invalid RLE text")
    flat=[]
    for a,b in toks:
        flat.extend([int(a)]*int(b))
    arr = np.asarray(flat, np.int64)
    if arr.size != hw[0]*hw[1]:
        raise ValueError(f"RLE text decode to {arr.size}, expect {hw[0]*hw[1]}")
    grid = np.reshape(arr, hw, order=order)
    grid = _apply_perm(grid, order, do_transpose, flip_h, flip_w)
    return grid

def csv_to_grid(s: str, hw=(50,50), order='F', do_transpose=True, flip_h=+1, flip_w=+1):
    nums = [int(x) for x in s.replace('\n',' ').replace('\t',' ').split(',') if x.strip()!='']
    arr = np.asarray(nums, np.int64)
    if arr.size != hw[0]*hw[1]:
        raise ValueError(f"CSV length {arr.size}, expect {hw[0]*hw[1]}")
    grid = np.reshape(arr, hw, order=order)
    grid = _apply_perm(grid, order, do_transpose, flip_h, flip_w)
    return grid

def parse_tokens_from_item(item, hw=(50,50), order='F', do_transpose=True, flip_h=+1, flip_w=+1):
    # 移除 offset 处理，避免双重 -1
    if 'target_rle' in item:
        raw = item['target_rle']
        pairs = raw if isinstance(raw, list) else json.loads(raw)
        grid = rle_pairs_to_grid(pairs, hw, order, do_transpose, flip_h, flip_w)
    elif 'target_csv' in item:
        grid = csv_to_grid(item['target_csv'], hw, order, do_transpose, flip_h, flip_w)
    else:
        conv = item.get('conversations', [])
        if not conv: raise ValueError("no conversations in item")
        msg = conv[-1]['value'] if conv[-1].get('from')=='gpt' else conv[0]['value']
        msg = msg.strip()
        grid = None
        try:
            parsed = json.loads(msg)
            if isinstance(parsed, list) and parsed and isinstance(parsed[0], list):
                grid = rle_pairs_to_grid(parsed, hw, order, do_transpose, flip_h, flip_w)
        except Exception:
            pass
        if grid is None and 'x' in msg:
            grid = rle_text_to_grid(msg, hw, order, do_transpose, flip_h, flip_w)
        if grid is None and ',' in msg:
            grid = csv_to_grid(msg, hw, order, do_transpose, flip_h, flip_w)
        if grid is None:
            raise ValueError("cannot parse tokens from conversations")
    # Debug print
    #print("tk50 min max:", grid.min(), grid.max())
    return grid

def guess_scene_from_image(path: str):
    m = re.search(r'(scene-\d+)', path)
    return m.group(1) if m else "unknown"