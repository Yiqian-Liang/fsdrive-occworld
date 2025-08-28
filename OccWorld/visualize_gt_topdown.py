# visualize_gt_topdown.py
import os, glob, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 18 类（含 0/17 背景/空）— 你可按项目 LUT 改
LUT = np.array(
 [[255,120, 50],[255,192,203],[255,255,  0],[  0,150,245],
  [  0,255,255],[255,127,  0],[255,  0,  0],[255,240,150],
  [135, 60,  0],[160, 32,240],[255,  0,255],[139,137,137],
  [ 75,  0, 75],[150,240, 80],[230,230,250],[  0,175,  0],
  [  0,  0,  0],[  0,  0,  0]], dtype=np.uint8)

def find_one_gt(root="data/nuscenes/gts"):
    cands = sorted(glob.glob(os.path.join(root, "**", "labels.npz"), recursive=True))
    if not cands:
        raise FileNotFoundError(f"No labels.npz under {root}")
    return cands[0]

def load_semantics(path):
    arr = np.load(path)["semantics"]  # 常见形状： (D,H,W) 或 (H,W,D)
    if arr.ndim == 3:
        # 统一到 (D,H,W)
        D,H,W = arr.shape
        # 这份数据普遍是 (H,W,D) 或 (D,H,W)，按最后一维=深度判断
        if arr.shape[2] in (16,56,60,64):   # 你数据常见 D
            arr = arr.transpose(2,0,1)      # (H,W,D)->(D,H,W)
    elif arr.ndim == 4:
        # 有的存成 (T, D, H, W)；这里只看第 0 帧
        arr = arr[0]
    else:
        raise ValueError(f"Unexpected semantics shape {arr.shape}")
    return arr.astype(np.uint8)             # (D,H,W)

def topdown_color(sem_dhw):
    # 只保留 1..16，顶层“最大类别”着色（沿 Z 方向做一个投影）
    D,H,W = sem_dhw.shape
    img = np.zeros((H,W,3), dtype=np.uint8)
    # 从上往下找第一个非(0/17)的类
    valid = ((sem_dhw>0) & (sem_dhw<17))
    has = valid.any(axis=0)                 # (H,W)
    first_idx = np.argmax(valid, axis=0)    # 第一次出现的 z
    cls = sem_dhw[first_idx, np.arange(H)[:,None], np.arange(W)[None,:]]
    cls[~has] = 0
    img = LUT[cls]                          # (H,W,3)
    return img

def main():
    out_dir = "./vis_gt_png"; os.makedirs(out_dir, exist_ok=True)
    gt_path = find_one_gt()
    print("[FOUND]", gt_path)
    sem = load_semantics(gt_path)          # (D,H,W)
    print("[INFO] sem shape:", sem.shape, sem.dtype)

    col = topdown_color(sem)
    out = os.path.join(out_dir, "gt_topdown.png")
    plt.imsave(out, col)
    print("✓ saved:", out)

if __name__ == "__main__":
    main()
