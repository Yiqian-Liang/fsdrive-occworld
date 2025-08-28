# visualize_gt_3d.py
import os, glob, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

VOX_ORIGIN = np.array([-40.0, -40.0, -1.0], dtype=np.float32)  # 世界坐标原点
VOX_SIZE   = np.array([  0.4,   0.4,  0.4], dtype=np.float32)  # (x,y,z) 每格 0.4m
LUT = np.array(
 [[255,120, 50],[255,192,203],[255,255,  0],[  0,150,245],
  [  0,255,255],[255,127,  0],[255,  0,  0],[255,240,150],
  [135, 60,  0],[160, 32,240],[255,  0,255],[139,137,137],
  [ 75,  0, 75],[150,240, 80],[230,230,250],[  0,175,  0],
  [  0,  0,  0],[  0,  0,  0]], dtype=np.uint8) / 255.0

def find_all_gt(root="data/nuscenes/gts", limit=1):
    cands = sorted(glob.glob(os.path.join(root, "**", "labels.npz"), recursive=True))
    if not cands:
        raise FileNotFoundError(f"No labels.npz under {root}")
    return cands[:limit]

def to_DHW(arr):
    if arr.ndim == 3:
        if arr.shape[2] in (16,56,60,64):  # (H,W,D) -> (D,H,W)
            arr = arr.transpose(2,0,1)
    elif arr.ndim == 4:
        arr = arr[0]
    else:
        raise ValueError(f"Unexpected semantics shape {arr.shape}")
    return arr

def dhw_to_points(sem):
    D,H,W = sem.shape
    z,y,x = np.nonzero((sem>0)&(sem<17))
    cls = sem[z,y,x]
    X = x*VOX_SIZE[0] + VOX_ORIGIN[0]
    Y = y*VOX_SIZE[1] + VOX_ORIGIN[1]
    Z = z*VOX_SIZE[2] + VOX_ORIGIN[2]
    return X,Y,Z,cls

def main():
    out_dir = "./vis_gt_3d"; os.makedirs(out_dir, exist_ok=True)
    for i, p in enumerate(find_all_gt(limit=3)):
        sem = to_DHW(np.load(p)["semantics"]).astype(np.uint8)
        X,Y,Z,cls = dhw_to_points(sem)
        # 采样避免太密
        max_pts = 50000
        if len(cls) > max_pts:
            idx = np.random.choice(len(cls), max_pts, replace=False)
            X,Y,Z,cls = X[idx],Y[idx],Z[idx],cls[idx]

        fig = plt.figure(figsize=(8,6))
        ax  = fig.add_subplot(111, projection='3d')
        ax.scatter(X,Y,Z, s=2, c=LUT[cls], marker='s', linewidths=0)
        ax.set_xlim(VOX_ORIGIN[0], VOX_ORIGIN[0] + sem.shape[2]*VOX_SIZE[0])
        ax.set_ylim(VOX_ORIGIN[1], VOX_ORIGIN[1] + sem.shape[1]*VOX_SIZE[1])
        ax.set_zlim(VOX_ORIGIN[2], VOX_ORIGIN[2] + sem.shape[0]*VOX_SIZE[2])
        ax.set_axis_off(); plt.tight_layout()
        out = os.path.join(out_dir, f"gt_3d_{i:02d}.png")
        plt.savefig(out, dpi=300); plt.close(fig)
        print("✓ saved:", out)

if __name__ == "__main__":
    main()
