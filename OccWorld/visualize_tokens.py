# visualize_tokens.py
import os, numpy as np, torch, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mmengine import Config
from mmengine.registry import MODELS

# ------- 路径配置 -------
config_path = 'config/occworld.py'
ckpt_path   = '/nas/vlm_driving/OccWorld/out/vqvae/epoch_125.pth'
npz_path    = '/nas/vlm_driving/OccWorld/tokens_dev/unknown_scene/1754123624033458538/token.npz'
save_dir    = './vis_token_decode_mpl'
os.makedirs(save_dir, exist_ok=True)

# ------- 颜色 LUT（1..16） -------
lut = np.array(
 [[255,120,50],[255,192,203],[255,255,0],[0,150,245],[0,255,255],[255,127,0],
  [255,0,0],[255,240,150],[135,60,0],[160,32,240],[255,0,255],[139,137,137],
  [75,0,75],[150,240,80],[230,230,250],[0,175,0],[0,0,0],[0,0,0]], dtype=np.uint8)/255.0

VOX_ORIGIN = np.array([-40.0, -40.0, -1.0], dtype=np.float32)
VOX_SIZE   = np.array([  0.4,   0.4,  0.4], dtype=np.float32)

def voxels_to_points(vox):  # (D,H,W) -> 点云 (X,Y,Z,cls)，过滤 0/17
    D,H,W = vox.shape
    z,y,x = np.nonzero(vox)
    cls   = vox[z,y,x]
    m = (cls > 0) & (cls < 17)
    x,y,z,cls = x[m], y[m], z[m], cls[m]
    X = x*VOX_SIZE[0] + VOX_ORIGIN[0]
    Y = y*VOX_SIZE[1] + VOX_ORIGIN[1]
    Z = z*VOX_SIZE[2] + VOX_ORIGIN[2]
    return X,Y,Z,cls

# ------- 1) 读 token -------
tokens = np.load(npz_path)['token']              # (12, 50, 50)

# ------- 2) 载入模型 -------
cfg = Config.fromfile(config_path)
model = MODELS.build(cfg.model).eval().cuda()
model.load_state_dict(torch.load(ckpt_path, map_location='cpu'), strict=False)

# ------- 3) token -> embedding -> decoder -------
tk  = torch.from_numpy(tokens.astype(np.int32)).unsqueeze(0).long().cuda()     # (1,12,50,50)
emb = model.vae.vqvae.embedding(tk.view(1, -1))                                # (1,30000,128)
emb = emb.permute(0,2,1).contiguous().view(1,128,12,50,50)                     # (1,128,12,50,50)

# decoder 的金字塔上采样到 200x200
shapes = [(50,50), (100,100), (200,200)]
dec_logits = []
for t in range(12):
    z = model.vae.forward_decoder(emb[:,:,t], shapes.copy(), [1,1,200,200,64])  # ⚠️ H,W=200,200
    dec_logits.append(z)  # [1,200,200,64,18]
dec_logits = torch.cat(dec_logits, dim=0)          # [12,200,200,64,18]
pred = dec_logits.argmax(dim=-1).permute(0,3,1,2)  # -> [12,64,200,200] = (T,D,H,W)
pred = pred.detach().cpu().numpy()

# ------- 4) 先画“单帧” -------
t_idx = 5                      # 你可以改
vox   = pred[t_idx]            # (64,200,200)
X,Y,Z,cls = voxels_to_points(vox)

plt.figure(figsize=(8,6))
plt.scatter(X, Y, s=0.8, c=lut[cls], marker='s', linewidths=0)  # 俯视图；要 3D 可换成 Axes3D
plt.xlim(-40,40); plt.ylim(-40,40)
plt.axis('off'); plt.tight_layout()
plt.savefig(os.path.join(save_dir, f'single_frame_{t_idx}.png'), dpi=300)
plt.close()

# ------- 5) 循环保存 12 帧（俯视） -------
max_pts = 50000
for t in range(12):
    vox = pred[t]
    X,Y,Z,cls = voxels_to_points(vox)
    if X.size > max_pts:  # 采样避免太密
        idx = np.random.choice(X.size, max_pts, replace=False)
        X,Y,Z,cls = X[idx],Y[idx],Z[idx],cls[idx]
    plt.figure(figsize=(8,6))
    plt.scatter(X, Y, s=0.8, c=lut[cls], marker='s', linewidths=0)
    plt.xlim(-40,40); plt.ylim(-40,40)
    plt.axis('off'); plt.tight_layout()
    out = os.path.join(save_dir, f'frame_{t:02d}.png')
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"[saved] {out}")

print("✅ 完成：", save_dir)
