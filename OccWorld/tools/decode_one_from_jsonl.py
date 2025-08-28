# tools/decode_one_from_jsonl.py
import os, sys, json, argparse, numpy as np, torch, imageio
from pathlib import Path

# 让 mmengine 能 import 项目内模块
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

from mmengine.config import Config
from mmengine.registry import MODELS
from mmengine.runner import load_checkpoint

# 官方色系（0=白底，1..16按 OccWorld）
PALETTE = np.array([
    [255,255,255],
    [255,120,50],[255,192,203],[255,255,0],[0,150,245],[0,255,255],
    [255,127,0],[255,0,0],[255,240,150],[135,60,0],[160,32,240],
    [255,0,255],[139,137,137],[75,0,75],[150,240,80],[230,230,250],[0,175,0]
], np.uint8)

def save_png(path, idx2d):
    imageio.imwrite(path, PALETTE[np.clip(idx2d, 0, 16)])

def reduce64_to16_firstnz(sem64, flipz=False):
    s = sem64[..., ::-1] if flipz else sem64
    H,W,_ = s.shape
    g = s.reshape(H, W, 16, 4)
    out = np.zeros((H, W, 16), np.int64)
    for i in range(16):
        g4  = g[:, :, i, :]
        nz  = (g4 != 0)
        has = nz.any(-1)
        idx = nz.argmax(-1)
        pick = np.zeros((H, W), np.int64)
        pick[has] = g4[has, idx[has]]
        out[:, :, i] = pick
    return out

def topdown_first(vol_hwD, drop17=True):
    v = vol_hwD.copy()
    if drop17:
        v[v == 17] = 0
    H,W,D = v.shape
    col = v.reshape(H*W, D)
    nz  = (col != 0)
    has = nz.any(1)
    idx = nz.argmax(1)
    out = np.zeros(H*W, np.uint8)
    out[has] = col[np.arange(H*W), idx][has]
    return out.reshape(H, W)

def load_bev_mask_from_npz(npz_path, prefer="camera"):
    """返回 (H,W) bool 掩膜；支持 mask_camera / mask_lidar；否则 None。"""
    f = np.load(npz_path)
    if prefer == "union":
        a = f["mask_camera"] if "mask_camera" in f.files else None
        b = f["mask_lidar"] if "mask_lidar" in f.files else None
        if a is None and b is None:
            return None
        def to2d(m):
            if m is None: return None
            m = m.astype(bool)
            return m if m.ndim==2 else m.any(axis=-1)
        A, B = to2d(a), to2d(b)
        if A is None: return B
        if B is None: return A
        return A | B
    key = "mask_camera" if prefer=="camera" else ("mask_lidar" if prefer=="lidar" else None)
    if key and key in f.files:
        m = f[key].astype(bool)
        return m if m.ndim==2 else m.any(axis=-1)
    return None

def hist_top(a, m=None, k=8):
    if m is not None: a = a[m]
    u,c = np.unique(a, return_counts=True)
    pairs = [(int(x), int(n)) for x,n in zip(u,c) if x!=0]
    pairs.sort(key=lambda x: -x[1])
    return dict(pairs[:k])

def miou_2d(gt, pd, m=None):
    ious=[]
    for c in range(1,17):
        g = (gt==c); p=(pd==c)
        if m is not None: g, p = g & m, p & m
        inter = (g & p).sum()
        union = (g | p).sum()
        if union>0: ious.append(inter/union)
    return float(np.mean(ious)) if ious else 0.0

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--npz",   help="直接指定某一帧 GT 的 labels.npz")
    g.add_argument("--index", help="JSONL 索引文件（含 occ_gt_npz / occ_token_npz）")
    ap.add_argument("--i", type=int, default=0, help="读取 JSONL 第 i 行（仅 --index）")
    ap.add_argument("--tokens", help="与 --npz 对应的 token.npz（优先使用）")
    ap.add_argument("--tokens_root", default="tokens_official")

    ap.add_argument("--cfg",  default="config/occworld.py")
    ap.add_argument("--ckpt", default="out/occworld/latest.pth")
    ap.add_argument("--root", default=".", help="把 JSONL 相对路径拼到这个根目录（仅 --index）")
    ap.add_argument("--out",  default="quick_vis_png/from_npz_or_jsonl")

    ap.add_argument("--offset", type=int, default=1, help="token 是否 1-based（1=先减一）")
    ap.add_argument("--flipz",  type=int, default=0, help="64→16 时是否从地面往上找 first")
    ap.add_argument("--drop17", type=int, default=1, help="topdown 时把 17 当作空")
    ap.add_argument("--use_mask", choices=["none","camera","lidar","union"], default="camera",
                    help="只在掩膜区域统计与出 diff_inmask.png")
    ap.add_argument("--save_npys", action="store_true", help="额外保存 *_gt_top.npy / *_pd_top.npy")
    ap.add_argument("--oracle_from_gt", action="store_true",
                    help="再跑一条 GT→encoder→codebook→decoder 的上限，保存 *_oracle_top.png")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # 解析输入
    if args.npz:
        gt_npz = Path(args.npz)
        parts = gt_npz.parts
        scene = parts[-3] if len(parts)>=3 else "scene"
        sample_token = parts[-2] if len(parts)>=2 else "sample"
        if args.tokens:
            tok_npz = Path(args.tokens)
        else:
            tok_npz = Path(args.tokens_root) / scene / sample_token / "token.npz"
        if not tok_npz.exists():
            tried = [str(tok_npz)] + ([args.tokens] if args.tokens else [])
            raise FileNotFoundError("找不到 token.npz；请显式传 --tokens。\n我尝试了：\n  - " + "\n  - ".join(tried))
    else:
        root = Path(args.root)
        with open(args.index, "r") as f:
            for k, line in enumerate(f):
                if k == args.i:
                    item = json.loads(line); break
            else:
                raise IndexError(f"index {args.i} 超范围")
        gt_npz  = root / item["occ_gt_npz"]
        tok_npz = root / item["occ_token_npz"]
        scene, sample_token = item["scene"], item["sample_token"]

    # 载入 GT
    sem = np.load(gt_npz)["semantics"].astype(np.int64)  # (H,W,16/64)
    H,W,D = sem.shape
    if D == 64:
        gt16 = reduce64_to16_firstnz(sem, flipz=bool(args.flipz))
    elif D == 16:
        gt16 = sem
    else:
        raise ValueError(f"GT depth 异常: {sem.shape}")

    # 载入 token（统一到 50×50）
    tk = np.load(tok_npz)["token"]  # (1,50,50) 或 (1,100,100)
    assert tk.ndim == 3 and tk.shape[0] == 1, f"token 形状异常: {tk.shape}"
    ids = tk[0].astype(np.int64)
    if args.offset == 1:
        ids = np.maximum(ids - 1, 0)      # → 0-based
    if ids.shape == (100,100):
        # 下采样到 50×50（保持和官方链路一致）
        ids = torch.nn.functional.interpolate(
            torch.from_numpy(ids)[None,None].float(), size=(50,50), mode="nearest"
        ).long().squeeze().numpy()
    elif ids.shape != (50,50):
        raise ValueError(f"不支持的 token 分辨率: {ids.shape}（只接受 50×50 或 100×100）")

    # 构建 / 加载模型
    cfg = Config.fromfile(args.cfg)
    model = MODELS.build(cfg.model).eval()
    load_checkpoint(model, args.ckpt, map_location="cpu", strict=False)
    C = int(getattr(model.vae, "expansion", 8))  # 8

    # === token 一致性检查（GT → encoder → VQ）===
    x = torch.from_numpy(gt16[None, None]).long()                         # (1,1,H,W,16)
    x = model.vae.class_embeds(x).reshape(1, H, W, 16, C).permute(0,3,4,1,2).reshape(1, 16*C, H, W)
    z,_ = model.vae.encoder(x)
    z_e = model.vae.vqvae.quant_conv(z) if hasattr(model.vae.vqvae, "quant_conv") else z
    E   = model.vae.vqvae.embedding.weight                                # (K, Cq)
    zf   = z_e.permute(0,2,3,1).reshape(-1, z_e.shape[1])
    dist = (zf**2).sum(1, keepdims=True) - 2 * zf @ E.t() + (E**2).sum(1)[None, :]
    idx50 = dist.argmin(1).reshape(z_e.shape[0], z_e.shape[2], z_e.shape[3]).squeeze(0).cpu().numpy().astype(np.int64)
    # 对齐分辨率（我们此时 ids 一定是 50×50）
    match = (ids == idx50).mean()
    print(f"[check] token_match@50x50 = {match:.3f}  (1.000 表示完全一致)")

    # 可视化不一致（白=一致，红=不一致）
    mismatch = (ids != idx50).astype(np.uint8) * 255
    base = f"{scene}_{sample_token}"
    imageio.imwrite(f"{args.out}/{base}_token_mismatch.png",
                    np.stack([mismatch, np.zeros_like(mismatch), np.zeros_like(mismatch)], -1))

    # === 解码：直接使用 50×50 的 token（和 compare_fair 一致）===
    ids_50 = torch.from_numpy(ids).long()             # (50,50)
    code = model.vae.vqvae.embedding(ids_50)          # (50,50,Cq)
    code = code.permute(2,0,1).unsqueeze(0)           # (1,Cq,50,50)
    if hasattr(model.vae.vqvae, "post_quant_conv"):
        code = model.vae.vqvae.post_quant_conv(code)
    dec  = model.vae.decoder(code, [torch.Size([200,200]), torch.Size([100,100])])  # (1,128,200,200)

    feat   = dec.permute(0,2,3,1).reshape(-1, 16, C)  # (-1,16,8)
    Wcls   = model.vae.class_embeds.weight.T.unsqueeze(0)
    pred16 = torch.matmul(feat, Wcls).argmax(-1).view(200,200,16).cpu().numpy().astype(np.uint8)

    # 顶视
    gt_top = topdown_first(gt16, drop17=bool(args.drop17))
    pd_top = topdown_first(pred16, drop17=bool(args.drop17))
    save_png(f"{args.out}/{base}_gt_top.png",  gt_top)
    save_png(f"{args.out}/{base}_pd_top.png",  pd_top)
    if args.save_npys:
        np.save(f"{args.out}/{base}_gt_top.npy", gt_top)
        np.save(f"{args.out}/{base}_pd_top.npy", pd_top)

    # 掩膜
    mask = None if args.use_mask=="none" else load_bev_mask_from_npz(gt_npz, prefer=args.use_mask)
    if mask is None:
        mask = np.ones_like(gt_top, dtype=bool)
    imageio.imwrite(f"{args.out}/{base}_mask.png", (mask.astype(np.uint8)*255))

    # diff（全图 & in-mask）
    miss = (gt_top!=0) & (pd_top==0)
    fp   = (gt_top==0) & (pd_top!=0)
    diff_full = np.full((*gt_top.shape,3), PALETTE[0], np.uint8)
    diff_full[miss] = [255, 0, 0]; diff_full[fp] = [0, 255, 0]
    imageio.imwrite(f"{args.out}/{base}_diff.png", diff_full)
    diff_m = np.full((*gt_top.shape,3), PALETTE[0], np.uint8)
    diff_m[miss & mask] = [255, 0, 0]; diff_m[fp & mask] = [0, 255, 0]
    imageio.imwrite(f"{args.out}/{base}_diff_inmask.png", diff_m)

    # 指标
    cov_full = lambda a: (a!=0).mean()
    cov_mask = lambda a: (a[mask]!=0).mean()
    miou_m   = miou_2d(gt_top, pd_top, mask)

    print(f"[OK] scene={scene} token={sample_token}")
    print(f"  cover(full): GT={cov_full(gt_top):.3f}  PD={cov_full(pd_top):.3f}")
    print(f"  cover(mask): GT={cov_mask(gt_top):.3f}  PD={cov_mask(pd_top):.3f}   use_mask={args.use_mask}")
    print(f"  mIoU(mask, 1..16, topdown) = {miou_m:.3f}")
    print(f"  GT-hist(mask): {hist_top(gt_top, mask)}")
    print(f"  PD-hist(mask): {hist_top(pd_top, mask)}")
    print(f"  token.shape={tk.shape}, id_range=[{ids.min()}..{ids.max()}], offset={args.offset}")
    print(f"  saved to: {args.out}")

    # 可选：oracle 上限（把 GT 走一遍 encoder→VQ→decoder）
    if args.oracle_from_gt:
        x = torch.from_numpy(gt16[None,None]).long()
        x = model.vae.class_embeds(x).reshape(1,H,W,16,C).permute(0,3,4,1,2).reshape(1,16*C,H,W)
        z,_  = model.vae.encoder(x)
        z_e  = model.vae.vqvae.quant_conv(z) if hasattr(model.vae.vqvae,"quant_conv") else z
        E    = model.vae.vqvae.embedding.weight
        zf   = z_e.permute(0,2,3,1).reshape(-1, z_e.shape[1])
        dist = (zf**2).sum(1,keepdims=True) - 2* zf @ E.t() + (E**2).sum(1)[None,:]
        idx  = dist.argmin(1).reshape(1, z_e.shape[2], z_e.shape[3])  # (1,50,50)
        code = model.vae.vqvae.embedding(idx).permute(0,3,1,2)        # (1,Cq,50,50)
        if hasattr(model.vae.vqvae, "post_quant_conv"):
            code = model.vae.vqvae.post_quant_conv(code)
        dec2 = model.vae.decoder(code, [torch.Size([200,200]), torch.Size([100,100])])
        feat2 = dec2.permute(0,2,3,1).reshape(-1,16,C)
        pred2 = torch.matmul(feat2, model.vae.class_embeds.weight.T.unsqueeze(0)).argmax(-1)
        pred2 = pred2.view(200,200,16).cpu().numpy().astype(np.uint8)
        oracle_top = topdown_first(pred2, drop17=bool(args.drop17))
        save_png(f"{args.out}/{base}_oracle_top.png", oracle_top)
        print(f"  oracle mIoU(mask) = {miou_2d(gt_top, oracle_top, mask):.3f}  （上限：token 量化带来的信息损失以外）")

if __name__ == "__main__":
    main()
