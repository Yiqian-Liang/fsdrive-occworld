# quick_decode_tokens.py
import os, argparse, numpy as np, torch
from mmengine.config import Config
from mmengine.runner import load_checkpoint
from mmengine.registry import MODELS

# ---- 可改参数 ----
VOX_ORIGIN = np.array([-40, -40, -1.0], dtype=np.float32)
VOX_SIZE   = np.array([0.4, 0.4, 0.4], dtype=np.float32)  # (W,H,D)步长
PALETTE = np.array([
    [0,0,0], [128,64,128],[244,35,232],[70,70,70],[102,102,156],
    [190,153,153],[153,153,153],[250,170,30],[220,220,0],[107,142,35],
    [152,251,152],[70,130,180],[220,20,60],[255,0,0],[0,0,142],
    [0,0,70],[0,60,100],[0,80,100],[0,0,230]    # 0..18（0=ignore）
], dtype=np.uint8)

def colorize_topdown(lbl_hw):
    # lbl_hw: (H,W) uint8 取值 0..18
    c = PALETTE[np.clip(lbl_hw, 0, len(PALETTE)-1)]
    return c  # (H,W,3)

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="config/occworld.py")
    ap.add_argument("--ckpt", default="out/occworld/latest.pth")
    ap.add_argument("--tokens", required=True, help="tokens.npy, 形状 (T,50,50)")
    ap.add_argument("--t", type=int, default=0, help="解码第 t 帧")
    ap.add_argument("--outdir", default="./quick_vis_png")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # 1) 构建模型并加载 ckpt（这是完整 OccWorld，包括 vae 与 transformer）
    cfg = Config.fromfile(args.cfg)
    model = MODELS.build(cfg.model)
    model.eval()
    load_checkpoint(model, args.ckpt, map_location="cpu", strict=False)

    assert hasattr(model, "vae") and hasattr(model.vae, "vqvae"), \
        "这个 ckpt 里没有 vae.vqvae，换用含 VQ-VAE 的权重/配置。"

    # 2) 读 token
    tk = np.load(args.tokens)              # (T,50,50)
    assert tk.ndim == 3 and tk.shape[1:] == (50,50), f"tokens 形状不对: {tk.shape}"
    T = tk.shape[0]
    t = max(0, min(args.t, T-1))
    ids = torch.from_numpy(tk[t]).long().contiguous()  # (50,50)

    # 一些数据集里 0 是 “占位/无效”，codebook 从 1..K-1 开始；做个安全裁剪
    K = model.vae.vqvae.embedding.weight.shape[0]
    ids = ids.clamp(0, K-1)

    # 3) 用 codebook 把 token → 向量，再过 post_quant_conv 得到 z，送入 2D decoder
    #    训练时的空间：输入分辨率 200×200，下采样两次 → 100×100 latent
    #    我们把 50×50 的 token 先上采样到 100×100（与训练对齐）
    ids_100 = torch.nn.functional.interpolate(
        ids[None,None].float(), size=(100,100), mode="nearest"
    ).long().squeeze(0).squeeze(0)    # (100,100)

    # codebook lookup: (H,W,Cq)
    code = model.vae.vqvae.embedding(ids_100)          # (100,100,Cq)
    code = code.permute(2,0,1).unsqueeze(0)            # (1,Cq,H,W)
    if hasattr(model.vae.vqvae, "post_quant_conv"):
        code = model.vae.vqvae.post_quant_conv(code)   # (1,C,H,W)

    # 4) 过 2D decoder 得到 per-voxel 的 8 维语义特征，再用 class_embeds 投到 18 类 logits
    shapes = [torch.Size([200,200]), torch.Size([100,100])]  # 与训练时 Encoder 记录一致
    feat = model.vae.decoder(code, shapes.copy())            # (1, C_out, 200, 200)
    feat = feat.permute(0,2,3,1).reshape(-1, 64, model.vae.expansion)  # (-1, D=64, 8)

    # 与类向量做相似度，得到 (-1,64,18)；再取 argmax 成标签
    template = model.vae.class_embeds.weight.T.unsqueeze(0)  # (1, 8, 18)
    logits = torch.matmul(feat, template)                    # (-1,64,18)
    pred = logits.argmax(dim=-1)                             # (-1,64)

    # 5) 把柱状 D=64 做个“俯视”融合（max/近地优先），得到 (200,200) 的 top-down 标签
    H,W = 200,200
    pred = pred.view(H*W, 64)
    # 这里用“离地面最近优先”：z=0..63，遇到第一个非背景(>0)就取；否则取全柱最大众
    pred_top = torch.zeros(H*W, dtype=torch.long)
    bg = (pred == 0)
    first_non_bg = (~bg).float().argmax(dim=1)   # 第一次非0出现的位置
    has_non_bg = (~bg).any(dim=1)
    idx = first_non_bg.clamp(max=63)
    gathered = pred[torch.arange(H*W), idx]
    pred_top[has_non_bg] = gathered[has_non_bg]
    # 背景柱用众数（这里用 mode 的近似：取出现次数最多的类别）
    if (~has_non_bg).any():
        # 简单用列直方图的 argmax（效率够用）
        counts = torch.stack([ (pred[~has_non_bg]==c).sum(dim=1) for c in range(18) ], dim=1)
        pred_top[~has_non_bg] = counts.argmax(dim=1)

    top = pred_top.view(H,W).cpu().numpy().astype(np.uint8)

    # 6) 上色并保存
    rgb = colorize_topdown(top)
    import imageio
    out_png = os.path.join(args.outdir, f"topdown_t{t:02d}.png")
    imageio.imwrite(out_png, rgb)
    print(f"✓ saved: {out_png}")

if __name__ == "__main__":
    main()
