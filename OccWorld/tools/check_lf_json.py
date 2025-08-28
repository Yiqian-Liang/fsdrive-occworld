# tools/check_lf_json.py  —— decode_one_from_jsonl 对齐版
import os, json, argparse, numpy as np, imageio, torch
from mmengine.config import Config
from mmengine.registry import MODELS
from mmengine.runner import load_checkpoint
from lf_utils import parse_tokens_from_item

# 0=白底，1..16 与官方一致
PALETTE = np.array([
    [255,255,255],
    [255,120,50],[255,192,203],[255,255,0],[0,150,245],[0,255,255],
    [255,127,0],[255,0,0],[255,240,150],[135,60,0],[160,32,240],
    [255,0,255],[139,137,137],[75,0,75],[150,240,80],[230,230,250],[0,175,0]
], np.uint8)

def save_png(path, idx2d):
    imageio.imwrite(path, PALETTE[np.clip(idx2d, 0, 16)])

def topdown_first_drop17(vol_hwz: np.ndarray) -> np.ndarray:
    """(H,W,Z)->(H,W)。先把 17 清零，再取第一处非零。与 decode_one 保持一致。"""
    v = vol_hwz.copy()
    v[v == 17] = 0
    H,W,Z = v.shape
    col = v.reshape(H*W, Z)
    nz  = (col != 0)
    has = nz.any(1)
    idx = nz.argmax(1)
    out = np.zeros(H*W, np.uint8)
    out[has] = col[np.arange(H*W), idx][has]
    return out.reshape(H,W)

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--out", default="quick_vis_png/lf_check")
    ap.add_argument("--idx", type=str, default="0", help="逗号分隔：'0,1,2'")
    # 与官方 token.npz 做一致性检验（可选）
    ap.add_argument("--npz_root", default="tokens_official",
                    help="若能从图片路径猜出 scene/sample，则会去 <npz_root>/<scene>/<sample>/token.npz 做一致性校验")
    # 模型
    PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    ap.add_argument("--cfg",  default=os.path.join(PROJ_ROOT, "config", "occworld.py"))
    ap.add_argument("--ckpt", default=os.path.join(PROJ_ROOT, "out", "occworld", "latest.pth"))
    # RLE 展开排列（与 find_best_perm 的 BEST 保持：F + transpose）
    ap.add_argument("--order", default="F", choices=["C","F"])
    ap.add_argument("--transpose", action="store_true", default=True)
    ap.add_argument("--no-transpose", dest="transpose", action="store_false")
    ap.add_argument("--flip_h", type=int, default=+1, choices=[+1,-1])
    ap.add_argument("--flip_w", type=int, default=+1, choices=[+1,-1])
    # Z 轴可选翻转（和 decode_one 一样：预测默认不翻）
    ap.add_argument("--flip_z", action="store_true", default=False)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    # 读 JSON / JSONL
    txt = open(args.json, "r").read().strip()
    items = json.loads(txt) if txt.startswith('[') else [json.loads(x) for x in txt.splitlines()]
    print("loaded items:", len(items))

    # 模型
    cfg = Config.fromfile(args.cfg)
    model = MODELS.build(cfg.model).eval()
    load_checkpoint(model, args.ckpt, map_location="cpu", strict=False)
    vae = getattr(getattr(model, 'module', model), 'vae')
    vq  = vae.vqvae
    C   = int(getattr(vae, "expansion", 8))
    # 自检
    emb_w = getattr(vq, "embedding", None) or getattr(getattr(vq, "quantize", None) or object(), "embedding", None)
    if emb_w is None and not hasattr(vq, "embed_code"):
        raise AttributeError("VQ module missing 'embedding'/'quantize.embedding'/'embed_code'")
    if emb_w is not None:
        print("codebook:", emb_w.weight.shape)
    print("class_embeds:", vae.class_embeds.weight.shape)

    # 解析索引
    sel = [int(x) for x in args.idx.split(',') if x.strip()]
    for k in sel:
        it = items[k]
        img_path = it["images"][0] if isinstance(it["images"], list) else it["images"]
        base = os.path.basename(img_path).replace(".jpg", "")

        # === 1) RLE → 50×50 token（严格按 F+transpose 还原） ===
        tk50 = parse_tokens_from_item(
            it, hw=(50,50),
            order=args.order, do_transpose=args.transpose,
            flip_h=args.flip_h, flip_w=args.flip_w
        ).astype(np.int64)
        print("tk50 min/max:", tk50.min(), tk50.max())
        imageio.imwrite(os.path.join(args.out, f"{base}_rawtk.png"),
                        PALETTE[(tk50 % 17).astype(np.uint8)])

        # === 2) 可选：与官方 token.npz 严格对齐校验（50×50） ===
        # 规则：从图片路径 .../vis/stitched/<scene>/<sample>.jpg 推断
        # token.npz 位于 <npz_root>/<scene>/<sample>/token.npz
        try:
            parts = img_path.split("/")
            scene = next(p for p in parts if p.startswith("scene-"))
            sample = os.path.basename(img_path).replace(".jpg","")
            npz_path = os.path.join(args.npz_root, scene, sample, "token.npz")
            if os.path.exists(npz_path):
                ids_npz = np.load(npz_path)["token"]
                assert ids_npz.ndim==3 and ids_npz.shape[0]==1
                ids_npz = ids_npz[0].astype(np.int64)
                if ids_npz.shape == (100,100):
                    ids_npz = torch.nn.functional.interpolate(
                        torch.from_numpy(ids_npz)[None,None].float(),
                        size=(50,50), mode="nearest"
                    ).long().squeeze().numpy()
                elif ids_npz.shape != (50,50):
                    raise ValueError(f"npz token shape {ids_npz.shape} not in {{50x50,100x100}}")
                match = (tk50 == np.maximum(ids_npz-1,0)).mean()  # npz 默认 1-based
                print(f"[check] token_match@50x50 vs npz = {match:.3f}")
            else:
                print(f"[warn] no npz found: {npz_path}")
        except Exception as e:
            print("[warn] npz compare failed:", e)

        # === 3) 50×50 → codebook → decoder（与 decode_one 相同写法） ===
        ids = torch.from_numpy(tk50).long()              # (50,50)
        if hasattr(vq, "embedding"):
            emb = vq.embedding(ids)                      # (50,50,Cq)
        elif hasattr(vq, "quantize") and hasattr(vq.quantize, "embedding"):
            emb = vq.quantize.embedding(ids)
        else:
            emb = vq.embed_code(ids.view(-1)).view(50,50,-1)
        code = emb.permute(2,0,1).unsqueeze(0)           # (1,Cq,50,50)
        if hasattr(vq, "post_quant_conv"):
            code = vq.post_quant_conv(code)
        dec  = vae.decoder(code, [torch.Size([200,200]), torch.Size([100,100])])  # (1, C*16, 200, 200)

        # === 4) 分类 logits（严格按 decode_one 的展平路径） ===
        B,F,H,W = dec.shape
        assert F == C*16, f"decoder out {F} != C*16({C*16})"
        feat   = dec.permute(0,2,3,1).reshape(-1, 16, C)          # (-1,16,8)
        Wcls   = vae.class_embeds.weight.T.unsqueeze(0)           # (1,8,18)
        pred16 = torch.matmul(feat, Wcls).argmax(-1).view(H,W,16) # (200,200,16)
        if args.flip_z:
            pred16 = pred16.flip(dims=[2])
        pred16 = pred16.cpu().numpy().astype(np.uint8)

        # === 5) 顶视：drop17→first-nz（与 decode_one 同步） ===
        bev = topdown_first_drop17(pred16)
        save_png(os.path.join(args.out, f"{base}_pred.png"), bev)

        # 额外信息
        u,c = np.unique(tk50, return_counts=True)
        hist = sorted(zip(u.tolist(), c.tolist()), key=lambda x:-x[1])[:6]
        print(f"[ok] {base} | tk50 hist top: {hist} | bev min/max: {bev.min()}/{bev.max()}")

if __name__ == "__main__":
    main()
