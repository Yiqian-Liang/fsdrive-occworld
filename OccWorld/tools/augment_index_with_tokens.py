# tools/augment_index_with_tokens.py
import os, sys, json, argparse, numpy as np, torch
from pathlib import Path

# --- 关键：把项目根目录及 MoVQGAN/ 加入 sys.path ---
def ensure_project_on_sys_path(project_root: str):
    root = Path(project_root).resolve()
    for p in (root, root / "MoVQGAN"):
        sp = str(p)
        if p.exists() and sp not in sys.path:
            sys.path.insert(0, sp)

from mmengine.config import Config
from mmengine.registry import MODELS
from mmengine.runner import load_checkpoint

def reduce64_to16_firstnz(sem64: np.ndarray) -> np.ndarray:
    H,W,_ = sem64.shape
    g = sem64.reshape(H, W, 16, 4)
    out = np.zeros((H, W, 16), np.int64)
    for i in range(16):
        g4 = g[:,:,i,:]
        nz = g4 != 0
        has = nz.any(-1)
        idx = nz.argmax(-1)
        pick = np.zeros((H,W), np.int64)
        pick[has] = g4[has, idx[has]]
        out[:,:,i] = pick
    return out

@torch.no_grad()
def encode_npz_to_tokens(model, npz_path: str, offset: int) -> np.ndarray:
    sem = np.load(npz_path)["semantics"].astype(np.int64)
    H,W,D = sem.shape
    if D == 64:
        sem = reduce64_to16_firstnz(sem)
    elif D != 16:
        raise ValueError(f"unexpected depth {D} in {npz_path}")

    C = int(getattr(model.vae, "expansion", 8))  # usually 8

    x = torch.from_numpy(sem[None,None]).long()
    x = model.vae.class_embeds(x).reshape(1,H,W,16,C).permute(0,3,4,1,2).reshape(1,16*C,H,W)

    z,_ = model.vae.encoder(x)
    z_e = model.vae.vqvae.quant_conv(z) if hasattr(model.vae.vqvae,"quant_conv") else z

    E = model.vae.vqvae.embedding.weight                      # (K, Cq)
    zf = z_e.permute(0,2,3,1).reshape(-1, z_e.shape[1])       # (N, Cq)
    dist = (zf**2).sum(1,keepdims=True) - 2* zf @ E.t() + (E**2).sum(1)[None,:]
    idx  = dist.argmin(1).reshape(1, z_e.shape[2], z_e.shape[3])  # (1,50,50)

    token = (idx + offset).cpu().numpy().astype(np.uint16)    # (1,50,50)
    return token

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--out",   required=True)
    ap.add_argument("--cfg",   default="config/occworld.py")
    ap.add_argument("--ckpt",  default="out/occworld/latest.pth")
    ap.add_argument("--tokens_root", default="tokens_official")
    ap.add_argument("--offset", type=int, default=1)
    ap.add_argument("--root", default=".", help="项目根目录（包含 model/）")
    args = ap.parse_args()

    # 把项目根加进 sys.path，并注册 MOVQ 到 mmengine
    ensure_project_on_sys_path(args.root)
    from movqgan.models.vqgan import MOVQ  # type: ignore
    if MODELS.get('MOVQ') is None:
        MODELS.register_module(MOVQ)

    root = Path(args.root)
    out_dir = Path(args.tokens_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    # build model once
    cfg = Config.fromfile(args.cfg)
    model = MODELS.build(cfg.model).eval()
    load_checkpoint(model, args.ckpt, map_location="cpu", strict=False)

    total = 0
    written = 0
    with open(args.index, "r") as fin, open(args.out, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line: continue
            item = json.loads(line)

            scene = item["scene"]
            samp  = item["sample_token"]

            # 目标 token 相对路径
            tok_path = item.get("occ_token_npz") or str(out_dir / scene / samp / "token.npz")
            abs_tok  = (root / tok_path).resolve()
            abs_tok.parent.mkdir(parents=True, exist_ok=True)

            if not abs_tok.exists():
                gt_npz = item["occ_gt_npz"]
                abs_gt = (root / gt_npz).resolve()
                token  = encode_npz_to_tokens(model, str(abs_gt), offset=args.offset)
                np.savez_compressed(str(abs_tok), token=token)
                written += 1

            item["occ_token_npz"] = tok_path
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")
            total += 1

    print(f"[DONE] total={total}, newly_written={written}, out_file={args.out}")

if __name__ == "__main__":
    main()
