#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single-file dual-mode training:
  --task token  : image -> 50x50x512 tokens (CE), eval token-acc and (optional) BEV mIoU via VQ-VAE
  --task bev16  : image -> 200x200x17 (0=ignore, 1..16 semantic), direct CE, eval mIoU

Why two modes?
  token: 为 LLaMA-Factory 生成 RLE 链接数据；可评估 token-acc / 合法率 / decode 后 mIoU
  bev16: 直接优化下游语义，mIoU 更容易起量，是实用 baseline

Usage examples:
  # A) token mode（复现你原来那条线，但支持权重/Eval BEV）
  python train_dual.py --task token \
    --train_json /nas/.../lf_train.json --val_json /nas/.../lf_val.json \
    --out runs/token_r18 \
    --bs 16 --epoch 20 --lr 1e-3 --offset 1 \
    --cfg config/occworld.py --ckpt out/occworld/latest.pth

  # B) bev16 直接监督（推荐并行跑）
  python train_dual.py --task bev16 \
    --train_json /nas/.../lf_train.json --val_json /nas/.../lf_val.json \
    --out runs/bev16_r18 \
    --bs 16 --epoch 20 --lr 1e-3 --offset 1 \
    --cfg config/occworld.py --ckpt out/occworld/latest.pth
"""

import os, json, argparse, random
from typing import List, Tuple
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms as T

# ---------- OccWorld decode deps ----------
from mmengine.config import Config
from mmengine.registry import MODELS
from mmengine.runner import load_checkpoint
from tools.lf_utils import parse_tokens_from_item

# ---------- small utils ----------
def set_seed(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def _load_items(json_path: str) -> List[dict]:
    txt = open(json_path, "r").read().strip()
    return json.loads(txt) if txt.startswith('[') else [json.loads(x) for x in txt.splitlines()]

# ---------- VQ-VAE decode chain (same as your check script, drop17->first) ----------
@torch.no_grad()
def decode_tokens_to_bev(ids50: np.ndarray, vae, vq, flip_z: bool=False) -> np.ndarray:
    dev = vae.class_embeds.weight.device
    ids = torch.from_numpy(ids50).long().to(dev)           # [50,50]

    if hasattr(vq, "embedding"):
        emb = vq.embedding(ids)                            # [50,50,e_dim]
    elif hasattr(vq, "quantize") and hasattr(vq.quantize, "embedding"):
        emb = vq.quantize.embedding(ids)
    else:
        emb = vq.embed_code(ids.view(-1)).view(50,50,-1)

    code = emb.permute(2,0,1).unsqueeze(0)                 # [1,e_dim,50,50]
    if hasattr(vq, "post_quant_conv"):
        code = vq.post_quant_conv(code)

    dec  = vae.decoder(code, [torch.Size([200,200]), torch.Size([100,100])])  # [1,C*16,200,200]
    C    = int(getattr(vae, "expansion", 8))
    B,F,H,W = dec.shape
    assert F == C*16, f"decoder out {F} != C*16({C*16})"

    feat   = dec.permute(0,2,3,1).reshape(-1, 16, C)       # (-1,16,C)
    Wcls   = vae.class_embeds.weight.T.unsqueeze(0)        # (1,C,18)
    pred16 = torch.matmul(feat, Wcls).argmax(-1).view(H,W,16)
    if flip_z:
        pred16 = pred16.flip(dims=[2])
    pred16 = pred16.detach().cpu().numpy().astype(np.uint8)

    v = pred16.copy()
    v[v == 17] = 0
    col = v.reshape(H*W, 16)
    nz  = col != 0
    has = nz.any(1)
    idx = nz.argmax(1)
    bev = np.zeros(H*W, np.uint8)
    bev[has] = col[np.arange(H*W), idx][has]
    return bev.reshape(H, W)                                # uint8 in [0..16]

def fast_confusion_matrix(pred: np.ndarray, gt: np.ndarray, num_classes=16) -> np.ndarray:
    mask = (gt >= 1) & (gt <= num_classes)
    pred = pred[mask]; gt = gt[mask]
    idx = gt * (num_classes + 1) + pred
    cm  = np.bincount(idx, minlength=(num_classes+1)**2).reshape(num_classes+1, num_classes+1)
    return cm[1:,1:]

def miou_from_cm(cm: np.ndarray) -> Tuple[np.ndarray, float]:
    iou = np.zeros(cm.shape[0], dtype=np.float64)
    for k in range(cm.shape[0]):
        denom = cm[k,:].sum() + cm[:,k].sum() - cm[k,k]
        iou[k] = (cm[k,k] / denom) if denom > 0 else np.nan
    miou = np.nanmean(iou)
    return iou, miou

def build_vae(cfg_path, ckpt_path, device):
    cfg = Config.fromfile(cfg_path)
    model = MODELS.build(cfg.model).eval()
    load_checkpoint(model, ckpt_path, map_location="cpu", strict=False)
    vae = getattr(getattr(model, 'module', model), 'vae').to(device)
    vq  = vae.vqvae
    return vae, vq

# ---------- datasets ----------
class CommonImageTF:
    """等比缩放 + 中心裁剪到矩形，默认仅颜色增强（不做几何）"""
    def __init__(self, img_short=320, img_ratio=2.0, aug=False):
        Hout = img_short
        Wout = int(round(img_short*img_ratio))
        self.size = (Hout, Wout)
        ops = [
            T.Resize(img_short),              # keep aspect, short side = img_short
            T.CenterCrop((Hout, Wout)),      # crop to HxW rectangle
        ]
        if aug:
            ops += [T.ColorJitter(0.2,0.2,0.2,0.1)]
        ops += [T.ToTensor(), T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]
        self.tf = T.Compose(ops)
    def __call__(self, img): return self.tf(img)

class TokenDataset(Dataset):
    def __init__(self, json_path, order="F", transpose=True, flip_h=+1, flip_w=+1,
                 offset=0, tf:CommonImageTF=None):
        self.items = _load_items(json_path)
        self.order = order; self.transpose = transpose
        self.flip_h = flip_h; self.flip_w = flip_w
        self.offset = offset
        self.tf = tf or CommonImageTF()
    def __len__(self): return len(self.items)
    def _img(self, it): return it["images"][0] if isinstance(it["images"], list) else it["images"]
    def __getitem__(self, idx):
        it = self.items[idx]
        img = Image.open(self._img(it)).convert("RGB")
        x = self.tf(img)
        tk50 = parse_tokens_from_item(
            it, hw=(50,50),
            order=self.order, do_transpose=self.transpose,
            flip_h=self.flip_h, flip_w=self.flip_w
        ).astype(np.int64)
        if self.offset: tk50 = np.clip(tk50 - self.offset, 0, 511)
        y = torch.from_numpy(tk50).long()     # [50,50]
        return x, y

class BEVDataset(Dataset):
    def __init__(self, json_path, vae, vq, cache_bev=True, img_short=320, img_ratio=2.0,
                 order="F", transpose=True, flip_h=+1, flip_w=+1, offset=0, aug=False):
        self.items = _load_items(json_path)
        self.vae = vae; self.vq = vq
        self.cache_bev = cache_bev
        self.cache = {}
        self.order=order; self.transpose=transpose; self.flip_h=flip_h; self.flip_w=flip_w
        self.offset=offset
        self.tf = CommonImageTF(img_short, img_ratio, aug)
    def __len__(self): return len(self.items)
    def _img(self, it): return it["images"][0] if isinstance(it["images"], list) else it["images"]
    def _tk(self, it):
        tk50 = parse_tokens_from_item(
            it, hw=(50,50),
            order=self.order, do_transpose=self.transpose,
            flip_h=self.flip_h, flip_w=self.flip_w
        ).astype(np.int64)
        if self.offset: tk50 = np.clip(tk50 - self.offset, 0, 511)
        return tk50
    def _bev(self, idx, it):
        if self.cache_bev and idx in self.cache: return self.cache[idx]
        bev = decode_tokens_to_bev(self._tk(it), self.vae, self.vq)  # [200,200], 0..16
        if self.cache_bev: self.cache[idx] = bev
        return bev
    def __getitem__(self, idx):
        it = self.items[idx]
        img = Image.open(self._img(it)).convert("RGB")
        x = self.tf(img)
        y = torch.from_numpy(self._bev(idx, it)).long()   # [200,200], 0..16
        return x, y

# ---------- class weights ----------
def token_class_weight(items: List[dict], order="F", transpose=True, flip_h=+1, flip_w=+1,
                       offset=0, pow_k=0.5, num_classes=512) -> torch.Tensor:
    """不加载图像，直接数 token 频率 -> 1/freq^k（均值归一）"""
    freq = np.zeros(num_classes, dtype=np.int64)
    for it in tqdm(items, desc="[count token freq]"):
        tk50 = parse_tokens_from_item(
            it, hw=(50,50), order=order, do_transpose=transpose, flip_h=flip_h, flip_w=flip_w
        ).astype(np.int64)
        if offset: tk50 = np.clip(tk50 - offset, 0, 511)
        binc = np.bincount(tk50.reshape(-1), minlength=num_classes)
        freq += binc
    freq = freq + 1
    w = (1.0 / (freq.astype(np.float64) ** pow_k))
    w = (w / w.mean()).astype(np.float32)
    return torch.from_numpy(w)

def bev_class_weight(dataset: BEVDataset, num_classes=17, pow_k=0.5) -> torch.Tensor:
    freq = np.zeros(num_classes, dtype=np.int64)
    for idx in tqdm(range(len(dataset)), desc="[count bev freq]"):
        _, y = dataset[idx]
        binc = np.bincount(y.view(-1).numpy(), minlength=num_classes)
        freq += binc
    freq = freq + 1
    w = (1.0 / (freq.astype(np.float64) ** pow_k)).astype(np.float32)
    w[0] = 0.0  # ignore
    w = w / w.mean()
    return torch.from_numpy(w)

# ---------- models ----------
class TokenHeadR18(nn.Module):
    def __init__(self, num_classes=512):
        super().__init__()
        backbone = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        self.stem = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4
        )
        self.head = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, num_classes, 1)
        )
    def forward(self, x):
        f = self.stem(x)                                        # stride 32
        logits = F.interpolate(self.head(f), size=(50,50), mode="bilinear", align_corners=False)
        return logits                                           # [B,512,50,50]

class BEVHeadR18(nn.Module):
    def __init__(self, num_classes=17):
        super().__init__()
        backbone = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        self.stem = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4
        )
        self.up = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),  # 8x
            nn.Conv2d(256, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),  # 16x
            nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),  # 32x
            nn.Conv2d(64, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.Conv2d(32, num_classes, 1)
        )
    def forward(self, x):
        f = self.stem(x)
        logits = F.interpolate(self.up(f), size=(200,200), mode="bilinear", align_corners=False)
        return logits                                           # [B,17,200,200]

# ---------- eval ----------
@torch.no_grad()
def validate_token(model, loader, device, criterion, eval_bev, vae, vq, flip_z=False):
    model.eval()
    tot_loss=0.0; corr=0; denom=0
    cm = np.zeros((16,16), dtype=np.int64) if eval_bev else None
    for x,y in tqdm(loader, desc="[val-token]", leave=False):
        x=x.to(device, non_blocking=True); y=y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        tot_loss += loss.item()*x.size(0)
        pred = logits.argmax(1)
        corr += (pred == y).sum().item()
        denom += y.numel()
        if eval_bev:
            pnp = pred.detach().cpu().numpy(); gnp = y.detach().cpu().numpy()
            for i in range(x.size(0)):
                bev_p = decode_tokens_to_bev(pnp[i], vae, vq, flip_z)
                bev_g = decode_tokens_to_bev(gnp[i], vae, vq, flip_z)
                cm += fast_confusion_matrix(bev_p, bev_g, num_classes=16)
    token_acc = corr / max(denom,1)
    avg_loss  = tot_loss / len(loader.dataset)
    if eval_bev:
        iou, miou = miou_from_cm(cm)
        return avg_loss, token_acc, miou, iou
    return avg_loss, token_acc, None, None

@torch.no_grad()
def validate_bev(model, loader, device, criterion):
    model.eval()
    tot_loss=0.0; corr=0; denom=0
    cm = np.zeros((16,16), dtype=np.int64)
    for x,y in tqdm(loader, desc="[val-bev16]", leave=False):
        x=x.to(device, non_blocking=True); y=y.to(device, non_blocking=True) # [B,200,200], 0..16
        logits = model(x)                                # [B,17,200,200]
        loss = criterion(logits, y)
        tot_loss += loss.item()*x.size(0)
        pred = logits.argmax(1)
        mask = (y!=0)
        corr += (pred[mask]==y[mask]).sum().item()
        denom+= mask.sum().item()
        p = pred.detach().cpu().numpy(); g = y.detach().cpu().numpy()
        for i in range(p.shape[0]):
            cm += fast_confusion_matrix(p[i], g[i], num_classes=16)
    pix_acc = corr / max(denom,1)
    iou, miou = miou_from_cm(cm)
    avg_loss  = tot_loss / len(loader.dataset)
    return avg_loss, pix_acc, miou, iou

# ---------- main train ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["token","bev16"], required=True)
    ap.add_argument("--train_json", required=True)
    ap.add_argument("--val_json",   required=True)
    ap.add_argument("--out",        required=True)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--epoch", type=int, default=20)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--img_short", type=int, default=320)
    ap.add_argument("--img_ratio", type=float, default=2.0)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--noamp", action="store_true")
    ap.add_argument("--noweight", action="store_true")
    ap.add_argument("--w_pow", type=float, default=0.5, help="reweight exponent k in 1/freq^k")
    ap.add_argument("--eval_bev", action="store_true", help="token-task: also eval BEV mIoU via VQ-VAE")
    # VAE ckpt for decode/eval
    ap.add_argument("--cfg",  default="config/occworld.py")
    ap.add_argument("--ckpt", default="out/occworld/latest.pth")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # build datasets / loaders
    if args.task == "token":
        tr_tf = CommonImageTF(args.img_short, args.img_ratio, aug=True)
        va_tf = CommonImageTF(args.img_short, args.img_ratio, aug=False)
        train_ds = TokenDataset(args.train_json, offset=args.offset, tf=tr_tf)
        val_ds   = TokenDataset(args.val_json,   offset=args.offset, tf=va_tf)
        train_ld = DataLoader(train_ds, batch_size=args.bs, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=True)
        val_ld   = DataLoader(val_ds, batch_size=args.bs, shuffle=False,
                              num_workers=args.workers, pin_memory=True)

        # class weight (512)
        if args.noweight:
            w = None
        else:
            w = token_class_weight(train_ds.items, offset=args.offset, pow_k=args.w_pow).to(device)

        model = TokenHeadR18(num_classes=512).to(device)
        opt   = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epoch*len(train_ld))
        scaler= torch.cuda.amp.GradScaler(enabled=not args.noamp)

        def ce_token(logits, target):
            return F.cross_entropy(logits, target, weight=w)

        # build VAE for optional BEV eval
        vae = vq = None
        if args.eval_bev:
            vae, vq = build_vae(args.cfg, args.ckpt, device)

        best_key = -1.0
        for ep in range(1, args.epoch+1):
            model.train()
            pbar = tqdm(train_ld, desc=f"[train token ep{ep}]")
            running = 0.0
            for i,(x,y) in enumerate(pbar):
                x=x.to(device, non_blocking=True); y=y.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=not args.noamp):
                    logits = model(x)
                    loss = ce_token(logits, y)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                sched.step()
                running += loss.item()
                if (i+1)%50==0:
                    pbar.set_postfix(loss=f"{running/50:.4f}"); running=0.0

            v = validate_token(model, val_ld, device, ce_token, args.eval_bev, vae, vq)
            if args.eval_bev:
                v_loss, v_acc, v_miou, _ = v
                print(f"[ep{ep}] val loss={v_loss:.4f}  token-acc={v_acc:.4f}  BEV mIoU={v_miou:.4f}")
                key = v_miou
            else:
                v_loss, v_acc, _, _ = v
                print(f"[ep{ep}] val loss={v_loss:.4f}  token-acc={v_acc:.4f}")
                key = v_acc

            torch.save({"model": model.state_dict(), "epoch": ep}, os.path.join(args.out, "last.pth"))
            if key > best_key:
                best_key = key
                torch.save({"model": model.state_dict(), "epoch": ep}, os.path.join(args.out, "best.pth"))
                print(f"[*] best {'mIoU' if args.eval_bev else 'acc'} updated: {best_key:.4f}")

    else:  # bev16
        vae, vq = build_vae(args.cfg, args.ckpt, device)
        train_ds = BEVDataset(args.train_json, vae, vq, cache_bev=True,
                              img_short=args.img_short, img_ratio=args.img_ratio,
                              offset=args.offset, aug=True)
        val_ds   = BEVDataset(args.val_json, vae, vq, cache_bev=True,
                              img_short=args.img_short, img_ratio=args.img_ratio,
                              offset=args.offset, aug=False)
        train_ld = DataLoader(train_ds, batch_size=args.bs, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=True)
        val_ld   = DataLoader(val_ds, batch_size=args.bs, shuffle=False,
                              num_workers=args.workers, pin_memory=True)

        if args.noweight:
            w = None
        else:
            w = bev_class_weight(train_ds, num_classes=17, pow_k=args.w_pow).to(device)

        model = BEVHeadR18(num_classes=17).to(device)
        opt   = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epoch*len(train_ld))
        scaler= torch.cuda.amp.GradScaler(enabled=not args.noamp)

        def ce_bev(logits, target):
            return F.cross_entropy(logits, target, weight=w, ignore_index=0)

        best_miou = -1.0
        for ep in range(1, args.epoch+1):
            model.train()
            pbar = tqdm(train_ld, desc=f"[train bev16 ep{ep}]")
            running=0.0
            for i,(x,y) in enumerate(pbar):
                x=x.to(device, non_blocking=True); y=y.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=not args.noamp):
                    logits = model(x)    # [B,17,200,200]
                    loss = ce_bev(logits, y)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                sched.step()
                running += loss.item()
                if (i+1)%50==0:
                    pbar.set_postfix(loss=f"{running/50:.4f}"); running=0.0

            v_loss, pix_acc, miou, _ = validate_bev(model, val_ld, device, ce_bev)
            print(f"[ep{ep}] val loss={v_loss:.4f}  pix-acc(≠0)={pix_acc:.4f}  BEV mIoU={miou:.4f}")
            torch.save({"model": model.state_dict(), "epoch": ep}, os.path.join(args.out, "last.pth"))
            if miou > best_miou:
                best_miou = miou
                torch.save({"model": model.state_dict(), "epoch": ep}, os.path.join(args.out, "best.pth"))
                print(f"[*] best mIoU updated: {best_miou:.4f}")

if __name__ == "__main__":
    main()
