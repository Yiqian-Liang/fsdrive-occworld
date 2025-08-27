import os, json, argparse, re
from transformers import AutoTokenizer

def build_vq_tokens(n_embed: int, prefix="<VQ_", suffix=">"):
    return [f"{prefix}{i}{suffix}" for i in range(n_embed)]

def parse_n_embed(path: str, fallback=8192):
    try:
        txt = open(path, "r", encoding="utf-8").read()
        m = re.search(r"(n_embed|codebook_size)\s*:\s*(\d+)", txt)
        if m:
            return int(m.group(2))
    except Exception:
        pass
    return fallback

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True, help="/home/users/nus/e0846828/scratch/fsdrive/models/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--movq_config", default="MoVQGAN/configs/movqgan_270M.yaml")
    ap.add_argument("--out_tokenizer_dir", required=True)
    ap.add_argument("--n_embed", type=int, default=None)
    args = ap.parse_args()

    n_embed = args.n_embed or parse_n_embed(args.movq_config, fallback=8192)
    new_tokens = build_vq_tokens(n_embed)


    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True, use_fast=True)


    added = tok.add_special_tokens({"additional_special_tokens": new_tokens})

    os.makedirs(args.out_tokenizer_dir, exist_ok=True)
    tok.save_pretrained(args.out_tokenizer_dir)
    with open(os.path.join(args.out_tokenizer_dir, "movq_added_tokens.json"), "w") as f:
        json.dump({"n_embed": n_embed, "added": added,
                   "example_tokens": new_tokens[:8]}, f, ensure_ascii=False, indent=2)

    print(f"[OK] add {added} MoVQ tokens -> saved tokenizer to {args.out_tokenizer_dir}")

if __name__ == "__main__":
    main()
