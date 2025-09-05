# tools/convert_rle_to_tokens_jsonl.py
import json
import argparse
from pathlib import Path

def rle_to_tokens(rle_pairs):
    """
    将 RLE pairs [[id, count], ...] 展开成 <OCC_id> 序列字符串。
    如 [[449,1], [64,4]] -> " <OCC_449> <OCC_64> <OCC_64> <OCC_64> <OCC_64>"
    注意：不包括 <OCC_START> 和 <OCC_END>，这些在最终字符串中添加。
    """
    tokens = []
    for id_val, count in rle_pairs:
        tokens.extend([f"<OCC_{id_val}>" for _ in range(count)])
    return " ".join(tokens)

def main():
    ap = argparse.ArgumentParser(description="Convert RLE jsonl to tokens jsonl")
    ap.add_argument("--input", required=True, help="Input RLE jsonl file (e.g., lf_train_first5.jsonl)")
    ap.add_argument("--output", required=True, help="Output tokens jsonl file (e.g., lf_train_first5.tokens.jsonl)")
    ap.add_argument("--ref_images", default=None, help="Optional reference jsonl for correcting image paths (if paths in input are wrong)")
    args = ap.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    # 如果提供了参考文件，加载图像路径映射（假设 sample_token 或其他唯一键对应）
    image_map = {}
    if args.ref_images:
        ref_path = Path(args.ref_images)
        with open(ref_path, "r") as fref:
            for line in fref:
                item = json.loads(line.strip())
                # 假设使用 "images" 字段的第一个图像路径，并用 conversations[0]["value"] 的某种唯一标识（如 sample_token，如果有）
                # 但在提供的示例中，没有 sample_token，这里假设使用 images[0] 作为键，或简单顺序对应
                # 注意：提供的示例只有一个项，所以假设顺序对应。如果有多个，需要调整逻辑
                conversations = item.get("conversations", [])
                images = item.get("images", [])
                if images:
                    # 这里简单用索引或路径作为键，但由于用户说路径错了，假设 ref_images 是正确的路径源
                    # 为简单起见，假设输入和参考是逐行对应的
                    image_map[len(image_map)] = images  # 用行号作为键

    with open(input_path, "r") as fin, open(output_path, "w") as fout:
        for idx, line in enumerate(fin):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)

            # 提取 conversations
            conversations = item.get("conversations", [])
            if len(conversations) != 2 or conversations[0]["from"] != "human" or conversations[1]["from"] != "gpt":
                print(f"Skipping invalid item at line {idx+1}")
                continue

            # human 部分保持不变
            human_value = conversations[0]["value"]

            # gpt value 是 RLE pairs 的字符串，需要解析并转换
            gpt_value_str = conversations[1]["value"]
            try:
                rle_pairs = json.loads(gpt_value_str)  # 假设是字符串表示的 [[id,count],...]
            except json.JSONDecodeError:
                print(f"Invalid RLE in item at line {idx+1}: {gpt_value_str}")
                continue

            # 转换为 tokens 序列
            tokens_seq = rle_to_tokens(rle_pairs)
            new_gpt_value = f"<OCC_START>{tokens_seq} <OCC_END>"

            # 更新 conversations
            new_conversations = [
                {"from": "human", "value": human_value},
                {"from": "gpt", "value": new_gpt_value}
            ]
            item["conversations"] = new_conversations

            # 修正 images 路径（如果提供了参考）
            if args.ref_images and idx in image_map:
                item["images"] = image_map[idx]
            elif "images" in item:
                # 如果没有参考，但用户说路径错了，这里可以添加逻辑修正，但由于未知正确路径，保持原样或警告
                print(f"Warning: Image path in item {idx+1} may be incorrect: {item['images']}")

            # 写出
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"[DONE] Converted {input_path} to {output_path}")

if __name__ == "__main__":
    main()