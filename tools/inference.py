
from transformers import AutoTokenizer, AutoProcessor, Qwen2_5_VLForConditionalGeneration  # Use Qwen2_5_VL class for Qwen2.5-VL
from PIL import Image
from qwen_vl_utils import process_vision_info
import torch
import re
import json

# Load from the same model_dir to ensure consistency
model_dir = "/nas/vlm_driving/fsdrive-occworld/OccWorld/exp/debug_mm_mini"
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(model_dir, use_fast=False, trust_remote_code=True)  # Slow for compatibility

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_dir, device_map="cuda:0", ignore_mismatched_sizes=True, trust_remote_code=True
)

# Test inference on train image to check overfit
image_path = "/nas/vlm_driving/OccWorld/vis/stitched/scene-0002/25fbd2d7377449aabfd2173c3a0a418f.jpg"
try:
    image = Image.open(image_path).convert("RGB")
    if image is None or image.size == (0, 0):
        raise ValueError("Failed to load or invalid image at {}".format(image_path))
except Exception as e:
    print(f"Image loading error: {e}")
    exit(1)

# Use messages format for Qwen2.5-VL
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "Predict occupancy tokens (size 50x50).\nReturn RLE pairs [[id,count],...], 0-based."},
        ],
    }
]

# Apply chat template and process vision info
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info(messages)

# Prepare inputs
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt"
).to("cuda:0")

# Debug: Print inputs shapes and input_ids
print("Processor inputs:", {k: v.shape if isinstance(v, torch.Tensor) else v for k, v in inputs.items()})
print("Input IDs:", inputs["input_ids"])

# Generate output
try:
    generated_ids = model.generate(**inputs, max_new_tokens=512)
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False)[0]
    print("Raw model output:", output_text)
except Exception as e:
    print(f"Generation error: {e}")
    exit(1)

# Post-process to convert <OCC_n> sequence to RLE pairs [[id, count], ...]
try:
    # Extract numbers from <OCC_n> tokens using regex
    occ_matches = re.findall(r'<OCC_(\d+)>', output_text)
    if not occ_matches:
        raise ValueError("No <OCC_n> tokens found in output.")
    
    ids = [int(n) for n in occ_matches]
    
    # Compute RLE: group consecutive same ids and count them
    rle_pairs = []
    if ids:
        current_id = ids[0]
        count = 1
        for next_id in ids[1:]:
            if next_id == current_id:
                count += 1
            else:
                rle_pairs.append([current_id, count])
                current_id = next_id
                count = 1
        rle_pairs.append([current_id, count])  # Add the last group
    
    # Print and save as JSON for verification
    print("Parsed RLE pairs:", rle_pairs)
    with open("generated_rle.json", "w") as f:
        json.dump({"image_path": image_path, "rle_pairs": rle_pairs}, f, indent=2)
    print("Saved RLE pairs to generated_rle.json")
except Exception as e:
    print(f"Post-processing error: {e}")






# from transformers import AutoTokenizer, AutoProcessor, Qwen2_5_VLForConditionalGeneration  # Use Qwen2_5_VL class for Qwen2.5-VL
# from PIL import Image
# from qwen_vl_utils import process_vision_info
# import torch

# # Load from the same model_dir to ensure consistency
# model_dir = "/nas/vlm_driving/fsdrive-occworld/OccWorld/exp/debug_mm_mini"
# tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
# processor = AutoProcessor.from_pretrained(model_dir, use_fast=False, trust_remote_code=True)  # Slow for compatibility

# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     model_dir, device_map="cuda:0", ignore_mismatched_sizes=True, trust_remote_code=True
# )

# # Test inference on train image to check overfit
# image_path = "/nas/vlm_driving/OccWorld/vis/stitched/scene-0002/25fbd2d7377449aabfd2173c3a0a418f.jpg"
# try:
#     image = Image.open(image_path).convert("RGB")
#     if image is None or image.size == (0, 0):
#         raise ValueError("Failed to load or invalid image at {}".format(image_path))
# except Exception as e:
#     print(f"Image loading error: {e}")
#     exit(1)

# # Use messages format for Qwen2.5-VL
# messages = [
#     {
#         "role": "user",
#         "content": [
#             {"type": "image", "image": image},
#             {"type": "text", "text": "Predict occupancy tokens (size 50x50).\nReturn RLE pairs [[id,count],...], 0-based."},
#         ],
#     }
# ]

# # Apply chat template
# text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# # Process vision info
# image_inputs, video_inputs = process_vision_info(messages)

# # Prepare inputs
# inputs = processor(
#     text=[text],
#     images=image_inputs,
#     videos=video_inputs,
#     padding=True,
#     return_tensors="pt"
# ).to("cuda:0")

# # Debug: Print inputs shapes and input_ids
# print("Processor inputs:", {k: v.shape if isinstance(v, torch.Tensor) else v for k, v in inputs.items()})
# print("Input IDs:", inputs["input_ids"])

# # Generate output
# try:
#     generated_ids = model.generate(**inputs, max_new_tokens=512)
#     # Trim to only new tokens
#     generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
#     output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
#     print("Model output:", output_text[0])
# except Exception as e:
#     print(f"Generation error: {e}")
    
    
    
    

python - <<'PY'
import json
import re
import os

def parse_tokens_to_rle(tokens_str):
    occ_numbers = re.findall(r'<OCC_(\d+)>', tokens_str)
    if not occ_numbers:
        return []

    ids = [int(num) for num in occ_numbers]
    
    rle_pairs = []
    if ids:
        current_id = ids[0]
        count = 1
        for next_id in ids[1:]:
            if next_id == current_id:
                count += 1
            else:
                rle_pairs.append([current_id, count])
                current_id = next_id
                count = 1
        rle_pairs.append([current_id, count])
    
    return rle_pairs

jsonl_path = "/nas/vlm_driving/fsdrive-occworld/OccWorld/data/lf_train_first5.tokens.jsonl"
generated_jsonl_path = "generated_rle.jsonl"
generated_samples = []

with open(jsonl_path, "r", encoding="utf-8") as f:
    for line in f:
        sample = json.loads(line.strip())
        gpt_value = sample['conversations'][1]['value']
        image_path = sample['images'][0] if 'images' in sample else "unknown"
        
        rle = parse_tokens_to_rle(gpt_value)
        total_cells = sum(count for _, count in rle)
        print(f"Processed {image_path}: Total cells = {total_cells}")
        
        generated_samples.append({
            "image_path": image_path,
            "rle_pairs": rle
        })

# Save to generated_rle.jsonl
with open(generated_jsonl_path, "w", encoding="utf-8") as f:
    for sample in generated_samples:
        f.write(json.dumps(sample) + "\n")

print(f"Generated JSONL saved to {generated_jsonl_path}")
PY