import os
import re
import json
import ndjson
import pickle
import tiktoken
from tqdm import tqdm
from nuscenes.nuscenes import NuScenes
from datetime import datetime, timedelta


def vq_ids_to_string(ids, prefix="<VQ_", suffix=">"):
    return " ".join(f"{prefix}{int(i)}{suffix}" for i in ids)

def parse_ids_any(x):

    if isinstance(x, (list, tuple)):
        return [int(i) for i in x]
    if isinstance(x, str):
        s = x.strip()
        if s.startswith("[") and s.endswith("]"):
            s = s[1:-1]
        parts = [p for p in s.split(",") if p.strip() != ""]
        return [int(p.strip()) for p in parts]
    raise ValueError(f"Unsupported token list type: {type(x)}")

system = (
    "You're an autonomous vehicle's brain. "
    "Coordinates: X-axis is perpendicular, and Y-axis is parallel to the direction you're facing. "
    "You're at point (0,0). Units: meters. Based on the provided particulars, "
    "you can generate CAM_FRONT image at the 0.5 second in the future.\n"
)

encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
num_language_tokens = 0
num_system_tokens = 0
num_user_tokens = 0
num_assistant_tokens = 0
traj_only = True
train_messages = []


gt_indices = json.load(open('./MoVQGAN/gt_indices_pretrain.json', 'r'))


dataroot = './LLaMA-Factory/data/nuscenes'
nusc = NuScenes(version='v1.0-mini', dataroot=dataroot, verbose=True)

cam_front_dir = os.path.join(dataroot, 'sweeps/CAM_FRONT')
supported_extensions = ('.png', '.jpg', '.jpeg', '.JPG', '.JPEG', '.PNG')
image_files = [
    f for f in os.listdir(cam_front_dir)
    if os.path.isfile(os.path.join(cam_front_dir, f)) and f.endswith(supported_extensions)
]
image_files = sorted(image_files)

for i, now_path in enumerate(tqdm(image_files)):
    images_path = []
    try:
        target_name = None
        filename = now_path
        parts = filename.split("__")

        timestamp_str = filename.split("__")[2].split(".")[0]
        original_timestamp = int(timestamp_str) / 1e6
        dt = datetime.fromtimestamp(original_timestamp)
        future_dt = dt + timedelta(seconds=1.0)
        new_timestamp = int(future_dt.timestamp() * 1e6)
        new_filename = filename.replace(timestamp_str, str(new_timestamp))
        new_filepath = os.path.join(cam_front_dir, new_filename)

        if os.path.exists(new_filepath):
            target_name = new_filename
        else:
            prefix = str(new_timestamp)[:11]
            for j in range(i + 1, min(i + 10, len(image_files))):
                if image_files[j].startswith(f"{parts[0]}__{parts[1]}__{prefix}"):
                    target_name = image_files[j]
                    break

            if target_name is None:
                prefix_new = str(int(prefix) - 1)
                for j in range(i + 1, min(i + 10, len(image_files))):
                    if image_files[j].startswith(f"{parts[0]}__{parts[1]}__{prefix_new}"):
                        target_name = image_files[j]
                        break

            if target_name is None:
                prefix_new = str(int(prefix) + 1)
                for j in range(i + 1, min(i + 10, len(image_files))):
                    if image_files[j].startswith(f"{parts[0]}__{parts[1]}__{prefix_new}"):
                        target_name = image_files[j]
                        break

        if target_name is None:
            continue

        raw_ids = gt_indices[target_name]
        ids = parse_ids_any(raw_ids)
        next_img_token = vq_ids_to_string(ids)

    except Exception as e:
        continue

    images_path.append(os.path.join('data/nuscenes/sweeps/CAM_FRONT', filename))

    train_message = {
        "id": filename,
        "images": images_path,
        "system": system,
        "conversations": [
            {
                "from": "user",
                "value": "This is the CAM_FRONT image of the current frame: <image>\n"
                         "Please generate CAM_FRONT image at the 1.0 second in the future.\n"
            },
            {
                "from": "assistant",
                "value": next_img_token 
            }
        ]
    }
    train_messages.append(train_message)


os.makedirs("./LLaMA-Factory/data", exist_ok=True)
with open("./LLaMA-Factory/data/pretrain_data.json", "w") as f:
    json.dump(train_messages, f, indent=2, ensure_ascii=False)
