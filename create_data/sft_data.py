import re
import json
import pickle
import argparse
import tiktoken
from nuscenes.nuscenes import NuScenes
from prompt_message import generate_user_message, generate_assistant_message


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
    "You're an autonomous vehicle's brain. Coordinates: X-axis is perpendicular, and Y-axis is parallel to the direction you're facing. "
    "You're at point (0,0). Units: meters. Based on the provided particulars, please output the CAM_FRONT image at the 1.0 second in the future "
    "and plan waypoints (0.5s intervals) for the next 3 seconds."
)

parser = argparse.ArgumentParser(description="Choose to use train or val tokens.")
parser.add_argument("--split", type=str, default="train", choices=["train", "val"], help="Select 'train' or 'val' token set")
args = parser.parse_args()

data = pickle.load(open('./create_data/cached_nuscenes_info.pkl', 'rb'))
split = json.load(open('./create_data/full_split.json', 'r'))
tokens = split[args.split]

num_train_samples = len(tokens)
train_ratio = 1.0

encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
num_language_tokens = 0
num_user_tokens = 0
num_assistant_tokens = 0
traj_only = True

dataroot = './LLaMA-Factory/data/nuscenes'
# nusc = NuScenes(version='v1.0-trainval', dataroot=dataroot, verbose=True)
nusc = NuScenes(version='v1.0-mini', dataroot=dataroot, verbose=True)


sft_indices = json.load(open('./MoVQGAN/gt_indices_sft.json', 'r'))

train_messages = []

for token_i, token in enumerate(tokens):
    if token_i >= int(train_ratio * num_train_samples):
        break


    assitant_message = generate_assistant_message(data, token, traj_only=traj_only)
    user_message, images_path = generate_user_message(data, token)

    num_language_tokens += len(encoding.encode(user_message))
    num_user_tokens += len(encoding.encode(user_message))
    num_language_tokens += len(encoding.encode(assitant_message))
    num_assistant_tokens += len(encoding.encode(assitant_message))


    try:
        next_token = nusc.get('sample', token)['next']
        raw_ids = sft_indices[next_token]['CAM_FRONT']
        ids = parse_ids_any(raw_ids)
        cot_visual = vq_ids_to_string(ids)
    except Exception as e:
        continue


    assistant_value = cot_visual + "\n" + assitant_message

    train_message = {
        "id": token,
        "images": images_path,
        "system": system,
        "conversations": [
            {
                "from": "user",
                "value": (
                    "Here are current six images from the car: "
                    "'CAM_FRONT': <image>\n,'CAM_FRONT_LEFT': <image>\n,'CAM_FRONT_RIGHT': <image>\n,"
                    "'CAM_BACK': <image>\n,'CAM_BACK_LEFT': <image>\n,'CAM_BACK_RIGHT': <image>\n"
                    + user_message +
                    "Based on the provided particulars, please output the CAM_FRONT image at the 1.0 second in the future "
                    "and then plan waypoints (0.5s intervals) for the next 3 seconds.\n"
                )
            },
            {
                "from": "assistant",
                "value": assistant_value
            }
        ]
    }
    train_messages.append(train_message)

print("#### Cost Summarization ####")
print(f"Number of user tokens: {num_user_tokens}")
print(f"Number of assistant tokens: {num_assistant_tokens}")
print(f"Number of total tokens: {num_language_tokens}")

out_path = f"./LLaMA-Factory/data/{args.split}_cot_motion.json"
with open(out_path, "w") as f:
    json.dump(train_messages, f, indent=2, ensure_ascii=False)
print(f"[OK] wrote {out_path} with {len(train_messages)} samples.")

