# gen_index_nuscenes_mini.py
import json, os
from nuscenes.nuscenes import NuScenes

cams = ["CAM_FRONT_LEFT","CAM_FRONT","CAM_FRONT_RIGHT",
        "CAM_BACK_LEFT","CAM_BACK","CAM_BACK_RIGHT"]

dataroot = "data_mini"
nusc = NuScenes("v1.0-mini", dataroot=dataroot, verbose=True)

with open("data/index_nuscenes_mini.jsonl","w") as f:
    for sample in nusc.sample:                     # 遍历 mini 集全部样本
        scene = nusc.get('scene', sample['scene_token'])['name']
        cams_rel = {
            cam: os.path.join("data_mini/v1.0-mini",
                              nusc.get('sample_data', sample['data'][cam])['filename'])
            for cam in cams
        }
        occ_path = f"occ_gt_npz/{sample['token']}.npz"
        rec = {
            "scene": scene,
            "sample_token": sample["token"],
            "cams": cams_rel,
            "occ_gt_npz": occ_path
        }
        f.write(json.dumps(rec, ensure_ascii=False)+"\n")
