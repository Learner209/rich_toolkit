import os
import json
import torch
import pickle
import trimesh
from smplx import SMPLH, SMPLX
import numpy as np
from egoallo import training_utils

training_utils.ipdb_safety_net()
## input arguments
SET = 'train'
SEQ_NAME = 'ParkingLot1_007_eating1'
SCENE_NAME, SUB_ID, _ = SEQ_NAME.split('_')
FRAME_ID = 115
CAMERA_ID = 0
gender_mapping = json.load(open('resource/gender.json','r'))
GENDER = gender_mapping[f'{int(SUB_ID)}']
imgext = json.load(open('resource/imgext.json','r'))
EXT = imgext[SCENE_NAME]

## SMPLH model
SMPLH_MODEL_DIR = '/home/minghao/src/robotflow/egoallo/assets/smpl_based_model/smplh'
body_model = SMPLH(
    SMPLH_MODEL_DIR,
    gender=GENDER,
    flat_hand_mean=False,
    use_pca=True,
    num_pca_comps=12,
)

SMPLX_MODEL_DIR = 'body_models/smplx'
body_model = SMPLX(
    SMPLX_MODEL_DIR,
    gender=GENDER,
    num_pca_comps=12,
    flat_hand_mean=False,
    create_expression=True,
    create_jaw_pose=True,
)

## passing the parameters through SMPL-H
smplh_params_fn = os.path.join('data/bodies', SET, SEQ_NAME, f'{FRAME_ID:05d}', f'{SUB_ID}.pkl')
body_params = pickle.load(open(smplh_params_fn,'rb'))
body_params = {k: torch.from_numpy(v) for k, v in body_params.items()}

# import ipdb; ipdb.set_trace()
model_output = body_model(return_verts=True,   
                        body_pose=body_params['body_pose'],
                        global_orient=body_params['global_orient'],
                        transl=body_params['transl'],
                        left_hand_pose=body_params['left_hand_pose'],
                        right_hand_pose=body_params['right_hand_pose'],
                        return_full_pose=True) 
vertices = model_output.vertices.detach().cpu().squeeze().numpy()
purple_color = np.ones((len(vertices), 4)) * [0.5, 0.0, 0.5, 1.0]  # Purple color in RGBA
mesh = trimesh.Trimesh(
    vertices=vertices,
    faces=body_model.faces,
    vertex_colors=purple_color,
    process=False
)

with open(f'data/multicam2world/{SCENE_NAME}_multicam2world.json', 'r') as f:
    cam2scan = json.load(f)
rot_mat = np.array(cam2scan['R'])
translation = cam2scan['t']


## scan
# Load and downsample scene scan for efficiency
scene_scan = trimesh.load(f'data/scan_calibration/{SCENE_NAME}/scan_camcoord.ply', process=False)
# Downsample to ~10% of original faces using vertex decimation
reduction_ratio = 0.9  # This will reduce to 10% of original faces (same as before)
scene_scan = scene_scan.simplify_quadric_decimation(reduction_ratio)
scan = scene_scan + mesh
print(scan.vertices.shape)
scan.vertices = cam2scan['c'] * scan.vertices @ rot_mat + translation
import ipdb; ipdb.set_trace()

scan.export(f'samples/body_scene_world.ply')

