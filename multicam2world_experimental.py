import os
import json
import torch
import pickle
import trimesh
import numpy as np
from pathlib import Path

from egoallo.fncsmpl import SmplhModel
from egoallo.transforms import SE3, SO3
from egoallo import training_utils
from smplx import SMPLX, SMPLH	
training_utils.ipdb_safety_net()

## input arguments
SET = 'train'
SEQ_NAME = 'ParkingLot2_008_pushup2'
SCENE_NAME, SUB_ID, _ = SEQ_NAME.split('_')
FRAME_ID = 115
CAMERA_ID = 0
gender_mapping = json.load(open('/home/minghao/src/robotflow/egoallo/third_party/rich_toolkit/resource/gender.json','r'))
GENDER = gender_mapping[f'{int(SUB_ID)}']
imgext = json.load(open('/home/minghao/src/robotflow/egoallo/third_party/rich_toolkit/resource/imgext.json','r'))
EXT = imgext[SCENE_NAME]

# Load camera to world transform
with open(f'/home/minghao/src/robotflow/egoallo/datasets/RICH/data/multicam2world/{SCENE_NAME}_multicam2world.json', 'r') as f:
    cam2scan = json.load(f)
R_cam_world = torch.from_numpy(np.array(cam2scan['R'])).float()
R_cam_world = SO3.from_matrix(R_cam_world)
t_cam_world = torch.from_numpy(np.array(cam2scan['t'])).float()
s_world_cam = cam2scan['c']


SMPLX_MODEL_DIR = '/home/minghao/src/robotflow/egoallo/assets/smpl_based_model/smplx'
body_model = SMPLX(
    SMPLX_MODEL_DIR,
    gender=GENDER,
    flat_hand_mean=False,
    use_pca=True,
    num_pca_comps=12,
)

## passing the parameters through SMPL-H
smplh_params_fn = os.path.join('/home/minghao/src/robotflow/egoallo/datasets/RICH/data/bodies', SET, SEQ_NAME, f'{FRAME_ID:05d}', f'{SUB_ID}.pkl')
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

ref_verts = model_output.vertices.detach().cpu().squeeze().numpy()
ref_jnts = model_output.joints.detach().cpu().squeeze().numpy()
purple_color = np.ones((len(ref_verts), 4)) * [0.5, 0.0, 0.5, 1.0]  # Purple color in RGBA
canonical_mesh = trimesh.Trimesh(
    vertices=ref_verts,
    faces=body_model.faces,
    vertex_colors=purple_color,
    process=False
)

R_cam_root = SO3.exp(model_output.global_orient)
R_world_root = R_cam_world.inverse().multiply(R_cam_root)

T_world_root = SE3.from_rotation_and_translation(
    rotation=R_world_root,
    translation=model_output.joints[0:1, 0, :3],
).parameters()
# import ipdb; ipdb.set_trace()



## Load body parameters
smplx_params_fn = os.path.join('/home/minghao/src/robotflow/egoallo/datasets/RICH/data/bodies', SET, SEQ_NAME, f'{FRAME_ID:05d}', f'{SUB_ID}.pkl')
with open(smplx_params_fn, 'rb') as f:
    body_params = pickle.load(f)
body_params = {k: torch.from_numpy(v).float() for k, v in body_params.items()}

global_orient = body_params['global_orient'].reshape(-1)
if len(global_orient) != 3:
    raise ValueError(f"Expected global_orient to have 3 elements, got {len(global_orient)}")

SMPLX_MODEL_DIR = Path('/home/minghao/src/robotflow/egoallo/assets/smpl_based_model/smplh')
body_model = SmplhModel.load(SMPLX_MODEL_DIR / f"SMPLH_{GENDER.upper()}.pkl")

# Convert hand PCA to axis-angle
left_hand_pose, right_hand_pose = body_model.convert_hand_poses(
    left_hand_pca=body_params['left_hand_pose'],
    right_hand_pca=body_params['right_hand_pose']
)

body_rots = body_params['body_pose'].reshape(-1, 21, 3)
body_quats = SO3.exp(body_rots).wxyz

# Create posed model
shaped = body_model.with_shape(body_params['betas'])
posed = shaped.with_pose_decomposed(
    T_world_root=T_world_root,
    body_quats=body_quats,
    left_hand_quats=SO3.exp(left_hand_pose.reshape(-1, 15, 3)).wxyz,
    right_hand_quats=SO3.exp(right_hand_pose.reshape(-1, 15, 3)).wxyz
)

# Get mesh
mesh = posed.lbs()
# Create red color array for all vertices
red_color = np.ones((len(mesh.verts.squeeze()), 4)) * [1.0, 0.0, 0.0, 1.0]  # Red color in RGBA

human_mesh = trimesh.Trimesh(
    vertices=mesh.verts.squeeze().numpy(force=True),
    faces=mesh.faces.numpy(force=True),
    vertex_colors=red_color,
    process=False
)

# Load and transform scene scan
scene_scan = trimesh.load(f'/home/minghao/src/robotflow/egoallo/datasets/RICH/data/scan_calibration/{SCENE_NAME}/scan_camcoord.ply', process=False)
# Downsample to ~10% of original faces using vertex decimation
reduction_ratio = 0.9  # This will reduce to 10% of original faces
scene_scan = scene_scan.simplify_quadric_decimation(reduction_ratio)


# import ipdb; ipdb.set_trace()

pred_verts = mesh.verts.squeeze().numpy(force=True)


canonical_mesh.vertices = s_world_cam * canonical_mesh.vertices @ R_cam_world.as_matrix().numpy(force=True) + t_cam_world.numpy(force=True)   
# scene_scan.vertices = scale * scene_scan.vertices @ R_world_cam + translation

# scene_scan = scene_scan + human_mesh
# scene_scan = scene_scan + canonical_mesh
scene_scan = human_mesh + canonical_mesh

scene_scan.export(f'/home/minghao/src/robotflow/egoallo/third_party/rich_toolkit/samples/body_scene_world_try.ply')

