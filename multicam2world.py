from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import json
import torch
import pickle
import trimesh
import numpy as np

from egoallo.fncsmpl import SmplhModel
from egoallo.transforms import SE3, SO3
from egoallo import training_utils
from smplx import SMPLX, SMPLH

@dataclass
class Config:
    """Configuration class for paths and constants."""
    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = base_dir or Path('/home/minghao/src/robotflow/egoallo')
        self.dataset_dir = self.base_dir / 'datasets/RICH/data'
        self.assets_dir = self.base_dir / 'assets'
        self.resource_dir = self.base_dir / 'third_party/rich_toolkit/resource'
        
        # Load configuration files
        self.gender_mapping = self._load_json('gender.json')
        self.imgext = self._load_json('imgext.json')
        
        # Add camera parameters path
        self.camera_params_dir = self.dataset_dir / 'multicam2world'
    
    def _load_json(self, filename: str) -> Dict:
        """Helper to load JSON files from resource directory."""
        return json.load(open(self.resource_dir / filename))
    
    def get_camera_params_path(self, scene_name: str) -> Path:
        """Get path to camera parameters file."""
        return self.camera_params_dir / f'{scene_name}_multicam2world.json'

class CameraParams:
    """Handles camera parameter loading and processing."""
    def __init__(self, config: Config, scene_name: str):
        self.config = config
        self.scene_name = scene_name
        self._load_camera_params()
    
    def _load_camera_params(self) -> None:
        """Load camera parameters from JSON file."""
        params_path = self.config.get_camera_params_path(self.scene_name)
        with open(params_path, 'r') as f:
            cam2scan = json.load(f)
            
        # Convert parameters to tensors
        self.R_cam_world = SO3.from_matrix(
            torch.from_numpy(np.array(cam2scan['R'])).float()
        )
        self.t_cam_world = torch.from_numpy(np.array(cam2scan['t'])).float()
        self.s_world_cam = cam2scan['c']

class BodyModelProcessor:
    """Handles body model processing and mesh generation."""
    
    def __init__(
        self, 
        config: Config,
        set_name: str,
        seq_name: str,
        frame_id: int,
        camera_id: int = 0
    ):
        self.config = config
        self.set_name = set_name
        self.frame_id = frame_id
        self.camera_id = camera_id
        
        # Parse sequence name
        self.scene_name, self.sub_id, _ = seq_name.split('_')
        self.gender = self.config.gender_mapping[f'{int(self.sub_id)}']
        self.seq_name = seq_name
        
        # Load camera parameters
        self.camera_params = CameraParams(config, self.scene_name)
        
    def _load_smplx_model(self) -> SMPLX:
        """Initialize SMPL-X model."""
        model_dir = self.config.assets_dir / 'smpl_based_model/smplx'
        return SMPLX(
            model_dir,
            gender=self.gender,
            flat_hand_mean=False,
            use_pca=True,
            num_pca_comps=12,
        )
    
    def _load_body_params(self) -> Dict[str, torch.Tensor]:
        """Load and process body parameters."""
        params_path = (
            self.config.dataset_dir / 'bodies' / self.set_name / self.seq_name /
            f'{self.frame_id:05d}' / f'{self.sub_id}.pkl'
        )
        with open(params_path, 'rb') as f:
            params = pickle.load(f)
        return {k: torch.from_numpy(v).float() for k, v in params.items()}
    
    def _process_smplx(self, model: SMPLX, params: Dict[str, torch.Tensor]) -> Any:
        """Process SMPL-X model with parameters."""
        return model(
            return_verts=True,   
            body_pose=params['body_pose'],  # Shape: (B, J*3)
            global_orient=params['global_orient'],  # Shape: (B, 3)
            transl=params['transl'],  # Shape: (B, 3)
            left_hand_pose=params['left_hand_pose'],  # Shape: (B, 45)
            right_hand_pose=params['right_hand_pose'],  # Shape: (B, 45)
            return_full_pose=True
        ) 

    def _create_mesh(self, model_output: Any, vertices: np.ndarray, faces: np.ndarray, color: list) -> trimesh.Trimesh:
        """Create colored mesh from model output."""
        vertex_colors = np.ones((len(vertices), 4)) * color  # Shape: (V, 4)
        
        return trimesh.Trimesh(
            vertices=vertices,
            faces=faces,
            vertex_colors=vertex_colors,
            process=False
        )

    def _calculate_world_transform(self, model_output: Any) -> torch.Tensor:
        """Calculate world transform from model output."""
        R_cam_root = SO3.exp(model_output.global_orient)
        R_world_root = self.camera_params.R_cam_world.inverse().multiply(R_cam_root)
        
        t_world_root = (
            self.camera_params.s_world_cam * 
            model_output.joints[0:1, 0, :3] @ 
            self.camera_params.R_cam_world.as_matrix() + 
            self.camera_params.t_cam_world
        )
        
        return SE3.from_rotation_and_translation(
            rotation=R_world_root,
            translation=t_world_root,
        ).parameters()

    def process_models(self) -> Tuple[trimesh.Trimesh, trimesh.Trimesh]:
        """Process both SMPL-X and SMPL-H models and generate meshes."""
        # Process SMPL-X model
        self.smplx_model = self._load_smplx_model()
        body_params = self._load_body_params()
        model_output = self._process_smplx(self.smplx_model, body_params)
        
        # Create canonical mesh
        canonical_mesh = self._create_mesh(
            model_output, 
            vertices=model_output.vertices.detach().cpu().squeeze().numpy(),
            faces=self.smplx_model.faces,
            color=[0.5, 0.0, 0.5, 1.0]  # Purple
        )
        
        # Calculate world transform
        T_world_root = self._calculate_world_transform(model_output)
        
        # Process SMPL-H model
        self.smplh_model = SmplhModel.load(
            self.config.assets_dir / 'smpl_based_model/smplh' / 
            f"SMPLH_{self.gender.upper()}.pkl"
        )
        
        # Convert hand poses
        left_hand_pose, right_hand_pose = self.smplh_model.convert_hand_poses(
            left_hand_pca=body_params['left_hand_pose'],
            right_hand_pca=body_params['right_hand_pose']
        )
        
        # Create posed model
        body_rots = body_params['body_pose'].reshape(-1, 21, 3)
        body_quats = SO3.exp(body_rots).wxyz
        
        shaped = self.smplh_model.with_shape(body_params['betas'])
        posed = shaped.with_pose_decomposed(
            T_world_root=T_world_root,
            body_quats=body_quats,
            left_hand_quats=SO3.exp(left_hand_pose.reshape(-1, 15, 3)).wxyz,
            right_hand_quats=SO3.exp(right_hand_pose.reshape(-1, 15, 3)).wxyz
        )
        
        # Create human mesh
        mesh = posed.lbs()
        human_mesh = self._create_mesh(
            mesh,
            vertices=mesh.verts.detach().cpu().squeeze().numpy(),
            faces=self.smplh_model.faces,
            color=[1.0, 0.0, 0.0, 1.0]  # Red
        )
        
        return human_mesh, canonical_mesh

    def transform_mesh_to_world(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """Transform mesh vertices to world coordinates."""
        mesh.vertices = (
            self.camera_params.s_world_cam * 
            mesh.vertices @ 
            self.camera_params.R_cam_world.as_matrix().numpy(force=True) + 
            self.camera_params.t_cam_world.numpy(force=True)
        )
        return mesh

def main():
    """Main execution function."""
    # Initialize configuration
    config = Config()
    
    # Initialize processor
    processor = BodyModelProcessor(
        config=config,
        set_name='train',
        seq_name='ParkingLot2_008_pushup2',
        frame_id=115
    )
    
    # Process models and get meshes
    human_mesh, canonical_mesh = processor.process_models()
    
    # Transform canonical mesh to world coordinates
    canonical_mesh = processor.transform_mesh_to_world(canonical_mesh)
    
    # Combine meshes
    final_mesh = human_mesh + canonical_mesh
    
    # Export result
    output_path = config.base_dir / 'third_party/rich_toolkit/samples/body_scene_world_try.ply'
    final_mesh.export(output_path)

if __name__ == "__main__":
    training_utils.ipdb_safety_net()
    main()
