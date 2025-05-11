import sys
import os

# Ensure project root is on path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

import argparse
import torch
from PIL import Image
from app.all_models import model_zoo
from app.custom_models.mvimg_prediction import run_mvprediction
from app.custom_models.normal_prediction import predict_normals
from scripts.refine_lr_to_sr import run_sr_fast
from scripts.utils import save_glb_and_video
from scripts.multiview_inference import geo_reconstruct
from pytorch3d.structures import Meshes

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--remove_bg", action="store_true")
    parser.add_argument("--refine", action="store_true")
    parser.add_argument("--render_video", action="store_true")
    parser.add_argument("--expansion_weight", type=float, default=0.1)
    parser.add_argument("--init_type", type=str, default="std")
    args = parser.parse_args()

    # Initialize the model pipeline
    model_zoo.init_models()
    # Reset any existing device_map so we can place on GPU
    pipe = model_zoo.pipe_disney_controlnet_tile_ipadapter_i2i
    if hasattr(pipe, 'reset_device_map'):
        pipe.reset_device_map()
    # Move pipeline to CUDA
    pipe.to('cuda')

    # Load image
    img = Image.open(args.input).convert("RGBA")
    if img.size[0] <= 512:
        img = run_sr_fast([img])[0]

    # Predict multiview and normals
    rgb_pils, front_pil = run_mvprediction(
        img,
        remove_bg=args.remove_bg,
        seed=args.seed
    )

    # Reconstruct geometry
    new_meshes = geo_reconstruct(
        rgb_pils,
        None,
        front_pil,
        do_refine=args.refine,
        predict_normal=True,
        expansion_weight=args.expansion_weight,
        init_type=args.init_type
    )
    
    # Adjust vertices
    verts = new_meshes.verts_packed()
    verts = verts / 2 * 1.35
    verts[..., [0, 2]] = -verts[..., [0, 2]]
    new_meshes = Meshes(
        verts=[verts],
        faces=new_meshes.faces_list(),
        textures=new_meshes.textures
    )

    # Save outputs
    save_glb_and_video(
        args.output_dir,
        new_meshes,
        with_timestamp=False,
        dist=3.5,
        fov_in_degrees=2 / 1.35,
        cam_type="ortho",
        export_video=args.render_video
    )

if __name__ == "__main__":
    torch.cuda.empty_cache()
    main()