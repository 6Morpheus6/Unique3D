import os
import sys
import tempfile
import subprocess
import gradio as gr
from PIL import Image
from pytorch3d.structures import Meshes
from app.utils import clean_up
from app.custom_models.mvimg_prediction import run_mvprediction
from app.custom_models.normal_prediction import predict_normals
from scripts.refine_lr_to_sr import run_sr_fast
from scripts.utils import save_glb_and_video
from scripts.multiview_inference import geo_reconstruct

# Generate 3D in a sub-process, emit files as prefix.glb and prefix.mp4

def generate3dv2(preview_img, input_processing, seed, render_video=True, do_refine=True, expansion_weight=0.1, init_type="std"):
    if preview_img is None:
        raise gr.Error("preview_img is none")

    # Write input to temporary image file
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_img:
        preview_img.save(tmp_img.name)
        # Create random prefix for output files
        tmp_prefix = tempfile.NamedTemporaryFile(delete=False).name

    # Locate sub-process script and Python executable
    python_exe = sys.executable
    script_dir = os.path.dirname(__file__)
    script_path = os.path.abspath(os.path.join(script_dir, "../scripts/run_generate3dv2.py"))

    # Build arguments
    args = [
        python_exe,
        script_path,
        "--input", tmp_img.name,
        "--output_dir", tmp_prefix,
        "--seed", str(seed),
        "--expansion_weight", str(expansion_weight),
        "--init_type", init_type
    ]
    if input_processing:
        args.append("--remove_bg")
    if render_video:
        args.append("--render_video")
    if do_refine:
        args.append("--refine")

    # Run the sub-process
    subprocess.run(args, check=True)

    # Expect tmp_prefix.glb and tmp_prefix.mp4
    mesh_path = tmp_prefix + ".glb"
    video_path = tmp_prefix + ".mp4"
    return mesh_path, (video_path if os.path.exists(video_path) else None)

#######################################
def create_ui(concurrency_id="wkl"):
    with gr.Row():
        with gr.Column(scale=2):
            input_image = gr.Image(type='pil', image_mode='RGBA', label='Frontview')
            example_folder = os.path.join(os.path.dirname(__file__), "./examples")
            example_fns = sorted([os.path.join(example_folder, path) for path in os.listdir(example_folder)])
            gr.Examples(
                examples=example_fns,
                inputs=[input_image],
                cache_examples=False,
                label='Examples',
                examples_per_page=12
            )
        with gr.Column(scale=3):
            output_mesh = gr.Model3D(value=None, label="Mesh Model", show_label=True, height=320)
            output_video = gr.Video(label="Preview", show_label=True, show_share_button=True, height=320, visible=False)
            input_processing = gr.Checkbox(value=True, label='Remove Background')
            do_refine = gr.Checkbox(value=True, label="Refine Multiview Details", visible=False)
            expansion_weight = gr.Slider(minimum=-1., maximum=1.0, value=0.1, step=0.1, label="Expansion Weight", visible=False)
            init_type = gr.Dropdown(choices=["std", "thin"], label="Mesh Initialization", value="std", visible=False)
            setable_seed = gr.Slider(-1, 1000000000, -1, step=1, label="Seed")
            render_video = gr.Checkbox(value=False, label="generate video", visible=False)
            fullrunv2_btn = gr.Button('Generate 3D', interactive=True)
    fullrunv2_btn.click(
        fn=generate3dv2,
        inputs=[input_image, input_processing, setable_seed, render_video, do_refine, expansion_weight, init_type],
        outputs=[output_mesh, output_video],
        concurrency_id=concurrency_id,
        api_name="generate3dv2",
    ).success(clean_up, api_name=False)
    return input_image
