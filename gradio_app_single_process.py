import argparse
import glob
import os
import re
import signal
import subprocess
import tempfile
import time
import logging
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import gradio as gr
import numpy as np
from scipy.ndimage import label, find_objects
import psutil
import trimesh
from PIL import Image
import torch
import time, traceback
from typing import NamedTuple
import contextlib
import importlib
import sys
import threestudio
from pytorch_lightning import Trainer
from tqdm import tqdm, trange
from torchvision import transforms
import imageio

from threestudio.utils.config import load_config, ExperimentConfig, parse_structured
from threestudio.utils.typing import *

from utils.tools import rotation_matrix, build_rotation, prune, REORDER_MTX
from scipy.spatial.transform import Rotation as R

INTERACTIVE_N = 8

class BasicPointCloud(NamedTuple):
    points: np.array
    colors: np.array
    normals: np.array

def load_custom_module(module_path):
    module_name = os.path.basename(module_path)
    if os.path.isfile(module_path):
        sp = os.path.splitext(module_path)
        module_name = sp[0]
    try:
        if os.path.isfile(module_path):
            module_spec = importlib.util.spec_from_file_location(
                module_name, module_path
            )
        else:
            module_spec = importlib.util.spec_from_file_location(
                module_name, os.path.join(module_path, "__init__.py")
            )

        module = importlib.util.module_from_spec(module_spec)
        sys.modules[module_name] = module
        module_spec.loader.exec_module(module)
        return True
    except Exception as e:
        print(traceback.format_exc())
        print(f"Cannot import {module_path} module for custom nodes:", e)
        return False


def load_custom_modules():
    node_paths = ["custom"]
    node_import_times = []
    for custom_node_path in node_paths:
        possible_modules = os.listdir(custom_node_path)
        if "__pycache__" in possible_modules:
            possible_modules.remove("__pycache__")

        for possible_module in possible_modules:
            module_path = os.path.join(custom_node_path, possible_module)
            if (
                os.path.isfile(module_path)
                and os.path.splitext(module_path)[1] != ".py"
            ):
                continue
            if module_path.endswith("_disabled"):
                continue
            time_before = time.perf_counter()
            success = load_custom_module(module_path)
            node_import_times.append(
                (time.perf_counter() - time_before, module_path, success)
            )

    if len(node_import_times) > 0:
        print("\nImport times for custom modules:")
        for n in sorted(node_import_times):
            if n[2]:
                import_message = ""
            else:
                   import_message = " (IMPORT FAILED)"
            print("{:6.1f} seconds{}:".format(n[0], import_message), n[1])
        print()

def set_system_status(system, ckpt_path):
    if ckpt_path is None:
        return
    ckpt = torch.load(ckpt_path, map_location="cpu")
    system.set_resume_status(ckpt["epoch"], ckpt["global_step"])

def load_ckpt(ckpt_name):
    if cfg.resume is None:
        if not cfg.use_timestamp and os.path.isfile(ckpt_name):
            print('load_' + ckpt_name)
            cfg.resume = ckpt_name
    
    set_system_status(system, cfg.resume)
    ckpt = torch.load(ckpt_name)
    num_pts = ckpt["state_dict"]["geometry._xyz"].shape[0]
    pcd = BasicPointCloud(
        points=np.zeros((num_pts, 3)),
        colors=np.zeros((num_pts, 3)),
        normals=np.zeros((num_pts, 3)),
    )
    system.geometry.create_from_pcd(pcd, 10)
    system.geometry.training_setup()
    
    o1, o2 = system.load_state_dict(ckpt['state_dict'], strict=False)        
    
    system.to('cuda')
    system.renderer.training=False

def tail(f, window=20):
    # Returns the last `window` lines of file `f`.
    if window == 0:
        return []

    BUFSIZ = 1024
    f.seek(0, 2)
    remaining_bytes = f.tell()
    size = window + 1
    block = -1
    data = []

    while size > 0 and remaining_bytes > 0:
        if remaining_bytes - BUFSIZ > 0:
            # Seek back one whole BUFSIZ
            f.seek(block * BUFSIZ, 2)
            # read BUFFER
            bunch = f.read(BUFSIZ)
        else:
            # file too small, start from beginning
            f.seek(0, 0)
            # only read what was not read
            bunch = f.read(remaining_bytes)

        bunch = bunch.decode("utf-8")
        data.insert(0, bunch)
        size -= bunch.count("\n")
        remaining_bytes -= BUFSIZ
        block -= 1

    return "\n".join("".join(data).splitlines()[-window:])


@dataclass
class ExperimentStatus:
    pid: Optional[int] = None
    progress: str = ""
    log: str = ""
    output_image: Optional[str] = None
    output_video: Optional[str] = None
    output_feat_video: Optional[str] = None   

    def tolist(self):
        return [
            self.progress,
            self.log,
            self.output_video,
            self.output_feat_video,
        ]


EXP_ROOT_DIR = "outputs-gradio"
DEFAULT_PROMPT = "a countryside"
DEFAULT_PROMPT_SIDE = "a countryside"
DEFAULT_PROMPT_EMPTY = "empty"

EXAMPLE_PROMPT_LIST = [
    "a teddy bear at Times Square",
    "Color photo of a corgi made of transparent glass, standing on the riverside in Yosemite National Park",
    "two men climbing stairs",
    "a boy standing near the window",
]

model_name_config = [
    ("3Dit Scene", "custom/threestudio-3dgs/configs/scene_lang.yaml"),
]

config_path = "custom/threestudio-3dgs/configs/scene_lang.yaml"
with open(config_path) as f:
    config_yaml = f.read()
config_obj = load_config(
    config_yaml,
    # set name and tag to dummy values to avoid creating new directories
    cli_args=[
        "name=dummy",
        "tag=dummy",
        "use_timestamp=false",
        f"exp_root_dir={EXP_ROOT_DIR}",
        "system.prompt_processor.prompt=placeholder",
    ],
    from_string=True,
)

def get_current_status(save_root, trial_dir, alive_path):
    status = ExperimentStatus()

    # write the current timestamp to the alive file
    # the watcher will know the last active time of this process from this timestamp
    if os.path.exists(os.path.dirname(alive_path)):
        alive_fp = open(alive_path, "w")
        alive_fp.seek(0)
        alive_fp.write(str(time.time()))
        alive_fp.flush()

    log_path = os.path.join(trial_dir, "logs")
    progress_path = os.path.join(trial_dir, "progress")
    save_path = os.path.join(trial_dir, "save")

    # read current progress from the progress file
    # the progress file is created by GradioCallback
    if os.path.exists(progress_path):
        status.progress = open(progress_path).read()
    else:
        status.progress = "Setting up everything ..."

    # read the last 10 lines of the log file
    if os.path.exists(log_path):
        status.log = tail(open(log_path, "rb"), window=10)
    else:
        status.log = ""

    # get the validation image and testing video if they exist
    if os.path.exists(save_path):
        videos = [f for f in glob.glob(os.path.join(save_path, "*.mp4")) if 'val' in f or 'test' in f]
        steps = [
            int(re.match(r"it(\d+)-(val|test)\.mp4", os.path.basename(f)).group(1))
            for f in videos
        ]
        videos = sorted(list(zip(videos, steps)), key=lambda x: x[1])
        if len(videos) > 0:
            status.output_video = videos[-1][0]

        videos = [f for f in glob.glob(os.path.join(save_path, "*.mp4")) if 'feat' in f]
        steps = [
            int(re.match(r"it(\d+)-feat\.mp4", os.path.basename(f)).group(1))
            for f in videos
        ]
        videos = sorted(list(zip(videos, steps)), key=lambda x: x[1])
        if len(videos) > 0:
            status.output_feat_video = videos[-1][0]

    return status

def create_bounding_boxes(mask):
    labeled_array, num_features = label(mask)
    bounding_boxes = find_objects(labeled_array)
    
    return bounding_boxes

# def train(name, tag, exp_root_dir, prompt, empty_prompt, side_prompt, seed, max_steps, ooi_bbox, image_path):
def train(extras, name, tag, config_path):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env_gpus_str = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    env_gpus = list(env_gpus_str.split(",")) if env_gpus_str else []
    selected_gpus = [0]
    # Always rely on CUDA_VISIBLE_DEVICES if specific GPU ID(s) are specified.
    # As far as Pytorch Lightning is concerned, we always use all available GPUs
    # (possibly filtered by CUDA_VISIBLE_DEVICES).
    devices = -1
    gpu = '0'
    if len(env_gpus) > 0:
        # CUDA_VISIBLE_DEVICES was set already, e.g. within SLURM srun or higher-level script.
        n_gpus = len(env_gpus)
    else:
        selected_gpus = list(gpu.split(","))
        n_gpus = len(selected_gpus)
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu

    import pytorch_lightning as pl
    import torch
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
    from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
    from pytorch_lightning.utilities.rank_zero import rank_zero_only
    import threestudio
    from threestudio.systems.base import BaseSystem
    from threestudio.utils.callbacks import (
        CodeSnapshotCallback,
        ConfigSnapshotCallback,
        CustomProgressBar,
        ProgressCallback,
    )
    from threestudio.utils.config import ExperimentConfig, load_config
    from threestudio.utils.misc import get_rank
    from threestudio.utils.typing import Optional

    logger = logging.getLogger("pytorch_lightning")

    for handler in logger.handlers:
        if handler.stream == sys.stderr:  # type: ignore
            handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))

    # parse YAML config to OmegaConf
    cfg: ExperimentConfig
    cfg = load_config(config_path, cli_args=extras, n_gpus=n_gpus)

    # set a different seed for each device
    pl.seed_everything(cfg.seed + get_rank(), workers=True)

    system.cfg = parse_structured(system.Config, cfg.system)
    system.configure()

    # set up dm
    dm = threestudio.find(cfg.data_type)(cfg.data)

    system.set_save_dir(os.path.join(cfg.trial_dir, "save")) 
    fh = logging.FileHandler(os.path.join(cfg.trial_dir, "logs"))
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(fh)
    callbacks = []
    callbacks += [
        ModelCheckpoint(
            dirpath=os.path.join(cfg.trial_dir, "ckpts"), **cfg.checkpoint
        ),
        LearningRateMonitor(logging_interval="step"),
        CodeSnapshotCallback(
            os.path.join(cfg.trial_dir, "code"), use_version=False
        ),
        ConfigSnapshotCallback(
            config_path,
            cfg,
            os.path.join(cfg.trial_dir, "configs"),
            use_version=False,
        ),
    ]
    callbacks += [
        ProgressCallback(save_path=os.path.join(cfg.trial_dir, "progress"))
    ]
    def write_to_text(file, lines):
        with open(file, "w") as f:
            for line in lines:
                f.write(line + "\n")
    loggers = []
    rank_zero_only(
        lambda: os.makedirs(os.path.join(cfg.trial_dir, "tb_logs"), exist_ok=True)
    )()
    loggers += [
        TensorBoardLogger(cfg.trial_dir, name="tb_logs"),
        CSVLogger(cfg.trial_dir, name="csv_logs"),
    ] + system.get_loggers()
    rank_zero_only(
        lambda: write_to_text(
            os.path.join(cfg.trial_dir, "cmd.txt"),
            ["python " + " ".join(sys.argv), str(args)],
        )
    )()
    trainer = Trainer(
        callbacks=callbacks,
        logger=loggers,
        inference_mode=False,
        accelerator="gpu",
        devices=devices,
        **cfg.trainer,
    )

    trainer.fit(system, datamodule=dm, ckpt_path=cfg.resume)


def launch(
    host='0.0.0.0',
    port=10092,
    listen=False,
    self_deploy=False,
    save_root=".",
    dm=None,
    system=None,
):
    dataloader = dm.test_dataloader()
    iter_dataloader = iter(dataloader)
    sampled_data = []
    for i in range(120):
        data = next(iter_dataloader)
        for k in data.keys():
            try:
                data[k] = data[k].cuda()
            except:
                pass
        sampled_data.append(data) 

    def run(
        image,
        prompt: str,
        side_prompt: str,
        empty_prompt: str,
        seed: int,
        max_steps: int,
    ):
        torch.cuda.empty_cache()
        if prompt in EXAMPLE_PROMPT_LIST:
            print(os.path.join("examples_cache/", prompt, "rgb.mp4"))
            print(os.path.join("examples_cache/", prompt, "feat.mp4"))
            status_list = ["Loaded!", "load log", os.path.join("examples_cache/", prompt, "rgb.mp4"), os.path.join("examples_cache/", prompt, "feat.mp4")]
            yield status_list + [
                gr.update(value="Run", variant="primary", visible=True),
                gr.update(visible=False),
            ] + [gr.update(interactive=True) for _ in range(INTERACTIVE_N)]
        else:
            save_root = '.'
            mask = np.array(image['layers'][0])[..., 3]
            bbox = create_bounding_boxes(mask)
            if len(bbox) == 0:
                status_list = ["You need to use the brush to mark the content of interest over the image!", "load log", None, None]
                yield status_list + [                    
                    gr.update(value="Run", variant="primary", visible=True),
                    gr.update(visible=False),                
                ] + [gr.update(interactive=True) for _ in range(INTERACTIVE_N)]
            else:
                bbox_str = ""
                for each in bbox:
                    bbox_str = bbox_str + f"{each[1].start},{each[0].start},{each[1].stop},{each[0].stop},"
                bbox_str = '[' + bbox_str + ']'
        
                # update status every 1 second
                status_update_interval = 1
        
                # save the config to a temporary file
                config_file = tempfile.NamedTemporaryFile()
        
                with open(config_file.name, "w") as f:
                    f.write(config_yaml)
        
                # manually assign the output directory, name and tag so that we know the trial directory
                name = os.path.basename(config_path).split(".")[0]
                tag = prompt + '_' + datetime.now().strftime("%Y%m%d-%H%M%S")
                trial_dir = os.path.join(save_root, EXP_ROOT_DIR, name, tag)
                alive_path = os.path.join(trial_dir, "alive")
                img_path = os.path.join(save_root, EXP_ROOT_DIR, f"{name}-{tag}.png")
                Image.fromarray(np.array(image['background'])[...,:3]).save(img_path)
    
                width, height = image['background'].size
                extras = [
                        f'name="{name}"',
                        f'tag="{tag}"',
                        # "trainer.enable_progress_bar=false",
                        f"exp_root_dir={os.path.join(save_root, EXP_ROOT_DIR)}",
                        "use_timestamp=false",
                        f'system.prompt_processor.prompt="{prompt}"',
                        f'system.empty_prompt="{empty_prompt}"',
                        f'system.side_prompt="{side_prompt}"',
                        f"system.guidance.guidance_scale=5",
                        f"seed={seed}",
                        f"trainer.max_steps={max_steps}",
                        "trainer.num_sanity_val_steps=120",
                        f"system.geometry.ooi_bbox={bbox_str}",
                        f"system.geometry.geometry_convert_from=depth:{img_path}",
                        f"system.geometry.img_resolution=[{width},{height}]",
                        f"data.width={width}", f"data.height={height}", f"data.eval_width={width}", f"data.eval_height={height}",
                        # system.outpaint_step=500 system.crop_with_lang=True  system.guidance.max_step_percent=[0,0.5,0.1,1000] system.geometry.max_scaling=0.2
                    ]
                thread = threading.Thread(target=train, args=(extras, name, tag, config_file.name))
                thread.start()
    
                while thread.is_alive():
                    thread.join(timeout=1)
                    status = get_current_status(save_root, trial_dir, alive_path)
    
                    yield status.tolist() + [
                        gr.update(visible=False),
                        gr.update(value="Stop", variant="stop", visible=True),
                    ] + [gr.update(interactive=False) for _ in range(INTERACTIVE_N)]
        
                status.progress = 'Finished!'
                load_ckpt(os.path.join(trial_dir, 'ckpts/last.ckpt'))
    
                yield status.tolist() + [
                    gr.update(value="Run", variant="primary", visible=True),
                    gr.update(visible=False),
                ] + [gr.update(interactive=True) for _ in range(INTERACTIVE_N)]
    
    def stop_run(pid):
        return [
            gr.update(
                value="Please Refresh the Page",
                variant="secondary",
                visible=True,
                interactive=False,
            ),
            gr.update(visible=False),
        ]

    def inference_image(x_offset, y_offset, z_offset, rotate, prompt):
        torch.cuda.empty_cache()
        if prompt in EXAMPLE_PROMPT_LIST:
            load_ckpt(os.path.join("examples_cache/", prompt, "last.ckpt"))
            # prune(system)

        xyz_bak = system.geometry.get_xyz.data
        rot_bak = system.geometry.get_rotation.data

        offset = torch.zeros(len(system.geometry._xyz), 3)
        ooi_mask = system.geometry.ooi_masks_0.view(-1).byte().to(device='cuda').float()

        offset[ooi_mask.bool(), 0] = x_offset
        offset[ooi_mask.bool(), 1] = y_offset
        offset[ooi_mask.bool(), 2] = z_offset
        system.geometry._xyz = torch.nn.Parameter(system.geometry._xyz + offset.cuda())

        rot_matrix = rotation_matrix(0, 0, rotate).cuda()
        prev_xyz = system.geometry.get_xyz.data
        ooi_xyz = prev_xyz[ooi_mask.bool()]
        mean = ooi_xyz.mean(0)
        ooi_xyz = ooi_xyz - mean
        after_xyz = torch.einsum('ab,nb->na', rot_matrix, ooi_xyz) + mean
        prev_xyz[ooi_mask.bool()] = after_xyz
        # system.geometry._xyz = torch.nn.Parameter(system.geometry._xyz + offset.cuda())
        system.geometry._xyz = torch.nn.Parameter(prev_xyz)

        prev_rotation = system.geometry.get_rotation.data
        prev_rotation_mtx = build_rotation(prev_rotation)
        after_rotation_mtx = torch.einsum('ab,nbc->nac', rot_matrix, prev_rotation_mtx)
        after_rotation = torch.from_numpy(R.from_matrix(after_rotation_mtx.detach().cpu()).as_quat()).cuda().float()
        after_rotation = torch.einsum('ab,nb->na', REORDER_MTX, after_rotation)
        prev_rotation[ooi_mask.bool()] = after_rotation[ooi_mask.bool()]
        system.geometry._rotation = torch.nn.Parameter(prev_rotation)

        with torch.no_grad():
            res = system(sampled_data[59])
            rgb = res['comp_rgb'][0].cpu().numpy()
            rgb = (rgb * 255).astype(np.uint8)

        system.geometry._xyz = torch.nn.Parameter(xyz_bak.cuda())
        system.geometry._rotation = torch.nn.Parameter(rot_bak.cuda())
        return rgb
   
    def inference_video(x_offset, y_offset, z_offset, rotate, prompt):
        torch.cuda.empty_cache()
        if prompt in EXAMPLE_PROMPT_LIST:
            load_ckpt(os.path.join("examples_cache/", prompt, "last.ckpt"))
            # prune(system)

        video_writer = imageio.get_writer('result.mp4', fps=40)
        xyz_bak = system.geometry.get_xyz.data
        rot_bak = system.geometry.get_rotation.data

        offset = torch.zeros(len(system.geometry._xyz), 3)
        ooi_mask = system.geometry.ooi_masks_0.view(-1).byte().to(device='cuda').float()

        offset[ooi_mask.bool(), 0] = x_offset
        offset[ooi_mask.bool(), 1] = y_offset
        offset[ooi_mask.bool(), 2] = z_offset
        system.geometry._xyz = torch.nn.Parameter(system.geometry._xyz + offset.cuda())

        rot_matrix = rotation_matrix(0, 0, rotate).cuda()
        prev_xyz = system.geometry.get_xyz.data
        ooi_xyz = prev_xyz[ooi_mask.bool()]
        mean = ooi_xyz.mean(0)
        ooi_xyz = ooi_xyz - mean
        after_xyz = torch.einsum('ab,nb->na', rot_matrix, ooi_xyz) + mean
        prev_xyz[ooi_mask.bool()] = after_xyz
        system.geometry._xyz = torch.nn.Parameter(prev_xyz)

        prev_rotation = system.geometry.get_rotation.data
        prev_rotation_mtx = build_rotation(prev_rotation)
        after_rotation_mtx = torch.einsum('ab,nbc->nac', rot_matrix, prev_rotation_mtx)
        after_rotation = torch.from_numpy(R.from_matrix(after_rotation_mtx.detach().cpu()).as_quat()).cuda().float()
        after_rotation = torch.einsum('ab,nb->na', REORDER_MTX, after_rotation)
        prev_rotation[ooi_mask.bool()] = after_rotation[ooi_mask.bool()]
        system.geometry._rotation = torch.nn.Parameter(prev_rotation)

        with torch.no_grad():
            for i in range(120):
                res = system(sampled_data[i])
                rgb = res['comp_rgb'][0].cpu().numpy()
                rgb = (rgb * 255).astype(np.uint8)
                video_writer.append_data(rgb)
        video_writer.close()

        system.geometry._xyz = torch.nn.Parameter(xyz_bak.cuda())
        system.geometry._rotation = torch.nn.Parameter(rot_bak.cuda())
        return "result.mp4"

    self_deploy = self_deploy or "TS_SELF_DEPLOY" in os.environ

    css = """
    #examples {color: black !important}
    .dark #examples {color: white !important}
    #config-accordion, #logs-accordion {color: black !important;}
    .dark #config-accordion, .dark #logs-accordion {color: white !important;}
    .stop {background: darkred !important;}
    """

    with gr.Blocks(
        title="3Dit Scene - Web Demo",
        theme=gr.themes.Monochrome(),
        css=css,
    ) as demo:
        with gr.Row(equal_height=True):
            header = """
            # Web demo for 3DitScene: Editing Any Scene via Language-guided Disentangled Gaussian Splatting

            <div>
            <a style="display: inline-block;" href="https://github.com/zqh0253/3DitScene"><img src="https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white"></a>
            <a style="display: inline-block;" href="https://huggingface.co/spaces/qihang/3Dit-Scene?duplicate=true"><img src="https://img.shields.io/badge/-Duplicate%20Space%20to%20skip%20the%20queue-blue?labelColor=white&style=flat&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAAAXNSR0IArs4c6QAAAP5JREFUOE+lk7FqAkEURY+ltunEgFXS2sZGIbXfEPdLlnxJyDdYB62sbbUKpLbVNhyYFzbrrA74YJlh9r079973psed0cvUD4A+4HoCjsA85X0Dfn/RBLBgBDxnQPfAEJgBY+A9gALA4tcbamSzS4xq4FOQAJgCDwV2CPKV8tZAJcAjMMkUe1vX+U+SMhfAJEHasQIWmXNN3abzDwHUrgcRGmYcgKe0bxrblHEB4E/pndMazNpSZGcsZdBlYJcEL9Afo75molJyM2FxmPgmgPqlWNLGfwZGG6UiyEvLzHYDmoPkDDiNm9JR9uboiONcBXrpY1qmgs21x1QwyZcpvxt9NS09PlsPAAAAAElFTkSuQmCC&logoWidth=14" alt="Duplicate Space"></a>
            </div>

            ### Usage
            - Input an image and use the brush to mark the content of interest.
            - Input the text prompt describing the image, the novel views content, and the background content.
            - Hit the `Run` button to start optimization.
            - After optimization, the interactive panel will be activated. You can control the object using the slider.
            - Optionally, you could skip the optimization process by loading examples, and interact with them.
            - If you experience long waiting time, consider duplicate this space.
            - **IMPORTANT NOTE: Keep this tab active when running the model.**
            """
            gr.Markdown(header)

        with gr.Row(equal_height=False):
            pid = gr.State()
            with gr.Column(scale=1):
                # generation status
                status = gr.Textbox(
                    value="Hit the Run button to start.",
                    label="Status",
                    lines=1,
                    max_lines=1,
                )
   
                # brush = gr.Brush(colors="rgba(100, 0, 0, 100)", default_color="rgba(100, 0, 0, 100)", color_mode="fixed")
                # brush = gr.Brush(colors=['#FFFFFF'])
                reference_image = gr.ImageEditor(label="Reference image", sources="upload", type="pil", transforms=None, layers=False)

                # prompt input
                input_prompt = gr.Textbox(value=DEFAULT_PROMPT, label="Describe the image:")
                side_prompt = gr.Textbox(value=DEFAULT_PROMPT_SIDE, label="Prompt the left and right areas in the novel view:")
                empty_prompt = gr.Textbox(value=DEFAULT_PROMPT_EMPTY, label="Prompt the area behind the foreground object:")

                with gr.Row():
                    # seed slider
                    seed_input = gr.Slider(
                        minimum=0, maximum=2147483647, value=0, step=1, label="Seed"
                    )

                    max_steps_input = gr.Slider(
                        minimum=1,
                        maximum=10000 if self_deploy else 5000,
                        value=3000 if self_deploy else 1000,
                        step=1,
                        label="Number of training steps",
                    )

                run_btn = gr.Button(value="Run", variant="primary")
                stop_btn = gr.Button(value="Stop", variant="stop", visible=False)

            with gr.Column(scale=1):
                with gr.Accordion(
                    "See terminal logs", open=False, elem_id="logs-accordion", visible=False
                ):
                    # logs
                    logs = gr.Textbox(label="Logs", lines=10)

                # validation image display
                # output_image = gr.Image(value=None, label="Image")

                # testing video display
                output_title = gr.Textbox(
                    value="Rendered RGB and semantic video",
                    label="The videos will keep updating during optimization. Stay tuned!",
                    lines=1,
                    max_lines=1,
                )
                with gr.Row():
                    output_video = gr.Video(value=None, label="Video")
                    output_feat_video = gr.Video(value=None, label="Video")

                interactive_status = gr.Textbox(
                    value="The panel is not interactive until training is finished or an example is loaded.",
                    label="Interactive Panel",
                    lines=1,
                    max_lines=1,
                )

                x_offset = gr.Slider(
                    minimum=-1,
                    maximum=1,
                    value=0,
                    step=0.1,
                    label="X offset",
                    visible=True,
                    interactive=False
                )

                y_offset = gr.Slider(
                    minimum=-1,
                    maximum=1,
                    value=0,
                    step=0.1,
                    label="Y offset",
                    visible=True,
                    interactive=False
                )

                z_offset = gr.Slider(
                    minimum=-1,
                    maximum=1,
                    value=0,
                    step=0.1,
                    label="Z offset",
                    visible=True,
                    interactive=False
                )

                rotate = gr.Slider(
                    minimum=-15,
                    maximum=15,
                    value=0,
                    step=1,
                    label="Rotation over Z",
                    visible=True,
                    interactive=False
                )

                with gr.Row():
                    inference_image_btn = gr.Button(value="inference single frame", variant="primary", visible=True, interactive=False)
                    inference_video_btn = gr.Button(value="inference Video", variant="primary", visible=True, interactive=False)

                with gr.Row():
                    edit_image = gr.Image(value=None, label="Edited image", interactive=False)
                    edit_video = gr.Video(value=None, label="Edited video", interactive=False)


        interactive_component_list = [x_offset, y_offset, z_offset, rotate, inference_image_btn, inference_video_btn, edit_image, edit_video]

        inputs = [
            reference_image,
            input_prompt,
            side_prompt,
            empty_prompt,
            seed_input,
            max_steps_input,
        ]

        outputs = [
            # pid,
            status,
            logs,
            # output_image,
            output_video,
            output_feat_video,
            run_btn,
            stop_btn,
        ] + interactive_component_list

        run_event = run_btn.click(
            fn=run,
            inputs=inputs,
            outputs=outputs,
            concurrency_limit=1,
        )

        stop_btn.click(
            fn=stop_run,
            inputs=[pid],
            outputs=[run_btn, stop_btn],
            cancels=[run_event],
            queue=False,
        )

        inference_image_btn.click(
            fn=inference_image,
            inputs=[x_offset, y_offset, z_offset, rotate, input_prompt],
            outputs=edit_image,
        )

        inference_video_btn.click(
            fn=inference_video,
            inputs=[x_offset, y_offset, z_offset, rotate, input_prompt],
            outputs=edit_video,
        )

        gr.Examples(
            examples=[
                [{"background": "examples/bear_background.png", "layers": ["examples/bear_layers.png"], "composite": "examples/bear_composite.png"}, "a teddy bear at Times Square", "Times Square", "Times Square", 1, 1500, False],
                [{"background": "examples/corgi_background.png", "layers": ["examples/corgi_layers.png"], "composite": "examples/corgi_composite.png"}, "Color photo of a corgi made of transparent glass, standing on the riverside in Yosemite National Park", "riverside in Yosemite National Park", "riverside in Yosemite National Park", 1, 1500, False],
                [{"background": "examples/stairs_background.png", "layers": ["examples/stairs_layers.png"], "composite": "examples/stairs_composite.png"}, "two men climbing stairs", "stairs", "stairs", 1, 1500, False],
                [{"background": "examples/boy_background.png", "layers": ["examples/boy_layers.png"], "composite": "examples/boy_composite.png"}, "a boy standing near the window", "a house", "empty", 1, 1500, False],
            ],
            inputs=inputs,
            outputs=outputs,
            fn=run,
            cache_examples=True,
            elem_id="examples"
        )

    launch_args = {"server_port": port}
    if listen:
        launch_args["server_name"] = "0.0.0.0"

    print('launch!', launch_args, flush=True)

    demo.queue(default_concurrency_limit=1).launch(**launch_args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args, extra = parser.parse_known_args()

    load_custom_modules()
    
    extras = ['tag=inference', 'system.prompt_processor.prompt=a bedroom, anime style', 'system.geometry.inference_only=true']
    cfg = load_config(config_path, cli_args=extras, n_gpus=1)
    dm = threestudio.find(cfg.data_type)(cfg.data)
    dm.setup(stage='test')
    system = threestudio.find(cfg.system_type)(cfg.system, resumed=False)

    parser.add_argument("--listen", action="store_true")
    parser.add_argument("--hf-space", action="store_true")
    parser.add_argument("--self-deploy", action="store_true")
    parser.add_argument("--save-root", type=str, default=".")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    launch(
        port=args.port,
        listen=args.listen,
        self_deploy=args.self_deploy,
        save_root=args.save_root,
        dm=dm, 
        system=system,
    )
