# 3DitScene: Editing Any Scene via Language-guided Disentangled Gaussian Splatting

[![Project Page](https://img.shields.io/badge/Project-Website-green)](https://zqh0253.github.io/3DitScene/)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/qihang/3Dit-Scene/)
[![arXiv](https://img.shields.io/badge/arXiv-2405.18424-b31b1b.svg)](https://arxiv.org/abs/2405.18424) 


<table class="center">
    <tr style="line-height: 0">
      <td width=35% style="border: none; text-align: center">Move the bear, and rotate the camera</td>
      <td width=30% style="border: none; text-align: center">Move / remove the girl, and rotate the camera</td>
    </tr>
    <tr style="line-height: 0">
      <td width=35% style="border: none"><img src="assets/bear.gif"></td>
      <td width=30% style="border: none"><img src="assets/cherry.gif"></td>
    </tr>
 </table>

## Installation

+ Install `Python >= 3.8`.
+ Install `Python >= 1.12`. We have tested on `torch==2.0.1+cu118`, but other versions should also work fine.
+ Install dependencies:
```
pip install -r requirements.txt
```
+ Clone our repo:
```
git clone https://github.com/zqh0253/3DitScene.git --recursive
```
+ Install submodules:
```
pip install ./submodules/segment-anything-langsplat
pip install ./submodules/MobileSAM-lang
pip install ./submodules/diff_gaussian_rasterization
pip install ./submodules/simple-knn
```
+ Prepare weights for `SAM`:
```
mkdir ckpts
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -O ./ckpts/sam_vit_h_4b8939.pth
cp submodules/MobileSAM-lang/weights/mobile_sam.pt ./ckpts/
```

## Usage

Run the following command to launch the optimization procedure: 
```
python -u launch.py --config custom/threestudio-3dgs/configs/scene_lang.yaml  --train --gpu 0 tag=3DitScene 
system.geometry.geometry_convert_from=depth:${IMGPATH} system.geometry.ooi_bbox=${BBOX}
system.prompt_processor.prompt="${PROMPT}" system.empty_prompt="${EMPTY_PROMPT}" system.side_prompt="${SIDE_PROMPT}"
```
You should specify the image path `IMGPATH`, the bounding box of the interested object  `BBOX`, and the promtps: `PROMPT`, `EMPTY_PROMPT`, `SIDE_PROMPT`. These prompts describe the image itself, the background area behind the image, and the content of the novel view region, respectively.

Here we provide an image (`./assets/teddy.png`) as example:
```
python -u launch.py --config custom/threestudio-3dgs/configs/scene_lang.yaml  --train --gpu 0 tag=3DitScene 
system.geometry.geometry_convert_from=depth:assets/teddy.png system.geometry.ooi_bbox=[122,119,387,495]
system.prompt_processor.prompt="a teddy bear in Times Square" system.empty_prompt="Times Square, out of focus" system.side_prompt="Times Square, out of focus"
```

## Huggingface demo

We provide a huggingface demo. You can either visit our [online huggingface space](https://huggingface.co/spaces/qihang/3Dit-Scene), or deploy it locally by:
```
python gradio_app_single_process.py --listen --hf-space --port 10091
```

## Citation

If you find our work useful, please consider citing:
```
inproceedings{zhang20243DitScene,
  author = {Qihang Zhang and Yinghao Xu and Chaoyang Wang and Hsin-Ying Lee and Gordon Wetzstein and Bolei Zhou and Ceyuan Yang},
  title = {{3DitScene}: Editing Any Scene via Language-guided Disentangled Gaussian Splatting},
  booktitle = {arXiv},
  year = {2024}
}
```
