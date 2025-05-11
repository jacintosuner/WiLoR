<div align="center">

# Scaling for better 3D Alignment

</div>

## Preamble
This work is motivated by a limitation of WiLoR: its predicted MANO hand model is not aligned with the actual hand in the real data in 3D space as it does not make use of real depth information (see "Before" section below). Nonetheless, WiLoR provides a close alignment between the 2D projection of its predicted 3D hand and the real hand in the image (see "2D Projection for Reference" section below). In this work, we leverage this 2D alignment together with real depth information to rescale the predicted MANO hand and achieve accurate 3D alignment.

## Method

- **Inputs**:  
  A single RGB image $I_\text{RGB}$, depth image $ I_\text{D}$, Camera Intrinsics $ K_{real}$ (either from the color camera or the depth camera, depending on whether we have aligned depth to color or color to depth respectively), the WiLoR predicted hand model $ H_{pred}$, and the predicted WiLoR Camera Intrinsics $ K_{wilor}$. 
  
  The latter is key, as it allows projecting the WiLoR vertices $ V_{wilor}$ onto the image, making the 2D WiLoR points and the 2D real points share the same space.

- **Output**: Predicted WiLoR hand vertices aligned with the real hand points.

- **Steps**:

  1. **Find which pixels in $ I_{RGB}$ correspond to the WiLoR hand**:  
     Render the WiLoR hand and project it onto the image using $ K_{wilor}$, obtaining 2D projected vertices $ PV_{wilor}$. This produces a 2D mask $ mask_{wilor}$ identifying where the projected WiLoR hand appears in the image.

  2. **Find which pixels in $ I_{RGB}$ correspond to the real hand**:  
     Use GSAM on $ I_\text{RGB}$ with the prompt *"hand."* to obtain a 2D mask $ mask_\text{real}$ identifying potential hand regions.  
     This step is crucial because $ mask_\text{wilor}$ may incorrectly include pixels corresponding to objects the hand is grasping or background elements, leading to erroneous depth values if used directly.  

     Since GSAM can produce multiple masks, select the one closest to $ mask_\text{wilor}$ using their centroids. This approach supports scenarios with multiple WiLoR hands in the image by independently selecting the closest GSAM mask for each hand.  

     Finally, apply a depth-based filtering step to reduce noise:  
     - Ignore pixels within $ mask_\text{real}$ that have no depth data.  
     - Discard pixels with depth values exceeding a defined threshold.

  3. **Find the common region of pixels between the WiLoR hand and the real hand**:  
     Intersect $ mask_{wilor}$ and $ mask_{real}$ to obtain $ mask_{final}$.

  4. **Find the depths of the WiLoR hand points in this common region**:  
     Render the WiLoR hand and use ray casting to compute the depths of only those points visible from the camera’s perspective, ignoring occluded or hidden areas.  
     Consider only the depths within $ mask_{final}$, obtaining $ regional\_depths_{wilor}$.

  5. **Find the depths of the real hand points in the common region**:  
     Simply filter $ I_D$ using $ mask_{final}$, obtaining $ regional\_depths_{real}$.

  6. **Take the ratio of both depths**:  
     For every corresponding WiLoR and real depth match $ i$, compute:  
     $ratios_i = \frac{regional\_depths_{real, i}}{regional\_depths_{wilor, i}}$

  7. **Take the average of the ratios**:  
     Compute the final scaling ratio as
     $final\_ratio = \text{avg}(ratios)$
     This gives the value to scale the WiLoR hand depths.

  8. **Scale WiLoR hand points**:  
     - Extract the depths of all WiLoR vertices $ depths_{wilor}$.  
     - Scale them:
       $depths_{wilor} = depths_{wilor} \cdot final\_ratio$
     - Reconstruct the 3D points using the new depths and $ K_{real} $, obtaining $ V_{wilor, aligned} $.  

     > **Note**:  
     We use $ K_{real} $ instead of $ K_{wilor} $ to ensure that $ V_{wilor, aligned} $ is expressed in the same coordinate space as the real hand and the rest of the scene.  
     This step not only rescales the depth ($ z $-coordinate) but also correctly maps the $ x $ and $ y $ coordinates, ensuring full spatial alignment.

  9. **Recreate the hand mesh**:  
     Using the aligned 3D vertices $ V_{wilor, aligned} $, reconstruct the final hand mesh $H_{aligned}$.


### Before:
<img src="https://github.com/user-attachments/assets/c34ad910-0fbe-4356-aded-486195a97485" width="300">

### After:
<img src="https://github.com/user-attachments/assets/4a54cdb3-648e-4a45-9d4d-afb4a2cc14ae" width="300">

### 2D Projection for Reference:
<img src="https://github.com/user-attachments/assets/46bf054b-a092-4b77-8b1a-b2a519e7f9f3" width="300">


## Further improvements using GSAM2 segmentation
GSAM2 segmentation is set by default.
<br>
Here's an example taken from the file "demo_rgbk/mug_lots_of_occlusion.npy":
### Before
<img src="https://github.com/user-attachments/assets/a50043ed-e2b3-43a7-a870-0deac4371477" width="300">

### After
<img src="https://github.com/user-attachments/assets/3865c951-5fc7-4883-a1a0-56671b6ce167" width="300">

## Model weights, submodules and mano hand model
```bash
wget https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/detector.pt -P ./pretrained_models/
wget https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/wilor_final.ckpt -P ./pretrained_models/
```

```bash
git submodule update --init --recursive
```

It is also required to download MANO model from [MANO website](https://mano.is.tue.mpg.de). 
Create an account by clicking Sign Up and download the models (mano_v*_*.zip). Unzip and place the right hand model `MANO_RIGHT.pkl` under the `mano_data/` folder. 
Note that MANO model falls under the [MANO license](https://mano.is.tue.mpg.de/license.html).

## Our installation (Cuda 12.1 / 12.4)
```bash
conda create --name wilor python=3.10
conda activate wilor
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
```

Install [GSAM2](https://github.com/jacintosuner/Grounded-SAM-2/) within the same conda environment as WiLoR.
Additional requirements for GSAM2 and visualizations:
```bash
pip install -r third_party/Grounded-SAM-2/grounding-dino/requirements.txt
# Additional requirements from the fork
pip install open3d
pip install pyk4a
```

## Further installation instructions
Download GSAM2 model weights:
```bash
cd third_party/Grounded-SAM-2/checkpoints
bash download_ckpts.sh
cd ../gdino_checkpoints
bash download_ckpts.sh
```

## Running it
```bash
python demo_rgbdk.py --npy_folder demo_rgbdk --out_folder demo_out --save_mesh
python demo_rgbdk.py --npy_folder demo_rgbdk --out_folder demo_out --save_mesh --no_gsam2
```

## Visualizations
Python scripts have been added to the folder [utils](https://github.com/jacintosuner/WiLoR/tree/main/utils) to:
- capture an rbgdk frame (rgb + depth + camera intrinsics): [capture_rgbdk.py](https://github.com/jacintosuner/WiLoR/blob/main/utils/capture_rgbdk.py)
- visualize an rgbdk frame together with / or an obj file together with / or a pcd file (npy file with point cloud info): [visualize_3d_rgbdk_or_obj_or_pcd.py ](https://github.com/jacintosuner/WiLoR/blob/main/utils/visualize_3d_rgbdk_or_obj_or_pcd.py)
- visualize a hand obj file and select the keypoints: [hand_viewer_picker.py](https://github.com/jacintosuner/WiLoR/blob/main/utils/hand_viewer_picker.py)

## Further comments:
I'm not sure if the MANO model also considers the size of the hand. That means that if we scale the hand, it might not correspond anymore to a MANO model hand.

<div align="center">

# WiLoR: End-to-end 3D hand localization and reconstruction in-the-wild

[Rolandos Alexandros Potamias](https://rolpotamias.github.io)<sup>1</sup> &emsp; [Jinglei Zhang]()<sup>2</sup> &emsp; [Jiankang Deng](https://jiankangdeng.github.io/)<sup>1</sup> &emsp; [Stefanos Zafeiriou](https://www.imperial.ac.uk/people/s.zafeiriou)<sup>1</sup>  

<sup>1</sup>Imperial College London, UK <br>
<sup>2</sup>Shanghai Jiao Tong University, China

<font color="blue"><strong>CVPR 2025</strong></font> 

<a href='https://rolpotamias.github.io/WiLoR/'><img src='https://img.shields.io/badge/Project-Page-blue'></a>
<a href='https://arxiv.org/abs/2409.12259'><img src='https://img.shields.io/badge/Paper-arXiv-red'></a>
<a href='https://huggingface.co/spaces/rolpotamias/WiLoR'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-green'></a>
<a href='https://colab.research.google.com/drive/1bNnYFECmJbbvCNZAKtQcxJGxf0DZppsB?usp=sharing'><img src='https://colab.research.google.com/assets/colab-badge.svg'></a>
</div>

<div align="center">

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/wilor-end-to-end-3d-hand-localization-and/3d-hand-pose-estimation-on-freihand)](https://paperswithcode.com/sota/3d-hand-pose-estimation-on-freihand?p=wilor-end-to-end-3d-hand-localization-and)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/wilor-end-to-end-3d-hand-localization-and/3d-hand-pose-estimation-on-ho-3d)](https://paperswithcode.com/sota/3d-hand-pose-estimation-on-ho-3d?p=wilor-end-to-end-3d-hand-localization-and)

</div>

This is the official implementation of **[WiLoR](https://rolpotamias.github.io/WiLoR/)**, an state-of-the-art hand localization and reconstruction model:

![teaser](assets/teaser.png)

## Installation
### [Update] Quick Installation
Thanks to [@warmshao](https://github.com/warmshao) WiLoR can now be installed using a single pip command:  
```
pip install git+https://github.com/warmshao/WiLoR-mini
```
Please head to [WiLoR-mini](https://github.com/warmshao/WiLoR-mini) for additional details. 

**Note:** the above code is a simplified version of WiLoR and can be used for demo only. 
If you wish to use WiLoR for other tasks it is suggested to follow the original installation instructued bellow: 
### Original Installation
```
git clone --recursive https://github.com/rolpotamias/WiLoR.git
cd WiLoR
```

The code has been tested with PyTorch 2.0.0 and CUDA 11.7. It is suggested to use an anaconda environment to install the the required dependencies:
```bash
conda create --name wilor python=3.10
conda activate wilor

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt
```

Download the pretrained models using: 
```bash
wget https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/detector.pt -P ./pretrained_models/
wget https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/wilor_final.ckpt -P ./pretrained_models/
```
It is also required to download MANO model from [MANO website](https://mano.is.tue.mpg.de). 
Create an account by clicking Sign Up and download the models (mano_v*_*.zip). Unzip and place the right hand model `MANO_RIGHT.pkl` under the `mano_data/` folder. 
Note that MANO model falls under the [MANO license](https://mano.is.tue.mpg.de/license.html).
## Demo
```bash
python demo.py --img_folder demo_img --out_folder demo_out --save_mesh 
```
## Start a local gradio demo
You can start a local demo for inference by running:
```bash
python gradio_demo.py
```
## WHIM Dataset
The dataset will be released soon. 

## Acknowledgements
Parts of the code are taken or adapted from the following repos:
- [HaMeR](https://github.com/geopavlakos/hamer/)
- [Ultralytics](https://github.com/ultralytics/ultralytics)

## License 
WiLoR models fall under the [CC-BY-NC--ND License](./license.txt). This repository depends also on [Ultralytics library](https://github.com/ultralytics/ultralytics) and [MANO Model](https://mano.is.tue.mpg.de/license.html), which are fall under their own licenses. By using this repository, you must also comply with the terms of these external licenses.
## Citing
If you find WiLoR useful for your research, please consider citing our paper:

```bibtex
@misc{potamias2024wilor,
    title={WiLoR: End-to-end 3D Hand Localization and Reconstruction in-the-wild},
    author={Rolandos Alexandros Potamias and Jinglei Zhang and Jiankang Deng and Stefanos Zafeiriou},
    year={2024},
    eprint={2409.12259},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```
