<img src="data/lab_logo2.png" width="35%">

# NerveStitcher 2.0  
Deep-Feature-Based, Large-Scale Microscopy Image Stitching Pipeline

> **Author / Maintainer:** Hongyu Qian – TGU-UOW Lab  
> **Built on:** [NerveStitcher v1](https://doi.org/10.1016/j.compbiomed.2022.106303) + SuperPoint / SuperGlue

---

##   Overview
NerveStitcher 2.0 delivers a **one-click, two-stage** stitching workflow for **hundreds to thousands** of partially overlapping microscopy images (corneal nerves, OCT, industrial surfaces, …).  
The pipeline

1. **Segment stitching (`new_stitching.py`)** – iteratively stitches consecutive frames, auto-detects breakpoints, and exports several mid-sized mosaics.  
2. **Global fusion (`bigmapstiching.py`)** – matches and merges the segment mosaics into one (or a few) ultra-large panoramas.

By leveraging SuperPoint + SuperGlue for deep feature matching, the method is far more robust than traditional SIFT/ORB approaches. The segment strategy keeps memory usage low and prevents local failures from ruining the whole set.

---

##   Key Features
**Robust two-stage stitching** – breakpoint self-recovery.  
**Deep feature matching** – SuperPoint/SuperGlue (`indoor` weights).  
**Low memory footprint** – real-time disk flushing; 1000×384² images fit into 8 GB RAM.  
**GPU / CPU auto-switch** – CUDA detected automatically; CPU enforced by `force_cpu`.  
**Single-point configuration** – all paths & thresholds at the top of each script.

---

##   Requirements & Installation
```bash
# create lightweight environment
conda create -n nervestitcher2 python=3.8
conda activate nervestitcher2
pip install -r requirements.txt        # incl. OpenCV-Python, torch, etc.
# for GPU support, install the CUDA-enabled PyTorch that matches your system
```

Download the pretrained weights and place them in `models/weights/`:

(Download link available in the project wiki / Google Drive.)
https://drive.google.com/drive/folders/1SgHwGcFwKbV6Bv7OgV1PbqCWmSJgx3jZ?usp=sharing
---

##   Quick Start
```bash
# 1. Prepare your images + stitching_list.txt
#    Example set shipped in test_data/test1/
# 2. Launch the one-click pipeline
python final_stitching.py
# 3. Inspect results
#    ├─ Segment mosaics : test_data/test1_result/
#    └─ Final panorama  : test_data/test1/test1_fina_result/
```

To use your own dataset, only change **two or three path constants** at the top of `new_stitching.py` and `bigmapstiching.py`.

---

##   Parameters at a Glance

| Script             | Parameter          | Default                              | Description                         |
|--------------------|--------------------|--------------------------------------|-------------------------------------|
| `new_stitching.py` | `MY_INPUT_DIR`     | `test_data/test1/`                   | Small-image input folder            |
|                    | `MY_OUTPUT_DIR`    | `test_data/test1_result/`            | Segment mosaic output folder        |
|                    | `match_threshold`  | `0.80`                               | SuperGlue confidence threshold      |
| `bigmapstiching.py`| `folder`           | `test_data/test1/test1_result`       | Segment mosaics input               |
|                    | `output_dir`       | `test_data/test1/test1_fina_result`  | Final panorama output               |
|                    | `threshold`        | `20`                                 | Gray-value threshold for masks      |

---

##   FAQ
1. **CUDA out of memory**  
   • Reduce `max_keypoints` or force CPU mode (`force_cpu=True`).  
2. **Too many breakpoints / segments**  
   • Decrease `match_threshold` slightly or capture images with more overlap.  
3. **Segment mosaics fail to merge**  
   • Verify overlap exists; print match count in `bigmapstiching.py` for debugging.

---
##  Contact Us
NerveStitcher2.0 can also be used to stitch other microscopy images. Some of the images we have successfully tested are: fundus vascular and thickness OCT images, fundus vascular images.

For additional questions or discussions, Please contact email:

liguangxu@tiangong.edu.cn

litianyu@tiangong.edu.cn

2431080994@tiangong.edu.cn

##   Citation
If this project or its predecessors benefit your research, please cite:

```bibtex
@article{li2022nervestitcher,
  title  = {NerveStitcher: Corneal confocal microscope images stitching with neural networks},
  author = {Li, Guangxu and Li, Tianyu and Li, Fangting and Zhang, Chen},
  journal= {Computers in Biology and Medicine},
  year   = {2022},
  pages  = {106303},
  publisher = {Elsevier}
}
```

---

##   License
Academic research use only – any commercial usage requires written permission.  
© 2023-2024 TGU-UOW Lab – Hongyu Qian
