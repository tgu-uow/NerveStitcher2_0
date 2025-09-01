import numpy as np
from PIL import Image
from pathlib import Path
import cv2
import torch
import os
import time

from models.matching import Matching
from models.utils import AverageTimer, read_image
# from make_img_list import make_img_list  # 注释掉
from combing_imgv2 import combine

torch.set_grad_enabled(False)

# ====== 你可以在这里自定义输入输出路径 ======
MY_INPUT_DIR = "test_data/test1/"         # 输入图片文件夹
MY_OUTPUT_DIR = "test_data/test1_result"        # 拼接结果保存目录
# ============================================

class stitcher:
    def __init__(self):
        self.input_dir = MY_INPUT_DIR
        self.input_pairs_path = self.input_dir + "/stitching_list.txt"
        self.nms_radius = 4
        self.keypoint_threshold = 0.005
        self.max_keypoints = 1024
        self.superglue = 'indoor'
        self.sinkhorn_iterations = 100
        self.match_threshold = 0.80
        self.resize = [-1, -1]
        self.resize_float = False
        self.force_cpu = False
        self.save_dir = MY_OUTPUT_DIR

# 创建保存目录
if not os.path.exists(MY_OUTPUT_DIR):
    os.makedirs(MY_OUTPUT_DIR)

t1 = time.time()
stitcher = stitcher()
# if os.path.exists(stitcher.input_pairs_path):
#     os.remove(stitcher.input_pairs_path)

# make_img_list(stitcher.input_dir, stitcher.input_pairs_path)  # 注释掉

with open(stitcher.input_pairs_path, 'r') as f:
    pairs = [l.split() for l in f.readlines()]

img_first_stitching = cv2.imread(stitcher.input_dir + pairs[0][0])

if len(stitcher.resize) == 2 and stitcher.resize[1] == -1:
    resize = stitcher.resize[0:1]

device = 'cuda' if torch.cuda.is_available() and not stitcher.force_cpu else 'cpu'
print('使用加速： "{}"'.format(device))
config = {
        'superpoint': {
            'nms_radius': stitcher.nms_radius,
            'keypoint_threshold': stitcher.keypoint_threshold,
            'max_keypoints': stitcher.max_keypoints
        },
        'superglue': {
            'weights': stitcher.superglue,
            'sinkhorn_iterations': stitcher.sinkhorn_iterations,
            'match_threshold': stitcher.match_threshold,
        }
    }
matching = Matching(config).eval().to(device)

Input_dir = Path(stitcher.input_dir)
print('加载图像数据地址： "{}"'.format(Input_dir))
timer = AverageTimer(newline=True)

x_move, y_move = [], []
print("正在运行中......")
count = 0  # 拼接图像计数器(用于拼接断续重连)
segment_id = 1  # 拼接段编号
segment_start = pairs[0][0]  # 当前段起始图片名
for i, pair in enumerate(pairs):
    name0, name1 = pair[:2]
    print("正在匹配", name0, name1)
    if len(pair) >= 5:
        rot0, rot1 = int(pair[2]), int(pair[3])
    else:
        rot0, rot1 = 0, 0

    # 加载图像(for SuperGlue).
    image0, inp0, scales0 = read_image(
        Input_dir / name0, device, resize, rot0, stitcher.resize_float)
    image1, inp1, scales1 = read_image(
        Input_dir / name1, device, resize, rot1, stitcher.resize_float)
    if image0 is None or image1 is None:
        print('问题图像组: {} {}'.format(
            stitcher.input_dir + name0, stitcher.input_dir + name1))
        exit(1)
    timer.update('load_image')

    # 图像检测和匹配部分
    pred = matching({'image0': inp0, 'image1': inp1})
    pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
    kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
    matches, conf = pred['matches0'], pred['matching_scores0']
    timer.update('matcher')

    # 保存匹配点
    valid = matches > -1
    mkpts0 = kpts0[valid]
    mkpts1 = kpts1[matches[valid]]
    mconf = conf[valid]

    try:
        H, SS = cv2.estimateAffinePartial2D(mkpts0, mkpts1)
        assert H is not None
    except:
        if x_move == [] and y_move == []:
            print("%s图像为孤立图像，拼接失败！" % name0)
            img_first_stitching = image1
            count += 1
            segment_start = name1  # 新段起点
            print(name1)
            print("开始下一组图像拼接")
            continue
        else:
            print(len(x_move))
            count, result_img = combine(img_first_stitching, x_move, y_move, stitcher.input_dir, count)
            # 保存当前拼接段大图
            segment_end = name0
            save_path = f"{stitcher.save_dir}/result_{segment_start}_to_{segment_end}.jpg"
            cv2.imwrite(save_path, result_img)
            print(f"已保存拼接段{segment_id}：{save_path}")
            segment_id += 1
            img_first_stitching = image1
            segment_start = name1  # 新段起点
            x_move, y_move = [], []
            print("开始下一组图像拼接")
    else:
        x_move.append(H[0, 2])
        y_move.append(H[1, 2])

# 循环结束后保存最后一段
if x_move != [] and y_move != []:
    count, result_img = combine(img_first_stitching, x_move, y_move, stitcher.input_dir, count)
    segment_end = pairs[-1][1]
    save_path = f"{stitcher.save_dir}/result_{segment_start}_to_{segment_end}.jpg"
    cv2.imwrite(save_path, result_img)
    print(f"已保存拼接段{segment_id}：{save_path}")

t2 = time.time()
print("共检测并拼接%s张图像" % count)
print("拼接完毕！！！")
timer.print('计算耗时：')
print("共耗时：{0}".format(t2-t1))