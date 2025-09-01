import os
import re
import cv2
import torch
import time
import numpy as np
from models.matching import Matching

def natural_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

def get_content_mask(img, threshold=20):
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    mask = (gray > threshold).astype(np.uint8)
    return mask

def mask_blend(canvas, img, mask):
    mask3 = np.repeat(mask[:, :, None], 3, axis=2)
    canvas = canvas * (1 - mask3) + img * mask3
    return canvas.astype(np.uint8)

def read_image_tensor(image, device):
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image.astype('float32')
    tensor = torch.from_numpy(image / 255.).float()[None, None].to(device)
    return tensor

def estimate_affine_superglue(imgA, imgB, matching, device='cpu'):
    inp0 = read_image_tensor(imgA, device)
    inp1 = read_image_tensor(imgB, device)
    with torch.no_grad():
        pred = matching({'image0': inp0, 'image1': inp1})
    pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
    kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
    matches = pred['matches0']
    valid = matches > -1
    if valid.sum() < 4:
        return None
    mkpts0 = kpts0[valid]
    mkpts1 = kpts1[matches[valid]]
    H, _ = cv2.estimateAffinePartial2D(mkpts1, mkpts0)  # imgB -> imgA
    return H

def multi_bigmap_stitch(folder_path, device='cpu', output_dir=None):
    config = {
        'superpoint': {
            'nms_radius': 4,
            'keypoint_threshold': 0.005,
            'max_keypoints': 1024
        },
        'superglue': {
            'weights': 'indoor',
            'sinkhorn_iterations': 100,
            'match_threshold': 0.80,
        }
    }
    matching = Matching(config).eval().to(device)

    img_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
                 if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    img_files = sorted(img_files, key=lambda x: natural_key(os.path.basename(x)))
    if not img_files:
        print("没有可用图片")
        return

    if output_dir is None:
        output_dir = os.path.join(folder_path, "output")
    os.makedirs(output_dir, exist_ok=True)

    unstitched = img_files.copy()
    bigmaps = []
    bigmap_count = 0

    total_start_time = time.time()
    while unstitched:
        img_path = unstitched.pop(0)
        big_img = cv2.imread(img_path)
        stitched_names = [os.path.basename(img_path)]
        changed = True
        while changed and unstitched:
            changed = False
            for i, other_path in enumerate(unstitched):
                other_img = cv2.imread(other_path)
                print(f"尝试拼接: 当前大图包含 {stitched_names} 和 {os.path.basename(other_path)}")
                start_time = time.time()
                H = estimate_affine_superglue(big_img, other_img, matching, device)
                elapsed = time.time() - start_time
                if H is not None:
                    # 计算新画布尺寸
                    hA, wA = big_img.shape[:2]
                    hB, wB = other_img.shape[:2]
                    cornersB = np.array([[0,0],[wB,0],[wB,hB],[0,hB]], dtype=np.float32)
                    warped_cornersB = cv2.transform(cornersB[None], H)[0]
                    all_corners = np.vstack(([[0,0],[wA,0],[wA,hA],[0,hA]], warped_cornersB))
                    [xmin, ymin] = np.floor(all_corners.min(axis=0)).astype(int)
                    [xmax, ymax] = np.ceil(all_corners.max(axis=0)).astype(int)
                    shift = np.array([-xmin, -ymin])
                    canvas_h = ymax - ymin
                    canvas_w = xmax - xmin

                    # 创建新画布并贴大图
                    new_canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
                    new_canvas[shift[1]:shift[1]+hA, shift[0]:shift[0]+wA] = big_img

                    # 仿射变换other_img和掩码
                    H_shift = H.copy()
                    H_shift[:,2] += shift
                    warped_imgB = cv2.warpAffine(other_img, H_shift, (canvas_w, canvas_h))
                    maskB = get_content_mask(other_img)
                    warped_maskB = cv2.warpAffine(maskB, H_shift, (canvas_w, canvas_h))

                    # 掩码融合
                    big_img = mask_blend(new_canvas, warped_imgB, warped_maskB)

                    print(f"拼接成功: {os.path.basename(other_path)} 加入大图，耗时 {elapsed:.2f} 秒")
                    print(f"当前大图尺寸: {big_img.shape}")
                    stitched_names.append(os.path.basename(other_path))
                    unstitched.pop(i)
                    changed = True
                    break  # 重新从头遍历
                else:
                    print(f"拼接失败: {os.path.basename(other_path)}，耗时 {elapsed:.2f} 秒")
        bigmaps.append((big_img, stitched_names))

        # 每完成一张大图就自动保存
        save_path = os.path.join(output_dir, f"bigmap_{bigmap_count}.jpg")
        cv2.imwrite(save_path, big_img)
        print(f"\n大图{bigmap_count}包含图片：")
        for n in stitched_names:
            print(n)
        print(f"已保存为: {save_path}")
        bigmap_count += 1

    total_elapsed = time.time() - total_start_time
    print(f"\n全部拼接流程总耗时：{total_elapsed:.2f} 秒")

if __name__ == "__main__":
    folder = "test_data/test1/test1_result"
    output_dir = "test_data/test1/test1_fina_result"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    multi_bigmap_stitch(folder, device, output_dir)