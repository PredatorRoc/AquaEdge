# infer.py —— 覆盖原文件（修正 init_bhwd 调用为位置参数）
import os
from glob import glob
import cv2
import numpy as np
import torch

from NeuFlow.neuflow import NeuFlow
from data_utils import flow_viz

# ========= 配置 =========
IMG_DIR   = 'test_images'   # 输入图片目录
SAVE_DIR  = 'test_results'  # 输出结果目录
#CKPT_PATH = r'D:\PyCharm\NeuFlow_v2-master\CheckpointFromServer\chairs\step_029000.pth'
CKPT_PATH = r'D:\PyCharm\NeuFlow_v2-master\CheckpointFromServer\things\step_120500.pth'

IMG_W = 768
IMG_H = 432

# 推理是否使用 AMP（推荐 True）
USE_AMP   = True
AMP_DTYPE = torch.float16  # 如遇到显卡不支持 fp16，可改为 torch.bfloat16

def load_image_as_tensor(path, w, h, device, dtype=torch.float32):
    """读取图像 -> resize -> [1,3,H,W] 到 device，保持 float32；精度交给 autocast 控制"""
    im = cv2.imread(path)  # BGR
    if im is None:
        raise FileNotFoundError(f"Image not found: {path}")
    im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
    t = torch.from_numpy(im).permute(2, 0, 1).to(device=device, dtype=dtype)  # [3,H,W], BGR
    return t[None, ...]  # [1,3,H,W]

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 1) 构建模型
    model = NeuFlow().to(device)
    model.eval()  # 推理模式

    # 2) 预热一次（FP32），触发所有懒构建模块（如 mnv4_adapter 的投影卷积）
    with torch.no_grad():
        dummy0 = torch.zeros(1, 3, IMG_H, IMG_W, device=device, dtype=torch.float32)
        dummy1 = torch.zeros(1, 3, IMG_H, IMG_W, device=device, dtype=torch.float32)
        # 预热前准备运行期网格/位置编码（按 FP32）；注意用位置参数
        model.init_bhwd(1, IMG_H, IMG_W, device)
        _ = model(dummy0, dummy1)

    # 3) 现在再加载权重（此时投影层等已存在，避免 Unexpected key）
    ckpt = torch.load(CKPT_PATH, map_location=device)
    state = ckpt['model'] if 'model' in ckpt else ckpt
    model.load_state_dict(state, strict=True)

    # 4) 收集待推理图片对
    patterns = ('*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG')
    img_paths = sorted(set(
        p for pat in patterns for p in glob(os.path.join(IMG_DIR, pat))
    ))

    if len(img_paths) < 2:
        print(f'No enough images in {IMG_DIR}')
        return

    # # 4) 收集待推理图片对
    # img_paths = sorted(glob(os.path.join(IMG_DIR, '*.jpg')))
    # if len(img_paths) < 2:
    #     print(f'No enough images in {IMG_DIR}')
    #     return

    # 5) 推理循环
    autocast_ctx = torch.amp.autocast(
        'cuda',
        enabled=(USE_AMP and device.type == 'cuda'),
        dtype=AMP_DTYPE
    )

    with torch.no_grad(), autocast_ctx:
        for p0, p1 in zip(img_paths[:-1], img_paths[1:]):
            print(p0)
            img0 = load_image_as_tensor(p0, IMG_W, IMG_H, device, dtype=torch.float32)
            img1 = load_image_as_tensor(p1, IMG_W, IMG_H, device, dtype=torch.float32)

            # 重要：按当前 batch/分辨率重新准备运行期网格（位置参数）
            model.init_bhwd(img0.shape[0], IMG_H, IMG_W, device)

            # 前向
            preds = model(img0, img1)
            flow  = preds[-1][0]                 # [2,H,W]
            flow  = flow.float().permute(1, 2, 0).cpu().numpy()

            vis = flow_viz.flow_to_image(flow)   # [H,W,3] uint8 (RGB)

            top_bgr = cv2.resize(cv2.imread(p0), (IMG_W, IMG_H))
            out     = np.vstack([top_bgr, vis[:, :, ::-1]])  # flow_viz 是 RGB，这里转 BGR 拼接

            file = os.path.basename(p0)
            cv2.imwrite(os.path.join(SAVE_DIR, file), out)

    print('Done.')

if __name__ == '__main__':
    main()
