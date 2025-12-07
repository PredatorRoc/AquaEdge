# infer.py —— 覆盖你的旧版 infer.py 直接用
import os
import time
from glob import glob
import cv2
import numpy as np
import torch

from NeuFlow.neuflow import NeuFlow
from data_utils.frame_utils import InputPadder
from data_utils import flow_viz

# ===== 手动配置 =====
IMG_DIR   = 'test_images'   # 输入图片目录（jpg/png都行）
SAVE_DIR  = 'test_results'  # 结果保存目录
CKPT_PATH = r'D:\PyCharm\NeuFlow_v2-master\CheckpointFromServer\chairs\step_029000.pth'

# 推理选项
USE_AMP   = False            # 为稳妥起见默认 False；若你已修好 Matching 的dtype问题，可改 True
AMP_DTYPE = torch.float16    # 若开启 AMP，建议 float16（支持 bfloat16 的卡也可用 bfloat16）
WARMUP_ITERS = 2             # FPS 统计前的热身次数

def load_image_tensor(path, device):
    """读取原图(BGR)，返回 [1,3,H,W] float32 张量、(H,W)、原始BGR图。"""
    bgr = cv2.imread(path)
    if bgr is None:
        raise FileNotFoundError(f'Cannot read image: {path}')
    h, w = bgr.shape[:2]
    t = torch.from_numpy(bgr).permute(2, 0, 1).to(device=device, dtype=torch.float32)  # [3,H,W]，不做resize/归一化
    return t[None], (h, w), bgr

def build_backbone_and_projs(model: NeuFlow, device, H, W):
    """
    仅构建 backbone 与 1x1 投影(懒创建的层)，避免严格加载时 'Unexpected key(s)'。
    不经过整网forward，不触发 matching / transformer。
    """
    # pad 一下，保证/16整除
    dummy = torch.zeros(1, 3, H, W, device=device, dtype=torch.float32)
    padder = InputPadder(dummy.shape, padding_factor=16)
    (dummy_pad,) = padder.pad(dummy)
    Hp, Wp = dummy_pad.shape[-2:]

    # 直接走 adapter 的 forward（只建投影层与pos），不进入 matching
    # 需要先给 adapter 的 s16 位置编码
    # 由于 NeuFlow.forward 内部拼接两帧，所以这里 batch_size=2
    if hasattr(model, 'backbone') and hasattr(model.backbone, 'init_bhwd'):
        # 适配我们自定义的 MobileNetV4 adapter 的接口: (batch, h16, w16, device, amp)
        h16, w16 = Hp // 16, Wp // 16
        try:
            model.backbone.init_bhwd(batch_size=2, h16=h16, w16=w16, device=device, amp=False)
        except TypeError:
            # 有些版本是位置参数
            model.backbone.init_bhwd(2, h16, w16, device, False)

        # 准备一个拼接后的 [B*2,3,H,W] 输入，触发 _build_projs_if_needed
        x_cat = torch.zeros(2, 3, Hp, Wp, device=device, dtype=torch.float32)
        _ = model.backbone(x_cat)  # 只经过 backbone + 1x1 投影 + pos 拼接

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True

    # 1) 收集图片
    paths = sorted(glob(os.path.join(IMG_DIR, '*.jpg')) + glob(os.path.join(IMG_DIR, '*.png')))
    if len(paths) < 2:
        print(f'No enough images in {IMG_DIR}')
        return

    # 2) 读取第一张，确定尺寸，构建模型 & 懒创建层
    img0_t, (H, W), img0_bgr = load_image_tensor(paths[0], device)
    model = NeuFlow().to(device).eval()

    # 3) 先把 adapter 的投影层等懒创建出来，再严格加载权重
    build_backbone_and_projs(model, device, H, W)

    ckpt = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(ckpt['model'], strict=True)

    # 4) 统计信息
    n_params = sum(p.numel() for p in model.parameters())
    print(f'Model params: {n_params/1e6:.2f}M')

    # 5) 逐对推理（只 padding 不缩放）
    times = []
    amp_ctx = torch.amp.autocast('cuda', enabled=USE_AMP, dtype=AMP_DTYPE) if device.type == 'cuda' else torch.cpu.amp.autocast(enabled=False)

    with torch.no_grad(), amp_ctx:
        for i, (p0, p1) in enumerate(zip(paths[:-1], paths[1:])):
            print(p0)
            img0_t, (H, W), img0_bgr = load_image_tensor(p0, device)
            img1_t, (H1, W1), img1_bgr = load_image_tensor(p1, device)
            assert (H, W) == (H1, W1), 'Two frames must have the same resolution'

            # pad 到 16 的倍数
            padder = InputPadder(img0_t.shape, padding_factor=16)
            img0_pad, img1_pad = padder.pad(img0_t, img1_t)
            Hp, Wp = img0_pad.shape[-2:]

            # NeuFlow 运行期缓存（Matching 网格等）
            try:
                model.init_bhwd(img0_pad.shape[0], Hp, Wp, device)
            except TypeError:
                # 某些实现可能还要求 amp 参数, 尝试两版
                try:
                    model.init_bhwd(img0_pad.shape[0], Hp, Wp, device, USE_AMP)
                except Exception:
                    model.init_bhwd(img0_pad.shape[0], Hp, Wp, device)

            # 计时
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.time()

            preds = model(img0_pad, img1_pad)
            flow_pad = preds[-1].float()  # [1,2,Hp,Wp]

            if device.type == 'cuda':
                torch.cuda.synchronize()
            t1 = time.time()

            if i >= WARMUP_ITERS:
                times.append(t1 - t0)

            # 去 padding，回到原始分辨率（无缩放）
            flow = padder.unpad(flow_pad[0]).permute(1, 2, 0).cpu().numpy()  # [H,W,2]
            flow_rgb = flow_viz.flow_to_image(flow)  # RGB uint8

            # 可视化：上原图、下 flow
            out = np.vstack([img0_bgr, flow_rgb[:, :, ::-1]])  # flow_viz是RGB，这里转BGR拼图
            fname = os.path.basename(p0)
            cv2.imwrite(os.path.join(SAVE_DIR, fname), out, [cv2.IMWRITE_JPEG_QUALITY, 95])

    # 6) FPS
    if times:
        avg = sum(times) / len(times)
        fps = 1.0 / avg if avg > 0 else 0.0
        print(f'Avg FPS (excl. {WARMUP_ITERS} warmup): {fps:.2f}')
    print('Done.')

if __name__ == '__main__':
    main()
