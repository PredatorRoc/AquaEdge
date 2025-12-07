# -*- coding: utf-8 -*-
import os
import glob
import argparse
import time
import numpy as np
import torch
import cv2
from PIL import Image
import concurrent.futures

# ====== YOLO（Ultralytics 8.1.x）======
try:
    from ultralytics import YOLO
    _HAS_ULTRA = True
except Exception as e:
    print("[YOLO Import Error]", repr(e))
    _HAS_ULTRA = False

# ====== NeuFlow 依赖（假设含 NeuFlow/ 包）======
try:
    from NeuFlow.neuflow import NeuFlow
except Exception as e:
    raise RuntimeError("未找到 NeuFlow 包，请确认 NeuFlow 在 PYTHONPATH 或同级目录存在 NeuFlow/ 目录") from e

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ==================== 基础工具 ====================
IMG_EXTS = {'.png','.jpg','.jpeg','.bmp','.tif','.tiff','.webp','.PNG','.JPG','.JPEG','.BMP','.TIF','.TIFF','.WEBP'}
VID_EXTS = {'.mp4','.avi','.mov','.mkv','.flv','.wmv','.MP4','.AVI','.MOV','.MKV','.FLV','.WMV'}

def load_image_np_rgb(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert('RGB'))

def to_bgr_tensor_4d_from_rgb(img_rgb_np: np.ndarray, w: int, h: int, device, dtype=torch.float32):
    """RGB numpy -> resize(w,h) -> BGR -> torch [1,3,h,w] on device"""
    img_bgr = cv2.cvtColor(img_rgb_np, cv2.COLOR_RGB2BGR)
    img_bgr = cv2.resize(img_bgr, (w, h), interpolation=cv2.INTER_LINEAR)
    t = torch.from_numpy(img_bgr).permute(2, 0, 1).to(device=device, dtype=dtype)
    return t[None, ...]

# ==================== YOLO 掩膜 ====================
def ensure_yolo_segment_task(model: YOLO):
    try:
        if getattr(model, 'task', None) != 'segment':
            model.task = 'segment'
        if hasattr(model, 'overrides') and isinstance(model.overrides, dict):
            model.overrides['task'] = 'segment'
    except Exception:
        pass

def union_mask_yolo(model: YOLO, img_rgb_np: np.ndarray, profile: bool = False):
    H_orig, W_orig = img_rgb_np.shape[:2]
    scale_factor = 0.5
    new_w, new_h = int(W_orig * scale_factor), int(H_orig * scale_factor)

    img_small = cv2.resize(img_rgb_np, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    t0 = time.perf_counter()
    ensure_yolo_segment_task(model)
    try:
        r = model(img_small, verbose=False)[0]
    except Exception as e:
        print('[YOLO call error]', repr(e))
        h, w = img_rgb_np.shape[:2]
        mask = np.ones((h, w), dtype=np.uint8)  # 退化为全图
        t1 = time.perf_counter()
        seg_time = t1 - t0
        if profile:
            return mask, seg_time
        else:
            return mask

    H, W = img_rgb_np.shape[:2]
    mask = np.zeros((H, W), dtype=np.uint8)

    has_masks = hasattr(r, 'masks') and (r.masks is not None) and hasattr(r.masks, 'data')
    boxes = getattr(r, 'boxes', None)

    if has_masks:
        m = r.masks.data.detach().cpu().numpy()
        if m.ndim == 3:
            for i in range(m.shape[0]):
                mi = (m[i] > 0.5).astype(np.uint8)
                if mi.shape[:2] != (H, W):
                    mi = cv2.resize(mi, (W, H), interpolation=cv2.INTER_NEAREST)
                mask = np.maximum(mask, mi)
    else:
        if boxes is None or getattr(boxes, 'xyxy', None) is None:
            mask[:] = 1
        else:
            xyxy = boxes.xyxy.detach().cpu().numpy().astype(int)
            for (x1, y1, x2, y2) in xyxy:
                x1 = np.clip(x1, 0, W-1); x2 = np.clip(x2, 0, W-1)
                y1 = np.clip(y1, 0, H-1); y2 = np.clip(y2, 0, H-1)
                mask[y1:y2+1, x1:x2+1] = 1
            if mask.sum() == 0:
                mask[:, :] = 1

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), iterations=1)

    t1 = time.perf_counter()
    seg_time = t1 - t0
    if profile:
        return mask, seg_time
    else:
        return mask

# ==================== NeuFlow 前端 ====================
class NeuFlowRunner:
    def __init__(self, ckpt_path, img_w=256, img_h=144, device=DEVICE):  # 降低输入分辨率
        self.ckpt_path = ckpt_path
        self.img_w = int(img_w)
        self.img_h = int(img_h)
        self.device = torch.device(device)

        # 构建与预热（FP32）
        self.model = NeuFlow().to(self.device)
        self.model.eval()
        with torch.no_grad():
            dummy0 = torch.zeros(1, 3, self.img_h, self.img_w, device=self.device, dtype=torch.float32)
            dummy1 = torch.zeros(1, 3, self.img_h, self.img_w, device=self.device, dtype=torch.float32)
            self.model.init_bhwd(1, self.img_h, self.img_w, self.device)
            _ = self.model(dummy0, dummy1)

        # 加载权重（严格匹配）
        ckpt = torch.load(self.ckpt_path, map_location=self.device)
        state = ckpt.get('model', ckpt)
        self.model.load_state_dict(state, strict=True)

        # AMP：CUDA 则启用 FP16
        self.autocast_ctx = torch.amp.autocast(
            'cuda', enabled=(self.device.type == 'cuda'), dtype=torch.float16
        )

    @torch.no_grad()
    def infer_flow_to_original(self, img1_rgb_np, img2_rgb_np, H0, W0, profile: bool = False):
        t0 = time.perf_counter()

        t0_tensor = to_bgr_tensor_4d_from_rgb(img1_rgb_np, self.img_w, self.img_h, self.device, dtype=torch.float32)
        t1_tensor = to_bgr_tensor_4d_from_rgb(img2_rgb_np, self.img_w, self.img_h, self.device, dtype=torch.float32)

        self.model.init_bhwd(t0_tensor.shape[0], self.img_h, self.img_w, self.device)
        with self.autocast_ctx:
            preds = self.model(t0_tensor, t1_tensor)
            flow_hw = preds[-1][0].float().detach()  # [2,h,w]

        f = flow_hw.cpu().numpy().astype(np.float32)
        u, v = f[0], f[1]
        sx = np.float32(W0 / float(self.img_w))
        sy = np.float32(H0 / float(self.img_h))
        u_rs = cv2.resize(u, (W0, H0), interpolation=cv2.INTER_LINEAR).astype(np.float32) * sx
        v_rs = cv2.resize(v, (W0, H0), interpolation=cv2.INTER_LINEAR).astype(np.float32) * sy
        flow_2hw = np.stack([u_rs, v_rs], axis=0).astype(np.float32)
        flow_tensor = torch.from_numpy(flow_2hw[None, ...])  # [1,2,H0,W0]

        t1 = time.perf_counter()
        flow_time = t1 - t0

        if profile:
            return flow_tensor, flow_time
        else:
            return flow_tensor

# ==================== 速度与可视化 ====================
ALPHA = 0.60
CLIP_LOW_P = 2.0
CLIP_HIGH_P = 98.0
CMAP = cv2.COLORMAP_TURBO

def speed_map_from_flow(flow_1x2hw: torch.Tensor, dt: float, gsd: float) -> np.ndarray:
    f = flow_1x2hw[0].detach().cpu().numpy().astype(np.float32)
    u, v = f[0], f[1]
    speed = np.sqrt(u*u + v*v) * (gsd / max(dt, 1e-6))
    return speed  # (H,W), m/s

def colorize_speed(speed: np.ndarray, mask: np.ndarray) -> np.ndarray:
    H, W = speed.shape
    mask_vis = np.ones((H, W), dtype=np.uint8) if mask is None else mask.astype(np.uint8)
    speed_masked = np.where(mask_vis > 0, speed, 0.0)
    valid_vals = speed_masked[mask_vis > 0]
    vmin = np.percentile(valid_vals, CLIP_LOW_P) if valid_vals.size else 0.0
    vmax = np.percentile(valid_vals, CLIP_HIGH_P) if valid_vals.size else 1.0
    norm = (np.clip(speed_masked, vmin, vmax) - vmin) / (vmax - vmin + 1e-12)
    norm_u8 = (norm * 255.0).astype(np.uint8)
    heat_bgr = cv2.applyColorMap(norm_u8, CMAP)
    heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)
    heat_rgb[mask_vis == 0] = (0, 0, 0)
    return heat_rgb

def overlay_heat(frame_bgr: np.ndarray, heat_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    H, W = frame_bgr.shape[:2]
    if heat_rgb.shape[:2] != (H, W):
        heat_rgb = cv2.resize(heat_rgb, (W, H), interpolation=cv2.INTER_LINEAR)
    mask_vis = mask if mask is not None else np.ones((H, W), dtype=np.uint8)
    out = frame_bgr.copy()
    idx = mask_vis > 0
    out[idx] = (ALPHA * heat_rgb[idx] + (1 - ALPHA) * frame_bgr[idx]).astype(np.uint8)
    return out

def draw_stats_text(img_bgr: np.ndarray, speed: np.ndarray, mask: np.ndarray, dt: float, gsd: float):
    H, W = img_bgr.shape[:2]
    mask_vis = mask if mask is not None else np.ones((H, W), dtype=np.uint8)
    vals = speed[mask_vis > 0]
    mean_v = float(np.mean(vals)) if vals.size else 0.0
    max_v = float(np.max(vals)) if vals.size else 0.0
    txt = f"Mean: {mean_v:.3f} m/s    Max: {max_v:.3f} m/s    dt={dt:.4f}s  gsd={gsd:.5f}m/px"
    cv2.putText(img_bgr, txt, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)

# ==================== 数据源统一读取 ====================
def list_images_from_source(source: str):
    if os.path.isdir(source):
        files = []
        for ext in IMG_EXTS:
            files.extend(glob.glob(os.path.join(source, f'*{ext}')))
        return sorted(files)
    if any(ch in source for ch in ['*','?']):
        files = glob.glob(source)
        return sorted([f for f in files if os.path.splitext(f)[1] in IMG_EXTS])
    if os.path.isfile(source) and os.path.splitext(source)[1] in IMG_EXTS:
        return [source]
    return []

def is_video_file(path: str) -> bool:
    return os.path.isfile(path) and os.path.splitext(path)[1] in VID_EXTS

# ==================== 读取视频帧 ====================
def read_frame(cap, prev_bgr):
    # 从视频捕捉对象中读取一帧
    ok, frame_bgr = cap.read()
    if not ok:
        return None, None  # 如果读取失败，返回 None
    # 将 BGR 转换为 RGB
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return frame_bgr, frame_rgb

# ==================== 主流程 ====================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', required=True, help='图片目录/通配符/单图；或视频路径；或摄像头索引(整数)')
    ap.add_argument('--neuflow_ckpt', required=True, help='NeuFlow 权重路径')
    ap.add_argument('--yolo_weights', required=True, help='YOLO 分割权重 (yolov8*-seg.pt 或自训)')
    ap.add_argument('--dt', type=float, default=1/60.0, help='帧间隔(s)')
    ap.add_argument('--gsd', type=float, default=0.0128, help='GSD (m/px)')

    args = ap.parse_args()

    if not _HAS_ULTRA:
        raise RuntimeError("未检测到 ultralytics，请先:  python -m pip install ultralytics")

    # 初始化模型
    yolo = YOLO(args.yolo_weights)
    try:
        yolo.to(DEVICE)
    except Exception:
        pass

    nf = NeuFlowRunner(args.neuflow_ckpt, img_w=256, img_h=144, device=DEVICE)

    cv2.namedWindow('NeuFlow Speed Overlay', cv2.WINDOW_NORMAL)

    if is_video_file(args.source):
        cap = cv2.VideoCapture(args.source)
        if not cap.isOpened():
            raise RuntimeError(f'无法打开视频：{args.source}')
        fps = cap.get(cv2.CAP_PROP_FPS)
        dt_default = 1.0 / fps if fps else args.dt

        ok, prev_bgr = cap.read()
        if not ok:
            raise RuntimeError('读取首帧失败')
        prev_rgb = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2RGB)
        prev_mask, _ = union_mask_yolo(yolo, prev_rgb, profile=True)

        t_prev = time.perf_counter()
        dt_smooth = dt_default

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            while True:
                futures.append(executor.submit(read_frame, cap, prev_bgr))
                frame_bgr, frame_rgb = futures[-1].result()
                if frame_bgr is None:
                    break

                # 时间计算
                now = time.perf_counter()
                dt_meas = max(now - t_prev, 1e-6)
                t_prev = now
                dt_smooth = dt_default

                # 分割耗时
                mask, seg_time = union_mask_yolo(yolo, frame_rgb, profile=True)

                # 光流耗时
                flow, flow_time = nf.infer_flow_to_original(prev_rgb, frame_rgb, frame_rgb.shape[0], frame_rgb.shape[1], profile=True)

                # 后处理
                t_vis0 = time.perf_counter()
                speed = speed_map_from_flow(flow, dt_smooth, args.gsd)
                
                heat = colorize_speed(speed, mask)
                t_vis1 = time.perf_counter()
                over = overlay_heat(frame_bgr, heat, mask)
                draw_stats_text(over, speed, mask, dt_smooth, args.gsd)
                

                # FPS 打印
                total_frame = t_vis1 - t_vis0
                curr_fps = 1.0 / total_frame if total_frame > 0 else 0.0
                print(f"[Frame] FPS: {curr_fps:.1f} | seg={seg_time * 1000:.1f}ms, flow={flow_time * 1000:.1f}ms, vis={total_frame * 1000:.1f}ms")
                cv2.imshow('NeuFlow Speed Overlay', over)

                prev_rgb = frame_rgb
                prev_mask = mask
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()
        return

# 启动主流程
if __name__ == '__main__':
    main()
