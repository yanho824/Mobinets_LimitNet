import argparse
import cv2
import os
import sys
import torch
import numpy as np
from PIL import Image

# 添加YOLOv5到系统路径
sys.path.append('.')  # 确保可以找到yolov5目录

# 导入YOLOv5模型
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import check_img_size, non_max_suppression, scale_boxes
from yolov5.utils.augmentations import letterbox
from yolov5.utils.torch_utils import select_device
from yolov5.utils.plots import Annotator, colors

def parse_args():
    parser = argparse.ArgumentParser(description='目标检测和追踪测试工具')
    parser.add_argument('--input_video', type=str, required=True, help='输入视频文件路径')
    parser.add_argument('--output_video', type=str, required=True, help='输出视频文件路径')
    parser.add_argument('--detector_weights', type=str, default='yolov5/yolov5s.pt', help='YOLOv5检测器权重路径')
    parser.add_argument('--confidence', type=float, default=0.25, help='检测置信度阈值')
    parser.add_argument('--extract_frames', action='store_true', help='是否提取视频帧')
    parser.add_argument('--frames_folder', type=str, default='./extracted_frames/', help='帧提取文件夹')
    return parser.parse_args()

def init_tracker(tracker_type='CSRT'):
    """初始化追踪器"""
    if tracker_type == 'CSRT':
        tracker = cv2.TrackerCSRT_create()
    elif tracker_type == 'KCF':
        tracker = cv2.TrackerKCF_create()
    elif tracker_type == 'MOSSE':
        tracker = cv2.legacy.TrackerMOSSE_create()
    else:
        tracker = cv2.TrackerCSRT_create()  # 默认使用CSRT
    return tracker

def create_saliency_map(frame, bbox, size=(28, 28)):
    """根据边界框创建二值显著性图"""
    h, w = frame.shape[:2]
    saliency_map = np.zeros((h, w), dtype=np.float32)
    
    # 边界框坐标
    x, y, width, height = bbox
    x, y, width, height = int(x), int(y), int(width), int(height)
    
    # 在边界框内填充1.0，表示高显著性区域
    saliency_map[y:y+height, x:x+width] = 1.0
    
    # 调整大小为模型所需的维度
    saliency_map = cv2.resize(saliency_map, size)
    
    return saliency_map

def main():
    args = parse_args()
    
    # 设置设备
    device = select_device('')
    
    # 加载YOLOv5模型
    print(f"加载YOLO模型: {args.detector_weights}")
    model = DetectMultiBackend(args.detector_weights, device=device)
    stride, names = model.stride, model.names
    imgsz = check_img_size((640, 640), s=stride)  # 检查图像大小
    
    # 打开视频文件
    cap = cv2.VideoCapture(args.input_video)
    if not cap.isOpened():
        print(f"无法打开视频: {args.input_video}")
        return
    
    # 获取视频属性
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 创建输出视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output_video, fourcc, fps, (width, height))
    
    # 创建帧提取文件夹（如果需要）
    if args.extract_frames:
        os.makedirs(args.frames_folder, exist_ok=True)
    
    # 初始化变量
    tracker = None
    tracking_bbox = None
    detection_interval = 10  # 每隔多少帧进行一次检测
    frame_count = 0
    tracking_success = False
    
    print(f"开始处理视频，总帧数: {total_frames}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        original_frame = frame.copy()
        
        # 每隔几帧执行一次检测，或者在追踪失败时
        if frame_count % detection_interval == 0 or not tracking_success:
            # 预处理图像
            img = letterbox(frame, imgsz, stride=stride, auto=True)[0]
            img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img).to(device)
            img = img.float() / 255.0  # 0 - 255 to 0.0 - 1.0
            if len(img.shape) == 3:
                img = img[None]  # 扩展批次维度
            
            # 执行检测
            pred = model(img, augment=False, visualize=False)
            pred = non_max_suppression(pred, args.confidence, 0.45, classes=None, max_det=1000)
            
            # 处理结果
            for i, det in enumerate(pred):  # 每张图像
                if len(det):
                    # 缩放边界框从img_size到图像大小
                    det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], frame.shape).round()
                    
                    # 获取置信度最高的检测结果
                    best_det = det[det[:, 4].argmax()]
                    
                    # 转换为跟踪器所需格式 [x, y, width, height]
                    x1, y1, x2, y2 = best_det[:4].cpu().numpy()
                    tracking_bbox = (int(x1), int(y1), int(x2-x1), int(y2-y1))
                    
                    # 创建新的追踪器并初始化
                    tracker = init_tracker()
                    tracking_success = tracker.init(frame, tracking_bbox)
                    
                    # 在图像上绘制检测框
                    cls = int(best_det[5])
                    conf = float(best_det[4])
                    label = f"{names[cls]} {conf:.2f}"
                    
                    # 使用Annotator绘制边界框
                    annotator = Annotator(frame, line_width=2, example=str(names))
                    annotator.box_label([x1, y1, x2, y2], label, color=colors(cls, True))
                    
                    # 创建显著性图
                    saliency_map = create_saliency_map(frame, tracking_bbox)
                    
                    # 可视化显著性图
                    saliency_vis = (saliency_map * 255).astype(np.uint8)
                    saliency_vis = cv2.resize(saliency_vis, (width // 4, height // 4))
                    saliency_vis = cv2.applyColorMap(saliency_vis, cv2.COLORMAP_JET)
                    
                    # 将显著性图放在视频的右上角
                    frame[0:saliency_vis.shape[0], width-saliency_vis.shape[1]:width] = saliency_vis
        
        else:
            # 使用追踪器更新目标位置
            if tracker is not None:
                tracking_success, bbox = tracker.update(frame)
                
                if tracking_success:
                    # 绘制追踪框
                    x, y, w, h = [int(v) for v in bbox]
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
                    
                    # 创建显著性图
                    saliency_map = create_saliency_map(frame, bbox)
                    
                    # 可视化显著性图
                    saliency_vis = (saliency_map * 255).astype(np.uint8)
                    saliency_vis = cv2.resize(saliency_vis, (width // 4, height // 4))
                    saliency_vis = cv2.applyColorMap(saliency_vis, cv2.COLORMAP_JET)
                    
                    # 将显著性图放在视频的右上角
                    frame[0:saliency_vis.shape[0], width-saliency_vis.shape[1]:width] = saliency_vis
        
        # 添加帧计数器
        cv2.putText(frame, f"Frame: {frame_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # 保存处理后的帧
        if args.extract_frames:
            cv2.imwrite(f"{args.frames_folder}/frame_{frame_count:05d}.jpg", frame)
        
        # 写入输出视频
        out.write(frame)
        
        # 更新帧计数器
        frame_count += 1
        if frame_count % 50 == 0:
            print(f"已处理 {frame_count}/{total_frames} 帧")
    
    # 释放资源
    cap.release()
    out.release()
    print(f"处理完成，输出视频保存为: {args.output_video}")

if __name__ == "__main__":
    main()
