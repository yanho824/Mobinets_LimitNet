import argparse
import cv2
import os
import sys
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import time
from tqdm import tqdm

# 移除YOLOv5路径添加
# sys.path.append('yolov5')

# 替换YOLOv5导入为ultralytics
from ultralytics import YOLO

# 导入LimitNet模型
from models.model import LimitNet

def parse_args():
    parser = argparse.ArgumentParser(description='基于检测-追踪的LimitNet渐进视频处理')
    
    # LimitNet相关参数
    parser.add_argument('--model', type=str, required=True, choices=['cifar', 'imagenet'], help='LimitNet模型类型')
    parser.add_argument('--model_path', type=str, required=True, help='LimitNet模型路径')
    
    # 视频处理参数
    parser.add_argument('--input_video', type=str, required=True, help='输入视频文件路径')
    parser.add_argument('--output_video', type=str, required=True, help='输出视频文件路径')
    parser.add_argument('--output_folder', type=str, default='./tracked_frames/', help='输出帧保存文件夹')
    parser.add_argument('--frame_rate', type=int, default=30, help='输出视频帧率')
    
    # 检测和传输参数
    parser.add_argument('--detector_weights', type=str, default='yolov5s.pt', help='YOLOv5检测器权重路径')
    parser.add_argument('--confidence', type=float, default=0.25, help='检测置信度阈值')
    parser.add_argument('--percentage', type=float, required=True, help='保留的潜在变量百分比')
    
    # 新增优化参数
    parser.add_argument('--max_objects', type=int, default=3, help='最大跟踪目标数量')
    parser.add_argument('--tracker_type', type=str, default='ByteTrack', choices=['CSRT', 'KCF', 'MOSSE', 'ByteTrack'], help='跟踪器类型')
    parser.add_argument('--saliency_blur', type=int, default=5, help='显著性图高斯模糊核大小')
    parser.add_argument('--detection_interval', type=int, default=5, help='检测间隔帧数')
    parser.add_argument('--background_quality', type=float, default=0.3, help='背景区域保留比例')
    parser.add_argument('--quality', type=int, default=95, help='输出视频质量')
    
    return parser.parse_args()

def init_tracker(tracker_type='ByteTrack'):
    """初始化追踪器"""
    if tracker_type == 'ByteTrack':
        # ByteTrack由YOLO集成
        return None
    elif tracker_type == 'CSRT':
        tracker = cv2.TrackerCSRT_create()
    elif tracker_type == 'KCF':
        tracker = cv2.TrackerKCF_create()
    elif tracker_type == 'MOSSE':
        tracker = cv2.legacy.TrackerMOSSE_create()
    else:
        tracker = cv2.TrackerCSRT_create()  # 默认使用CSRT
    return tracker

def create_saliency_map(frame, boxes, size=(28, 28), blur_size=5, bg_quality=0.3):
    """创建基于多目标的渐变显著性图"""
    h, w = frame.shape[:2]
    
    # 创建基础背景显著性值
    saliency_map = np.ones((h, w), dtype=np.float32) * bg_quality
    
    if boxes is None or len(boxes) == 0:
        # 如果没有检测到物体，返回均匀显著性图
        return cv2.resize(saliency_map, size)
    
    # 为每个检测到的目标添加显著性
    for box in boxes:
        # 获取边界框坐标
        x1, y1, x2, y2 = box[:4]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # 计算边界框中心
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
        
        # 创建径向渐变显著性，从中心向边缘递减
        Y, X = np.ogrid[:h, :w]
        # 计算到中心的距离，并归一化
        dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
        # 获取边界框对角线长度作为最大距离
        max_dist = np.sqrt((x2-x1)**2 + (y2-y1)**2) / 2
        
        # 在边界框内创建渐变效果
        mask = np.zeros((h, w), dtype=np.float32)
        mask[y1:y2, x1:x2] = 1.0 - np.clip(dist_from_center[y1:y2, x1:x2] / max_dist, 0, 1)
        
        # 更新显著性图，取最大值
        saliency_map = np.maximum(saliency_map, mask)
    
    # 应用高斯模糊以平滑显著性边界
    if blur_size > 0:
        saliency_map = cv2.GaussianBlur(saliency_map, (blur_size, blur_size), 0)
    
    # 调整大小为模型所需的维度
    saliency_map = cv2.resize(saliency_map, size)
    
    return saliency_map

def process_video_with_tracking(args):
    """使用检测和追踪处理视频"""
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载检测模型并设置为跟踪模式
    print(f"加载YOLO模型: {args.detector_weights}")
    if args.tracker_type == 'ByteTrack':
        detector_model = YOLO(args.detector_weights)
    else:
        detector_model = YOLO(args.detector_weights)
    
    # 加载LimitNet模型
    print(f"加载LimitNet模型: {args.model_path}")
    limit_model = LimitNet(args.model)
    limit_model = torch.load(args.model_path)
    limit_model.eval().to(device)
    limit_model.p = args.percentage
    limit_model.replace_tensor = torch.FloatTensor([0.0])[0].to(device)
    
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
    
    # 创建输出目录
    os.makedirs(args.output_folder, exist_ok=True)
    
    # 设置图像转换
    transform = transforms.Compose([
        transforms.Resize((224, 224), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 初始化变量
    trackers = []
    tracking_boxes = []
    frame_count = 0
    processed_frames = []
    
    # 根据追踪器类型设置不同的处理逻辑
    use_bytetrack = (args.tracker_type == 'ByteTrack')
    track_history = {}  # 用于ByteTrack
    
    print(f"开始处理视频，总帧数: {total_frames}")
    
    pbar = tqdm(total=total_frames)
    
    # 初始化I帧列表，每隔一定帧数设置I帧
    i_frames = {0}  # 第一帧必须是I帧
    gop_size = 30  # 固定GOP大小
    for i in range(gop_size, total_frames, gop_size):
        i_frames.add(i)
    
    # 视频直接写入器 - 不再保存单独的帧文件
    output_video_path = args.output_video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        original_frame = frame.copy()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 是否执行检测
        do_detection = frame_count % args.detection_interval == 0 or frame_count in i_frames
        
        if use_bytetrack:
            # 使用ByteTrack进行检测和追踪
            if do_detection:
                # 使用跟踪模式，track=True
                results = detector_model.track(frame, persist=True, conf=args.confidence, verbose=False)
                
                if results[0].boxes is not None and len(results[0].boxes) > 0:
                    # 获取跟踪结果
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    track_ids = results[0].boxes.id
                    
                    if track_ids is not None:
                        track_ids = track_ids.int().cpu().numpy().tolist()
                        
                        # 更新跟踪历史
                        for i, box_id in enumerate(track_ids):
                            if box_id not in track_history:
                                track_history[box_id] = []
                            
                            # 记录边界框中心位置
                            center = ((boxes[i][0] + boxes[i][2]) / 2, (boxes[i][1] + boxes[i][3]) / 2)
                            track_history[box_id].append(center)
                            
                            # 限制历史轨迹长度
                            if len(track_history[box_id]) > 30:
                                track_history[box_id].pop(0)
                    
                    # 仅保留最大目标数量的边界框
                    if len(boxes) > args.max_objects:
                        # 按面积排序
                        areas = [(box[2]-box[0])*(box[3]-box[1]) for box in boxes]
                        largest_indices = np.argsort(areas)[-args.max_objects:]
                        boxes = boxes[largest_indices]
                    
                    tracking_boxes = boxes
                else:
                    tracking_boxes = []
            
        else:
            # 使用OpenCV跟踪器
            if do_detection:
                # 执行检测
                results = detector_model(frame, conf=args.confidence, verbose=False)
                
                # 重置跟踪器
                trackers = []
                tracking_boxes = []
                
                if results[0].boxes is not None and len(results[0].boxes) > 0:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    
                    # 仅跟踪前N个最大目标
                    if len(boxes) > args.max_objects:
                        areas = [(box[2]-box[0])*(box[3]-box[1]) for box in boxes]
                        largest_indices = np.argsort(areas)[-args.max_objects:]
                        boxes = boxes[largest_indices]
                    
                    # 为每个目标创建一个跟踪器
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box)
                        tracker = init_tracker(args.tracker_type)
                        track_box = (x1, y1, x2-x1, y2-y1)
                        if tracker.init(frame, track_box):
                            trackers.append(tracker)
                            tracking_boxes.append(box)
            else:
                # 更新现有跟踪器
                updated_trackers = []
                updated_boxes = []
                
                for i, tracker in enumerate(trackers):
                    success, box = tracker.update(frame)
                    if success:
                        x, y, w, h = box
                        updated_box = np.array([x, y, x+w, y+h])
                        updated_trackers.append(tracker)
                        updated_boxes.append(updated_box)
                
                trackers = updated_trackers
                tracking_boxes = updated_boxes
        
        # 创建显著性图
        saliency_map = create_saliency_map(
            frame, 
            tracking_boxes, 
            size=(28, 28), 
            blur_size=args.saliency_blur,
            bg_quality=args.background_quality
        )
        
        saliency_tensor = torch.from_numpy(saliency_map).unsqueeze(0).unsqueeze(0).float().to(device)
        
        # 处理帧使用LimitNet
        pil_img = Image.fromarray(frame_rgb)
        input_tensor = transform(pil_img).unsqueeze(0).to(device)
        
        # 使用自定义显著性图代替SalDecoder输出
        with torch.no_grad():
            # 编码图像
            encoded = limit_model.encoder(input_tensor)
            
            # 量化显著性图
            saliency_tensor = limit_model.sal_quantization_and_dequantization(saliency_tensor)
            
            # 使用显著性图进行渐进式数据丢弃
            encoded = limit_model.gradual_dropping(encoded, saliency_tensor)
            
            # 解码得到重建图像
            output = limit_model.decoder(encoded)
        
        # 将输出转换回BGR格式用于显示
        output_np = output[0].cpu().detach().numpy().transpose(1, 2, 0)
        output_np = np.clip(output_np * 255, 0, 255).astype(np.uint8)
        output_bgr = cv2.cvtColor(output_np, cv2.COLOR_RGB2BGR)
        
        # 调整大小以匹配原始帧
        output_resized = cv2.resize(output_bgr, (width, height))
        
        # 可视化显著性图
        saliency_vis = (saliency_map * 255).astype(np.uint8)
        saliency_vis = cv2.resize(saliency_vis, (width // 4, height // 4))
        saliency_vis = cv2.applyColorMap(saliency_vis, cv2.COLORMAP_JET)
        
        # 添加跟踪结果到原始帧 (仅用于可视化)
        result_frame = output_resized.copy()
        
        # 可视化跟踪结果
        for box in tracking_boxes:
            x1, y1, x2, y2 = map(int, box[:4])
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # 可视化跟踪轨迹 (仅ByteTrack)
        if use_bytetrack and results[0].boxes.id is not None:
            for track_id, track_points in track_history.items():
                if len(track_points) > 1:
                    for i in range(1, len(track_points)):
                        p1 = tuple(map(int, track_points[i-1]))
                        p2 = tuple(map(int, track_points[i]))
                        cv2.line(result_frame, p1, p2, (0, 255, 255), 2)
        
        # 添加显著性图到右上角
        h_sal, w_sal = saliency_vis.shape[:2]
        result_frame[0:h_sal, width-w_sal:width, :] = saliency_vis
        
        # 添加信息文本
        cv2.putText(result_frame, f"LimitNet ({args.percentage*100:.0f}%)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(result_frame, f"Objects: {len(tracking_boxes)}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(result_frame, f"Frame: {frame_count}", (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 直接写入视频
        out.write(result_frame)
        
        # 更新帧计数器
        frame_count += 1
        pbar.update(1)
    
    pbar.close()
    cap.release()
    out.release()
    
    print(f"处理完成，输出视频保存为: {output_video_path}")
    
    # 优化最终视频文件大小和质量
    if args.quality < 100:
        optimize_video(output_video_path, args.quality)

def optimize_video(video_path, quality=95):
    """使用FFmpeg优化视频大小和质量"""
    try:
        # 检查FFmpeg是否可用
        import shutil
        if shutil.which('ffmpeg') is None:
            print("警告: FFmpeg未安装，跳过视频优化。请安装FFmpeg: apt-get update && apt-get install -y ffmpeg")
            return
            
        temp_path = video_path.replace('.mp4', '_temp.mp4')
        
        # 使用FFmpeg重新编码视频以优化大小
        cmd = f"ffmpeg -i {video_path} -vcodec libx264 -crf {100-quality} {temp_path} -y"
        print(f"优化视频: {cmd}")
        os.system(cmd)
        
        # 替换原始文件
        if os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
            os.remove(video_path)
            os.rename(temp_path, video_path)
            print(f"视频已优化: {video_path}")
        else:
            print("视频优化失败，保留原始文件")
    except Exception as e:
        print(f"视频优化过程中出错: {str(e)}")

def sal_quantization_and_dequantization(self, data):
    """对显著性图进行量化和反量化"""
    min_ = torch.min(data)
    max_ = torch.max(data)
    data = (data - min_) / (max_ - min_)
    data = data * 255
    data = data.type(dtype=torch.uint8)

    data = data / 8
    data = data.type(dtype=torch.uint8)
    data = data * 8

    data = data / 255.0
    data = data * (max_ - min_) + min_
    return data

# 将量化和反量化方法添加到LimitNet类
LimitNet.sal_quantization_and_dequantization = sal_quantization_and_dequantization

def main():
    """主函数"""
    args = parse_args()
    process_video_with_tracking(args)

if __name__ == "__main__":
    main() 