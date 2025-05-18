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
import matplotlib.pyplot as plt

# 移除YOLOv5到系统路径
# sys.path.append('yolov5')

# 替换YOLOv5导入
from ultralytics import YOLO

# 导入LimitNet模型
from models.model import LimitNet, Encoder, Decoder

class AdaptiveTrackingEncoder(torch.nn.Module):
    """自适应编码器，将检测/追踪结果直接集成到编码过程中"""
    
    def __init__(self, limitnet_model, detector_model, conf_thres=0.25):
        super(AdaptiveTrackingEncoder, self).__init__()
        self.encoder = limitnet_model.encoder
        self.decoder = limitnet_model.decoder
        self.detector = detector_model
        self.conf_thres = conf_thres
        self.device = next(self.encoder.parameters()).device
        
        # 追踪相关参数
        self.tracking_boxes = []  # 追踪到的边界框列表
        self.detection_interval = 1  # 每帧都进行追踪
        self.frame_count = 0
        self.track_history = {}  # 跟踪历史 {id: [positions]}
        self.max_tracks = 30  # 历史轨迹最大长度
        self.use_bytetrack = True  # 使用ByteTrack
        
        # 区域编码的自适应参数
        self.region_weights = torch.ones((28, 28), device=self.device)  # 初始区域权重
        self.temporal_smooth_factor = 0.8  # 时间平滑因子
        self.adaptation_rate = 0.2  # 区域自适应速率
        self.max_objects = 5  # 最大跟踪目标数量
        
        # 重要性权重参数
        self.motion_weight = 0.4
        self.size_weight = 0.3
        self.conf_weight = 0.2
        self.center_weight = 0.1
        
        # 设置替换张量
        self.replace_tensor = torch.FloatTensor([0.0])[0].to(self.device)
    
    def detect_and_track_objects(self, frame):
        """使用ByteTrack检测和追踪对象"""
        # 使用track=True启用追踪
        results = self.detector.track(frame, persist=True, conf=self.conf_thres, verbose=False)
        
        tracked_objects = []
        
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            # 获取跟踪结果
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()
            track_ids = results[0].boxes.id
            
            if track_ids is not None:
                track_ids = track_ids.int().cpu().numpy().tolist()
                
                # 更新跟踪历史
                for i, track_id in enumerate(track_ids):
                    box = boxes[i]
                    conf = confidences[i]
                    
                    # 计算中心位置
                    center_x = (box[0] + box[2]) / 2
                    center_y = (box[1] + box[3]) / 2
                    center = (center_x, center_y)
                    
                    # 更新轨迹历史
                    if track_id not in self.track_history:
                        self.track_history[track_id] = []
                    
                    self.track_history[track_id].append(center)
                    
                    # 限制历史轨迹长度
                    if len(self.track_history[track_id]) > self.max_tracks:
                        self.track_history[track_id].pop(0)
                    
                    # 计算运动向量和区域大小
                    width = box[2] - box[0]
                    height = box[3] - box[1]
                    size = width * height
                    
                    # 计算重要性分数
                    importance = self._calculate_importance(track_id, center, size, conf)
                    
                    tracked_objects.append({
                        'id': track_id,
                        'box': box,
                        'confidence': conf,
                        'importance': importance,
                        'position': center,
                        'size': size
                    })
                
                # 按重要性排序
                tracked_objects.sort(key=lambda x: x['importance'], reverse=True)
                
                # 仅保留最重要的目标
                tracked_objects = tracked_objects[:self.max_objects]
        
        # 清理长时间未见的轨迹
        current_ids = {obj['id'] for obj in tracked_objects}
        ids_to_remove = []
        for track_id in self.track_history:
            if track_id not in current_ids:
                ids_to_remove.append(track_id)
        
        for track_id in ids_to_remove:
            del self.track_history[track_id]
        
        return tracked_objects
    
    def _calculate_importance(self, track_id, position, size, confidence):
        """计算目标重要性分数，增强敏感度"""
        # 1. 区域大小重要性 - 使用平方根函数增强小目标的重要性
        frame_size = 1920 * 1080  # 假设标准HD分辨率
        size_ratio = size / (frame_size * 0.25)
        size_importance = min(1.0, np.sqrt(size_ratio * 2))  # 平方根增强小值
        
        # 2. 运动重要性 - 基于最近几帧的运动量，降低阈值增加敏感度
        motion_importance = 0.0
        if track_id in self.track_history and len(self.track_history[track_id]) > 1:
            # 获取最近几帧的位置
            recent_positions = self.track_history[track_id][-min(5, len(self.track_history[track_id])):]
            
            # 计算总位移
            total_displacement = 0.0
            for i in range(1, len(recent_positions)):
                prev_pos = recent_positions[i-1]
                curr_pos = recent_positions[i]
                displacement = np.sqrt((curr_pos[0] - prev_pos[0])**2 + (curr_pos[1] - prev_pos[1])**2)
                total_displacement += displacement
            
            # 归一化，降低最大位移阈值
            motion_importance = min(1.0, total_displacement / 100)  # 从200降到100
            
            # 非线性增强
            motion_importance = motion_importance ** 0.6  # 指数<1增强低值
        
        # 3. 置信度重要性 - 使用sigmoid函数增强中等置信度
        conf_sigmoid = 1 / (1 + np.exp(-10 * (confidence - 0.5)))  # sigmoid在0.5处中心化
        conf_importance = min(1.0, conf_sigmoid * 1.2)  # 稍微放大效果
        
        # 4. 中心位置重要性 - 增强靠近边缘的目标
        frame_center = (1920/2, 1080/2)  # 假设标准HD分辨率
        distance_to_center = np.sqrt((position[0] - frame_center[0])**2 + (position[1] - frame_center[1])**2)
        max_distance = np.sqrt(frame_center[0]**2 + frame_center[1]**2)
        # 使位置重要性曲线更平坦
        center_importance = 0.6 + 0.4 * (1.0 - min(1.0, distance_to_center / max_distance))
        
        # 5. 新增目标年龄因素 - 长时间跟踪的目标更重要
        age_importance = 0.0
        if track_id in self.track_history:
            # 计算目标已被跟踪的帧数
            track_age = len(self.track_history[track_id])
            # 归一化年龄重要性，5帧后开始接近满值
            age_importance = min(1.0, track_age / 5.0)
        
        # 加权计算总重要性 - 增加年龄因素权重
        importance = (
            self.size_weight * size_importance +
            self.motion_weight * motion_importance +
            self.conf_weight * conf_importance +
            self.center_weight * center_importance +
            0.2 * age_importance  # 添加年龄权重
        )
        
        # 归一化总和为1
        importance_sum = self.size_weight + self.motion_weight + self.conf_weight + self.center_weight + 0.2
        importance = importance / importance_sum
        
        # 不要让任何目标的重要性小于0.3
        importance = max(0.3, importance)
        
        return importance
    
    def create_saliency_weights(self, frame, tracked_objects, size=(28, 28)):
        """根据多个追踪对象创建复合显著性权重图，增强对比度"""
        h, w = frame.shape[:2]
        # 设置更低的背景基础值，增加对比度
        saliency_map = np.ones((h, w), dtype=np.float32) * 0.05  # 从0.1降到0.05
        
        if not tracked_objects:
            # 如果没有检测到物体，返回均匀低权重图
            return cv2.resize(saliency_map, size)
        
        # 计算最大重要性用于归一化
        max_importance = max(obj['importance'] for obj in tracked_objects)
        
        # 为每个跟踪目标添加显著性
        for obj in tracked_objects:
            box = obj['box']
            importance = obj['importance'] / max_importance  # 归一化重要性
            
            # 增强重要性，扩大有效范围
            enhanced_importance = min(1.0, importance * 1.5)
            
            # 获取边界框坐标
            x1, y1, x2, y2 = map(int, box)
            
            # 计算中心点
            center_x, center_y = int((x1 + x2) / 2), int((y1 + y2) / 2)
            
            # 创建径向渐变显著性
            Y, X = np.ogrid[:h, :w]
            # 计算到中心的距离
            dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
            # 最大距离为边界框对角线长度的一半
            max_dist = np.sqrt((x2-x1)**2 + (y2-y1)**2) / 2
            
            # 创建高斯衰减
            mask = np.zeros((h, w), dtype=np.float32)
            valid_region = (Y >= y1) & (Y < y2) & (X >= x1) & (X < x2)
            dist_norm = np.clip(dist_from_center / max_dist, 0, 1)
            
            # 使用指数函数增加对比度
            gradient = (1.0 - dist_norm) ** 0.7  # 指数<1增强低值区域
            
            # 应用重要性作为权重，并增强小值
            mask[valid_region] = gradient[valid_region] * enhanced_importance
            
            # 更新显著性图，取最大值
            saliency_map = np.maximum(saliency_map, mask)
        
        # 应用高斯模糊以平滑边界，使用更小的核增强细节
        saliency_map = cv2.GaussianBlur(saliency_map, (5, 5), 0)  # 从(9,9)改为(5,5)
        
        # 全局对比度增强
        # 将显著性值进行指数映射，增强对比度
        saliency_map = saliency_map ** 0.8  # 指数<1会增强低值
        
        # 确保背景区域也有一个最小值
        saliency_map = np.maximum(saliency_map, 0.05)
        
        # 重新归一化到0-1
        min_val = np.min(saliency_map)
        max_val = np.max(saliency_map)
        if max_val > min_val:
            saliency_map = (saliency_map - min_val) / (max_val - min_val)
        
        # 再次增强对比度
        saliency_map = 0.05 + 0.95 * saliency_map
        
        # 调整大小为模型所需的维度
        saliency_map = cv2.resize(saliency_map, size)
        
        return saliency_map
    
    def temporal_smooth_weights(self, new_weights):
        """时间平滑区域权重，避免突变"""
        smoothed = self.temporal_smooth_factor * self.region_weights + \
                   (1 - self.temporal_smooth_factor) * torch.from_numpy(new_weights).float().to(self.device)
        return smoothed
    
    def adaptive_gradual_dropping(self, x, weights, percentage):
        """自适应渐进丢弃算法，基于权重和传输百分比"""
        temp_x = x.clone()
        
        for i in range(x.shape[0]):
            # 创建通道特定的权重
            channel_weights = weights.repeat(12, 1, 1)
            
            # 应用通道衰减因子，优先保留较低通道的数据（通常包含更多结构信息）
            for j in range(channel_weights.shape[0]):
                # 通道衰减：通道索引越大，衰减越多
                channel_decay = max(0.5, 1.0 - j * 0.05)
                channel_weights[j, :, :] = channel_weights[j, :, :] * channel_decay
            
            # 根据百分比确定丢弃阈值
            if percentage < 1.0:
                # 根据权重计算数据保留量
                flattened_weights = channel_weights.view(-1)
                threshold = torch.quantile(flattened_weights, 1 - percentage)
                
                # 确定丢弃哪些区域
                drop_mask = (channel_weights < threshold)
                
                # 应用丢弃
                temp_x[i, :, :, :] = torch.where(
                    drop_mask,
                    self.replace_tensor,
                    x[i, :, :, :]
                )
        
        return temp_x
    
    def process_frame(self, frame, frame_tensor, min_percentage=0.1, max_percentage=0.7):
        """处理单帧，集成检测/追踪与编码，动态调整压缩率"""
        self.frame_count += 1
        
        # 使用ByteTrack进行检测和追踪
        tracked_objects = self.detect_and_track_objects(frame)
        
        # 创建显著性权重图
        new_weights = self.create_saliency_weights(frame, tracked_objects)
        
        # 时间平滑权重
        self.region_weights = self.temporal_smooth_weights(new_weights)
        
        # 根据检测到的对象数量和运动情况动态调整压缩率
        adaptive_percentage = min_percentage
        
        if tracked_objects:
            # 获取最大重要性分数
            max_importance = max(obj['importance'] for obj in tracked_objects)
            
            # 计算平均运动量
            avg_motion = 0.0
            motion_scores = []
            
            for obj in tracked_objects:
                track_id = obj['id']
                if track_id in self.track_history and len(self.track_history[track_id]) > 1:
                    positions = self.track_history[track_id]
                    # 计算最近几帧的运动
                    recent_positions = positions[-min(5, len(positions)):]
                    total_motion = 0.0
                    for i in range(1, len(recent_positions)):
                        p1, p2 = recent_positions[i-1], recent_positions[i]
                        motion = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
                        total_motion += motion
                    
                    if len(recent_positions) > 1:
                        motion_scores.append(total_motion / (len(recent_positions) - 1))
            
            if motion_scores:
                avg_motion = sum(motion_scores) / len(motion_scores)
                # 归一化到0-1，降低阈值以增加敏感度
                avg_motion = min(1.0, avg_motion / 100)  # 从200改为100，增加动作敏感度
            
            # 根据目标数量、重要性和运动调整压缩率
            # 使用非线性映射增加对比度
            num_objects_factor = min(1.0, len(tracked_objects) / max(1, self.max_objects/2))  # 降低除数，增加系数
            
            # 将重要性分数进行指数增强，增加对比度
            enhanced_importance = max_importance ** 0.7  # 指数<1会增强低值
            
            # 增强运动因子的影响
            enhanced_motion = avg_motion ** 0.6
            
            # 可以设置最小保底值
            base_percentage = min_percentage + (max_percentage - min_percentage) * 0.3
            
            # 综合计算自适应压缩率
            adaptive_percentage = base_percentage + (max_percentage - base_percentage) * (
                0.4 * num_objects_factor + 
                0.4 * enhanced_importance + 
                0.2 * enhanced_motion
            )
            
            # 添加帧序号因素（如果需要在场景转换处提高质量）
            scene_change_boost = 0.0
            if self.frame_count % 30 == 0:  # 每30帧给一个质量提升
                scene_change_boost = 0.1  # 增加10%的压缩率
            
            adaptive_percentage = min(max_percentage, adaptive_percentage + scene_change_boost)
            
            # 确保至少有一个最小值
            adaptive_percentage = max(min_percentage + 0.1, adaptive_percentage)
        
        # 编码图像
        with torch.no_grad():
            encoded = self.encoder(frame_tensor)
            
            # 自适应渐进丢弃
            encoded = self.adaptive_gradual_dropping(encoded, self.region_weights, adaptive_percentage)
            
            # 解码
            output = self.decoder(encoded)
        
        # 获取主要对象的边界框（用于可视化）
        main_bbox = None
        if tracked_objects:
            main_obj = tracked_objects[0]
            box = main_obj['box']
            x1, y1, x2, y2 = map(int, box)
            main_bbox = (x1, y1, x2-x1, y2-y1)
        
        return output, self.region_weights.cpu().numpy(), main_bbox, tracked_objects, adaptive_percentage
    
    def create_diagnostic_image(self, frame, output, weights, bbox=None, percentage=0.5, tracked_objects=None):
        """创建诊断图像，显示原始帧、重建帧和权重图"""
        # 将输出张量转换为NumPy数组
        output_np = output[0].cpu().detach().numpy().transpose(1, 2, 0)
        output_np = np.clip(output_np * 255, 0, 255).astype(np.uint8)
        
        # 将输出调整为与原始帧相同的尺寸
        output_bgr = cv2.cvtColor(output_np, cv2.COLOR_RGB2BGR)
        output_resized = cv2.resize(output_bgr, (frame.shape[1], frame.shape[0]))
        
        # 创建权重热力图
        weights_vis = (weights * 255).astype(np.uint8)
        weights_vis = cv2.resize(weights_vis, (frame.shape[1], frame.shape[0]))
        weights_vis = cv2.applyColorMap(weights_vis, cv2.COLORMAP_JET)
        
        # 在原始帧上绘制所有追踪目标
        result_frame = frame.copy()
        
        if tracked_objects:
            for obj in tracked_objects:
                box = obj['box']
                importance = obj['importance']
                track_id = obj['id']
                
                x1, y1, x2, y2 = map(int, box)
                
                # 根据重要性使用不同颜色
                # 绿色(低) -> 黄色 -> 红色(高)
                color = (0, 
                      int(255 * (1 - importance)), 
                      int(255 * importance))
                
                # 绘制边界框
                cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, 2)
                
                # 添加ID和重要性信息
                label = f"ID:{track_id} Imp:{importance:.2f}"
                cv2.putText(result_frame, label, (x1, y1 - 10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # 绘制轨迹
                if track_id in self.track_history and len(self.track_history[track_id]) > 1:
                    positions = self.track_history[track_id]
                    for i in range(1, len(positions)):
                        p1 = tuple(map(int, positions[i-1]))
                        p2 = tuple(map(int, positions[i]))
                        cv2.line(result_frame, p1, p2, color, 2)
        
        # 创建组合图像：原始帧、重建帧和权重图
        h, w = frame.shape[:2]
        combined = np.zeros((h, w*3, 3), dtype=np.uint8)
        combined[:, :w] = result_frame
        combined[:, w:2*w] = weights_vis
        combined[:, 2*w:] = output_resized
        
        # 添加标题
        cv2.putText(combined, "Input + Tracking", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(combined, "Weights", (w+10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(combined, f"Output ({percentage*100:.0f}%)", (2*w+10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        return combined

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='自适应检测-追踪的LimitNet渐进视频处理')
    
    # LimitNet相关参数
    parser.add_argument('--model', type=str, required=True, choices=['cifar', 'imagenet'], help='LimitNet模型类型')
    parser.add_argument('--model_path', type=str, required=True, help='LimitNet模型路径')
    
    # 视频处理参数
    parser.add_argument('--input_video', type=str, required=True, help='输入视频文件路径')
    parser.add_argument('--output_video', type=str, required=True, help='输出视频文件路径')
    parser.add_argument('--output_folder', type=str, default='./adaptive_frames/', help='输出帧保存文件夹')
    parser.add_argument('--frame_rate', type=int, default=30, help='输出视频帧率')
    
    # 检测和传输参数
    parser.add_argument('--detector_weights', type=str, default='yolov5s.pt', help='YOLOv5检测器权重路径')
    parser.add_argument('--confidence', type=float, default=0.25, help='检测置信度阈值')
    parser.add_argument('--min_percentage', type=float, default=0.1, help='最小保留的潜在变量百分比')
    parser.add_argument('--max_percentage', type=float, default=0.7, help='最大保留的潜在变量百分比')
    
    # 追踪和适应性参数
    parser.add_argument('--max_objects', type=int, default=5, help='最大跟踪目标数量')
    parser.add_argument('--motion_weight', type=float, default=0.5, help='运动重要性权重')
    parser.add_argument('--size_weight', type=float, default=0.3, help='目标大小重要性权重')
    parser.add_argument('--center_weight', type=float, default=0.2, help='中心位置重要性权重')
    parser.add_argument('--background_quality', type=float, default=0.3, help='背景区域保留比例')
    parser.add_argument('--i_frame_interval', type=int, default=30, help='I帧间隔')
    
    # 视频编码参数
    parser.add_argument('--quality', type=int, default=95, help='输出视频质量')
    
    # 可视化和评估参数
    parser.add_argument('--visualization', action='store_true', help='是否生成可视化诊断图像')
    parser.add_argument('--save_metrics', action='store_true', help='是否保存性能指标')
    
    return parser.parse_args()

def main():
    """主函数"""
    args = parse_args()
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载LimitNet模型
    print(f"加载LimitNet模型: {args.model_path}")
    limitnet = LimitNet(args.model)
    limitnet = torch.load(args.model_path)
    limitnet.eval().to(device)
    
    # 加载检测器模型
    print(f"加载YOLO模型: {args.detector_weights}")
    detector = YOLO(args.detector_weights)
    
    # 创建自适应编码器
    adaptive_encoder = AdaptiveTrackingEncoder(limitnet, detector, args.confidence)
    adaptive_encoder.max_objects = args.max_objects
    
    # 设置重要性权重
    adaptive_encoder.motion_weight = args.motion_weight
    adaptive_encoder.size_weight = args.size_weight 
    adaptive_encoder.center_weight = args.center_weight
    adaptive_encoder.conf_weight = 1.0 - (args.motion_weight + args.size_weight + args.center_weight)
    
    # 打开视频
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
    frame_count = 0
    processed_frames = []
    metrics = {'psnr': [], 'bbox_info': [], 'compression_ratios': []}
    
    # 创建输出视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    if args.visualization:
        out = cv2.VideoWriter(args.output_video, fourcc, args.frame_rate, (width*3, height))
    else:
        out = cv2.VideoWriter(args.output_video, fourcc, args.frame_rate, (width, height))
    
    print(f"开始处理视频，总帧数: {total_frames}")
    pbar = tqdm(total=total_frames)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 将图像转换为RGB并进行归一化处理
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        input_tensor = transform(pil_img).unsqueeze(0).to(device)
        
        # 使用自适应编码器处理帧，使用min_percentage和max_percentage
        output, weights, bbox, tracked_objects, adaptive_percentage = adaptive_encoder.process_frame(
            frame, input_tensor, args.min_percentage, args.max_percentage
        )
        
        # 创建可视化图像
        if args.visualization:
            diagnostic_img = adaptive_encoder.create_diagnostic_image(
                frame.copy(), output, weights, bbox, adaptive_percentage, tracked_objects
            )
            output_path = f"{args.output_folder}/frame_{frame_count:05d}.jpg"
            cv2.imwrite(output_path, diagnostic_img)
            processed_frames.append(output_path)
            out.write(diagnostic_img)
        else:
            # 只输出重建帧
            output_np = output[0].cpu().detach().numpy().transpose(1, 2, 0)
            output_np = np.clip(output_np * 255, 0, 255).astype(np.uint8)
            output_bgr = cv2.cvtColor(output_np, cv2.COLOR_RGB2BGR)
            output_resized = cv2.resize(output_bgr, (width, height))
            
            # 添加压缩率信息到帧
            cv2.putText(output_resized, f"压缩率: {adaptive_percentage:.2f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            output_path = f"{args.output_folder}/frame_{frame_count:05d}.jpg"
            cv2.imwrite(output_path, output_resized)
            processed_frames.append(output_path)
            out.write(output_resized)
        
        # 保存指标（如果需要）
        if args.save_metrics:
            # 计算PSNR
            mse = np.mean((frame_rgb / 255.0 - output_np / 255.0) ** 2)
            psnr = 20 * np.log10(1.0 / np.sqrt(mse)) if mse > 0 else 100
            metrics['psnr'].append(psnr)
            
            # 保存边界框信息
            metrics['bbox_info'].append(bbox)
            
            # 保存压缩率
            metrics['compression_ratios'].append(adaptive_percentage)
        
        # 更新进度
        frame_count += 1
        pbar.update(1)
    
    # 清理资源
    pbar.close()
    cap.release()
    out.release()
    
    # 保存指标（如果需要）
    if args.save_metrics:
        np.save(f"{args.output_folder}/metrics.npy", metrics)
        
        # 创建图表目录
        metrics_dir = f"{args.output_folder}/metrics"
        os.makedirs(metrics_dir, exist_ok=True)
        
        # 绘制PSNR图
        plt.figure(figsize=(10, 5))
        plt.plot(metrics['psnr'])
        plt.title('PSNR vs Frame Number')
        plt.xlabel('Frame Number')
        plt.ylabel('PSNR (dB)')
        plt.savefig(f"{metrics_dir}/psnr_plot.png")
        plt.close()
        
        # 绘制压缩率曲线
        plt.figure(figsize=(10, 5))
        plt.plot(metrics['compression_ratios'])
        plt.title('动态压缩率变化曲线')
        plt.xlabel('帧序号')
        plt.ylabel('压缩率 (百分比)')
        plt.ylim([0, 1])
        plt.grid(True)
        plt.savefig(f"{metrics_dir}/compression_ratio_plot.png")
        plt.close()
        
        # 生成压缩率统计报告
        if metrics['compression_ratios']:
            avg_ratio = sum(metrics['compression_ratios']) / len(metrics['compression_ratios'])
            min_ratio = min(metrics['compression_ratios'])
            max_ratio = max(metrics['compression_ratios'])
            
            with open(f"{metrics_dir}/compression_report.txt", 'w') as f:
                f.write(f"压缩率统计报告\n")
                f.write(f"=================\n")
                f.write(f"平均压缩率: {avg_ratio:.4f}\n")
                f.write(f"最小压缩率: {min_ratio:.4f}\n")
                f.write(f"最大压缩率: {max_ratio:.4f}\n")
                f.write(f"压缩率标准差: {np.std(metrics['compression_ratios']):.4f}\n")
                f.write(f"平均PSNR: {sum(metrics['psnr'])/len(metrics['psnr']):.2f} dB\n")
    
    print(f"处理完成，输出视频保存为: {args.output_video}")

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

if __name__ == "__main__":
    main() 