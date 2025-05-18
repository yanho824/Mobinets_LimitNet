#!/usr/bin/env python3

import os
import argparse
import time
import cv2

def parse_args():
    parser = argparse.ArgumentParser(description='测试自适应编码方法')
    
    parser.add_argument('--input_video', type=str, default='video.mp4', help='输入视频文件路径')
    parser.add_argument('--model', type=str, default='imagenet', choices=['cifar', 'imagenet'], help='LimitNet模型类型')
    parser.add_argument('--model_path', type=str, default='LimitNet-ImageNet', help='LimitNet模型路径')
    parser.add_argument('--output_folder', type=str, default='./adaptive_results/', help='输出文件夹')
    
    # 自适应编码参数
    parser.add_argument('--max_percentage', type=float, default=0.7, help='最大保留的潜在变量百分比')
    parser.add_argument('--min_percentage', type=float, default=0.1, help='最小保留的潜在变量百分比')
    parser.add_argument('--detector_weights', type=str, default='yolov5su.pt', help='检测器权重')
    
    # 追踪参数
    parser.add_argument('--max_objects', type=int, default=5, help='最大跟踪目标数量')
    parser.add_argument('--motion_weight', type=float, default=0.4, help='运动重要性权重')
    parser.add_argument('--size_weight', type=float, default=0.3, help='目标大小重要性权重')
    parser.add_argument('--center_weight', type=float, default=0.1, help='中心位置重要性权重')
    
    # 可视化和评估参数
    parser.add_argument('--visualization', action='store_true', help='是否生成可视化诊断图像')
    parser.add_argument('--save_metrics', action='store_true', help='是否保存性能指标')
    parser.add_argument('--quality', type=int, default=95, help='输出视频质量')
    
    # 比较参数
    parser.add_argument('--compare_with_static', action='store_true', help='是否与静态方法比较')
    parser.add_argument('--static_percentages', type=str, default='0.1,0.3,0.5,0.7', help='静态百分比,用逗号分隔')
    parser.add_argument('--plot_comparison', action='store_true', help='是否绘制比较图表')
    
    return parser.parse_args()

def calculate_metrics(video_path):
    """计算视频文件的大小和其他指标"""
    file_size = os.path.getsize(video_path) / (1024 * 1024)  # MB
    
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = frame_count / fps if fps > 0 else 0
    cap.release()
    
    return {
        'size_mb': file_size,
        'frames': frame_count,
        'fps': fps,
        'duration': duration,
        'bitrate_kbps': (file_size * 8 * 1024) / duration if duration > 0 else 0
    }

def generate_comparison_plots(output_folder, adaptive_metrics, static_results):
    """生成比较图表"""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        # 创建图表目录
        plots_dir = f"{output_folder}/plots"
        os.makedirs(plots_dir, exist_ok=True)
        
        # 提取数据
        methods = ['自适应'] + [f'静态 {r["percentage"]:.1f}' for r in static_results]
        sizes = [adaptive_metrics['size_mb']] + [r['size_mb'] for r in static_results]
        bitrates = [adaptive_metrics['bitrate_kbps']] + [r['bitrate_kbps'] for r in static_results]
        times = [adaptive_metrics['process_time']] + [r['process_time'] for r in static_results]
        
        # 颜色设置
        colors = ['red'] + ['blue'] * len(static_results)
        
        # 文件大小比较图
        plt.figure(figsize=(10, 6))
        bars = plt.bar(methods, sizes, color=colors)
        plt.title('文件大小比较 (MB)', fontsize=14)
        plt.ylabel('大小 (MB)', fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.xticks(rotation=45)
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f"{plots_dir}/size_comparison.png")
        plt.close()
        
        # 比特率比较图
        plt.figure(figsize=(10, 6))
        bars = plt.bar(methods, bitrates, color=colors)
        plt.title('比特率比较 (kbps)', fontsize=14)
        plt.ylabel('比特率 (kbps)', fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.xticks(rotation=45)
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f"{plots_dir}/bitrate_comparison.png")
        plt.close()
        
        # 处理时间比较图
        plt.figure(figsize=(10, 6))
        bars = plt.bar(methods, times, color=colors)
        plt.title('处理时间比较 (秒)', fontsize=14)
        plt.ylabel('时间 (秒)', fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.xticks(rotation=45)
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f"{plots_dir}/time_comparison.png")
        plt.close()
        
        print(f"比较图表已保存至: {plots_dir}/")
        return True
    except Exception as e:
        print(f"生成图表时出错: {str(e)}")
        return False

def main():
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_folder, exist_ok=True)
    
    # 运行自适应编码
    print("\n--- 运行自适应编码 ---")
    
    adaptive_output = f"{args.output_folder}/adaptive_encoded.mp4"
    adaptive_frames_dir = f"{args.output_folder}/adaptive_frames"
    
    start_time = time.time()
    
    # 构建命令
    cmd = [
        "python", "adaptive_tracking_encode.py",
        "--model", args.model,
        "--model_path", args.model_path,
        "--input_video", args.input_video,
        "--output_video", adaptive_output,
        "--output_folder", adaptive_frames_dir,
        "--detector_weights", args.detector_weights,
        "--max_percentage", str(args.max_percentage),
        "--min_percentage", str(args.min_percentage),
        "--max_objects", str(args.max_objects),
        "--motion_weight", str(args.motion_weight),
        "--size_weight", str(args.size_weight),
        "--center_weight", str(args.center_weight),
        "--quality", str(args.quality)
    ]
    
    # 添加可选参数
    if args.visualization:
        cmd.append("--visualization")
    
    if args.save_metrics:
        cmd.append("--save_metrics")
    
    # 执行命令
    cmd_str = " ".join(cmd)
    print(f"运行命令: {cmd_str}")
    os.system(cmd_str)
    
    adaptive_time = time.time() - start_time
    
    # 如果需要与静态方法比较
    if args.compare_with_static:
        # 解析静态百分比
        static_percentages = [float(p) for p in args.static_percentages.split(',')]
        static_results = []
        
        for percentage in static_percentages:
            print(f"\n--- 运行静态百分比: {percentage:.1f} ---")
            
            static_output = f"{args.output_folder}/static_{percentage:.1f}.mp4"
            static_frames_dir = f"{args.output_folder}/static_frames_{percentage:.1f}"
            
            start_time = time.time()
            
            # 构建命令
            cmd = [
                "python", "object_tracking_process.py",
                "--model", args.model,
                "--model_path", args.model_path,
                "--input_video", args.input_video,
                "--output_video", static_output,
                "--output_folder", static_frames_dir,
                "--detector_weights", args.detector_weights,
                "--percentage", str(percentage),
                "--quality", str(args.quality),
                "--tracker_type", "ByteTrack",
                "--max_objects", str(args.max_objects)
            ]
            
            # 执行命令
            cmd_str = " ".join(cmd)
            print(f"运行命令: {cmd_str}")
            os.system(cmd_str)
            
            static_time = time.time() - start_time
            
            # 计算指标
            if os.path.exists(static_output):
                metrics = calculate_metrics(static_output)
                metrics['process_time'] = static_time
                metrics['percentage'] = percentage
                static_results.append(metrics)
                print(f"处理完成! 大小: {metrics['size_mb']:.2f} MB, 时间: {static_time:.2f} 秒")
            else:
                print(f"警告: 无法找到输出视频 {static_output}")
        
        # 计算自适应方法的指标
        if os.path.exists(adaptive_output):
            adaptive_metrics = calculate_metrics(adaptive_output)
            adaptive_metrics['process_time'] = adaptive_time
            
            # 打印比较结果
            print("\n--- 结果比较 ---")
            print(f"{'方法':<15} {'大小 (MB)':<12} {'比特率 (kbps)':<15} {'处理时间 (秒)':<15}")
            print("-" * 60)
            
            print(f"{'自适应':<15} {adaptive_metrics['size_mb']:<12.2f} {adaptive_metrics['bitrate_kbps']:<15.2f} {adaptive_metrics['process_time']:<15.2f}")
            
            for result in static_results:
                method_name = f"静态 {result['percentage']:.1f}"
                print(f"{method_name:<15} {result['size_mb']:<12.2f} {result['bitrate_kbps']:<15.2f} {result['process_time']:<15.2f}")
                
            # 创建对比总结文件
            with open(f"{args.output_folder}/comparison_summary.txt", 'w') as f:
                f.write("方法,大小(MB),比特率(kbps),处理时间(秒)\n")
                f.write(f"自适应,{adaptive_metrics['size_mb']:.2f},{adaptive_metrics['bitrate_kbps']:.2f},{adaptive_metrics['process_time']:.2f}\n")
                
                for result in static_results:
                    f.write(f"静态 {result['percentage']:.1f},{result['size_mb']:.2f},{result['bitrate_kbps']:.2f},{result['process_time']:.2f}\n")
            
            # 生成比较图表
            if args.plot_comparison:
                generate_comparison_plots(args.output_folder, adaptive_metrics, static_results)
            
            print(f"\n比较完成! 结果保存在 {args.output_folder}/comparison_summary.txt")
        else:
            print(f"警告: 无法找到自适应方法的输出视频 {adaptive_output}")
    else:
        # 只计算自适应方法的指标
        if os.path.exists(adaptive_output):
            metrics = calculate_metrics(adaptive_output)
            print(f"处理完成! 大小: {metrics['size_mb']:.2f} MB, 比特率: {metrics['bitrate_kbps']:.2f} kbps, 时间: {adaptive_time:.2f} 秒")
        else:
            print(f"警告: 无法找到输出视频 {adaptive_output}")
    
    print(f"\n测试完成! 结果保存在 {args.output_folder}")

if __name__ == "__main__":
    main() 