import argparse
import subprocess
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import time
import json

def parse_args():
    parser = argparse.ArgumentParser(description='对比不同视频处理方法的性能')
    
    parser.add_argument('--input_video', type=str, required=True, help='输入视频文件路径')
    parser.add_argument('--model', type=str, default='imagenet', choices=['cifar', 'imagenet'], help='LimitNet模型类型')
    parser.add_argument('--model_path', type=str, default='LimitNet-ImageNet', help='LimitNet模型路径')
    parser.add_argument('--detector_weights', type=str, default='yolov5su.pt', help='检测器权重')
    parser.add_argument('--output_folder', type=str, default='./comparison_results/', help='输出文件夹')
    parser.add_argument('--percentages', type=str, default='0.05,0.1,0.3,0.5,0.7,1.0', help='测试的压缩率，以逗号分隔')
    
    return parser.parse_args()

def calculate_metrics(video_path):
    """计算视频文件的大小和其他指标"""
    file_size = float(os.path.getsize(video_path) / (1024 * 1024))  # MB
    
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    duration = float(frame_count / fps if fps > 0 else 0)
    cap.release()
    
    bitrate = float((file_size * 8 * 1024) / duration if duration > 0 else 0)
    
    return {
        'size_mb': file_size,
        'frames': frame_count,
        'fps': fps,
        'duration': duration,
        'bitrate_kbps': bitrate
    }

def run_method(method, args, percentage):
    """运行指定的视频处理方法"""
    output_video = f"{args.output_folder}/{method}_p{percentage:.2f}.mp4"
    output_folder = f"{args.output_folder}/{method}_frames_p{percentage:.2f}"
    
    start_time = time.time()
    
    if method == "static":
        # 运行原始LimitNet方法
        cmd = [
            "python", "process_video_frames.py",
            "--model", args.model,
            "--model_path", args.model_path,
            "--input_video", args.input_video,
            "--output_video", output_video,
            "--output_folder", output_folder,
            "--percentage", str(percentage)
        ]
    elif method == "tracking":
        # 运行检测-追踪方法
        cmd = [
            "python", "object_tracking_process.py",
            "--model", args.model,
            "--model_path", args.model_path,
            "--input_video", args.input_video,
            "--output_video", output_video,
            "--output_folder", output_folder,
            "--detector_weights", args.detector_weights,
            "--percentage", str(percentage),
            "--tracker_type", "ByteTrack",
            "--quality", "95"
        ]
    
    cmd_str = " ".join(cmd)
    print(f"运行命令: {cmd_str}")
    subprocess.run(cmd_str, shell=True)
    
    process_time = time.time() - start_time
    
    # 计算视频指标
    metrics = calculate_metrics(output_video)
    metrics['process_time'] = process_time
    metrics['percentage'] = percentage
    metrics['method'] = method
    
    return metrics

def calculate_psnr(original_video, processed_video, max_frames=100):
    """计算两个视频之间的PSNR"""
    cap_orig = cv2.VideoCapture(original_video)
    cap_proc = cv2.VideoCapture(processed_video)
    
    total_frames = min(int(cap_orig.get(cv2.CAP_PROP_FRAME_COUNT)), 
                      int(cap_proc.get(cv2.CAP_PROP_FRAME_COUNT)))
    
    # 限制帧数以加快计算
    frames_to_process = min(total_frames, max_frames)
    
    psnr_values = []
    
    for i in tqdm(range(frames_to_process), desc="计算PSNR"):
        ret_orig, frame_orig = cap_orig.read()
        ret_proc, frame_proc = cap_proc.read()
        
        if not ret_orig or not ret_proc:
            break
        
        # 确保尺寸匹配
        if frame_orig.shape != frame_proc.shape:
            frame_proc = cv2.resize(frame_proc, (frame_orig.shape[1], frame_orig.shape[0]))
        
        # 计算MSE
        mse = np.mean((frame_orig.astype(np.float32) / 255.0 - 
                      frame_proc.astype(np.float32) / 255.0) ** 2)
        if mse == 0:
            psnr = 100
        else:
            psnr = 20 * np.log10(1.0 / np.sqrt(mse))
        
        psnr_values.append(psnr)
    
    cap_orig.release()
    cap_proc.release()
    
    return np.mean(psnr_values) if psnr_values else 0

def generate_plots(results, args):
    """生成比较图表"""
    # 按方法分组结果
    static_results = [r for r in results if r['method'] == 'static']
    tracking_results = [r for r in results if r['method'] == 'tracking']
    
    # 确保按百分比排序
    static_results.sort(key=lambda x: x['percentage'])
    tracking_results.sort(key=lambda x: x['percentage'])
    
    # 提取数据
    percentages = [r['percentage'] for r in static_results]
    static_sizes = [r['size_mb'] for r in static_results]
    tracking_sizes = [r['size_mb'] for r in tracking_results]
    
    static_bitrates = [r['bitrate_kbps'] for r in static_results]
    tracking_bitrates = [r['bitrate_kbps'] for r in tracking_results]
    
    static_times = [r['process_time'] for r in static_results]
    tracking_times = [r['process_time'] for r in tracking_results]
    
    static_psnrs = [r.get('psnr', 0) for r in static_results]
    tracking_psnrs = [r.get('psnr', 0) for r in tracking_results]
    
    # 创建图表
    plt.figure(figsize=(10, 8))
    
    # 文件大小对比
    plt.subplot(2, 2, 1)
    plt.plot(percentages, static_sizes, 'o-', label='Original LimitNet')
    plt.plot(percentages, tracking_sizes, 's-', label='Detection-Tracking')
    plt.title('File Size Comparison')
    plt.xlabel('Retention Percentage')
    plt.ylabel('Size (MB)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 比特率对比
    plt.subplot(2, 2, 2)
    plt.plot(percentages, static_bitrates, 'o-', label='Original LimitNet')
    plt.plot(percentages, tracking_bitrates, 's-', label='Detection-Tracking')
    plt.title('Bitrate Comparison')
    plt.xlabel('Retention Percentage')
    plt.ylabel('Bitrate (kbps)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 处理时间对比
    plt.subplot(2, 2, 3)
    plt.plot(percentages, static_times, 'o-', label='Original LimitNet')
    plt.plot(percentages, tracking_times, 's-', label='Detection-Tracking')
    plt.title('Processing Time Comparison')
    plt.xlabel('Retention Percentage')
    plt.ylabel('Time (seconds)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # PSNR对比
    plt.subplot(2, 2, 4)
    plt.plot(percentages, static_psnrs, 'o-', label='Original LimitNet')
    plt.plot(percentages, tracking_psnrs, 's-', label='Detection-Tracking')
    plt.title('PSNR Comparison')
    plt.xlabel('Retention Percentage')
    plt.ylabel('PSNR (dB)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{args.output_folder}/comparison_plots.png", dpi=300)
    plt.close()
    
    # 生成文件大小vs质量图表
    plt.figure(figsize=(10, 6))
    plt.plot(static_sizes, static_psnrs, 'o-', label='Original LimitNet')
    plt.plot(tracking_sizes, tracking_psnrs, 's-', label='Detection-Tracking')
    plt.title('File Size vs Quality')
    plt.xlabel('File Size (MB)')
    plt.ylabel('PSNR (dB)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{args.output_folder}/size_vs_quality.png", dpi=300)
    plt.close()
    
    # 生成操作特性曲线 (ROC类似的比特率vs质量)
    plt.figure(figsize=(10, 6))
    plt.plot(static_bitrates, static_psnrs, 'o-', label='Original LimitNet')
    plt.plot(tracking_bitrates, tracking_psnrs, 's-', label='Detection-Tracking')
    plt.title('Bitrate vs Quality')
    plt.xlabel('Bitrate (kbps)')
    plt.ylabel('PSNR (dB)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{args.output_folder}/bitrate_vs_quality.png", dpi=300)
    plt.close()

def main():
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_folder, exist_ok=True)
    
    # 解析百分比列表
    percentages = [float(p) for p in args.percentages.split(',')]
    
    all_results = []
    
    # 运行两种方法的不同百分比测试
    for percentage in percentages:
        print(f"\n--- Testing Original LimitNet Method, Retention Percentage: {percentage:.2f} ---")
        static_result = run_method("static", args, percentage)
        all_results.append(static_result)
        
        print(f"\n--- Testing Detection-Tracking Method, Retention Percentage: {percentage:.2f} ---")
        tracking_result = run_method("tracking", args, percentage)
        all_results.append(tracking_result)
    
    # 计算PSNR
    print("\n--- Calculating Video Quality Metrics ---")
    for result in all_results:
        method = result['method']
        percentage = result['percentage']
        processed_video = f"{args.output_folder}/{method}_p{percentage:.2f}.mp4"
        
        print(f"Calculating PSNR for {method} p={percentage:.2f}...")
        psnr = calculate_psnr(args.input_video, processed_video)
        result['psnr'] = float(psnr)
        print(f"PSNR: {psnr:.2f} dB")
    
    # 保存结果 - 确保所有值都是JSON可序列化的
    serializable_results = []
    for result in all_results:
        serializable_result = {}
        for key, value in result.items():
            # 将所有NumPy类型转换为Python原生类型
            if hasattr(value, 'item'):
                serializable_result[key] = value.item()
            else:
                serializable_result[key] = value
        serializable_results.append(serializable_result)
    
    with open(f"{args.output_folder}/comparison_results.json", 'w') as f:
        json.dump(serializable_results, f, indent=4)
    
    # 生成对比图表
    generate_plots(all_results, args)
    
    # 打印对比摘要
    print("\n--- Method Comparison Summary ---")
    print(f"{'Method':<15} {'Percentage':<10} {'Size(MB)':<12} {'Bitrate(kbps)':<15} {'PSNR(dB)':<10} {'Time(s)':<15}")
    print("-" * 80)
    
    for result in all_results:
        method_name = "Original LimitNet" if result['method'] == 'static' else "Detection-Tracking"
        print(f"{method_name:<15} {result['percentage']:<10.2f} {result['size_mb']:<12.2f} "
              f"{result['bitrate_kbps']:<15.2f} {result.get('psnr', 0):<10.2f} {result['process_time']:<15.2f}")
    
    print(f"\nComparison complete! Results saved in {args.output_folder}")

if __name__ == "__main__":
    main() 