#!/usr/bin/env python3

import os
import argparse
import time

def parse_args():
    parser = argparse.ArgumentParser(description='测试优化后的检测-追踪方法')
    
    parser.add_argument('--input_video', type=str, default='video.mp4', help='输入视频文件路径')
    parser.add_argument('--model', type=str, default='imagenet', choices=['cifar', 'imagenet'], help='LimitNet模型类型')
    parser.add_argument('--model_path', type=str, default='LimitNet-ImageNet', help='LimitNet模型路径')
    parser.add_argument('--output_folder', type=str, default='./optimized_results/', help='输出文件夹')
    parser.add_argument('--percentages', type=str, default='0.05,0.1,0.25,1.0', help='保留的潜在变量百分比,用逗号分隔')
    
    # 追踪方法参数
    parser.add_argument('--tracker_type', type=str, default='ByteTrack', 
                        choices=['CSRT', 'KCF', 'MOSSE', 'ByteTrack'], help='跟踪器类型')
    parser.add_argument('--max_objects', type=int, default=3, help='最大跟踪目标数量')
    parser.add_argument('--background_quality', type=float, default=0.3, help='背景区域保留比例')
    parser.add_argument('--detector_weights', type=str, default='yolov5s.pt', help='检测器权重')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_folder, exist_ok=True)
    
    # 解析百分比列表
    percentages = [float(p) for p in args.percentages.split(',')]
    
    # 运行不同百分比的测试
    for percentage in percentages:
        print(f"\n--- 处理百分比: {percentage:.2f} ---")
        
        output_video = f"{args.output_folder}/optimized_tracking_{percentage:.2f}.mp4"
        
        start_time = time.time()
        
        # 构建命令
        cmd = [
            "python", "object_tracking_process.py",
            "--model", args.model,
            "--model_path", args.model_path,
            "--input_video", args.input_video,
            "--output_video", output_video,
            "--detector_weights", args.detector_weights,
            "--confidence", "0.25",
            "--percentage", str(percentage),
            "--tracker_type", args.tracker_type,
            "--max_objects", str(args.max_objects),
            "--background_quality", str(args.background_quality)
        ]
        
        # 执行命令
        cmd_str = " ".join(cmd)
        print(f"运行命令: {cmd_str}")
        os.system(cmd_str)
        
        process_time = time.time() - start_time
        
        # 检查输出视频是否存在
        if os.path.exists(output_video):
            file_size = os.path.getsize(output_video) / (1024 * 1024)  # MB
            print(f"处理完成! 大小: {file_size:.2f} MB, 时间: {process_time:.2f} 秒")
        else:
            print(f"警告: 无法找到输出视频 {output_video}")
    
    print(f"\n所有测试完成! 结果保存在 {args.output_folder}")

if __name__ == "__main__":
    main() 