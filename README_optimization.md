# 检测-追踪方法优化

本项目包含对LimitNet的检测-追踪方法的多项优化，旨在提高视频处理质量、减小文件大小并减少处理时间。

## 主要改进

### 1. 优化的检测-追踪方法 (`object_tracking_process.py`)

对原始检测-追踪方法进行了以下改进：

- **多目标支持**：支持同时跟踪多个目标，并按面积大小自动选择最重要的目标
- **径向渐变显著性图**：从目标中心向边缘的渐变显著性，提供更自然的质量过渡
- **高斯模糊平滑**：应用高斯模糊使显著性边界更加平滑
- **背景区域保留**：可配置的背景质量保留比例，避免背景完全丢失
- **集成ByteTrack**：整合更先进的ByteTrack追踪器，提高追踪准确性
- **视频编码优化**：使用FFmpeg进行二次编码，减小文件大小
- **GOP结构优化**：合理设置I帧间隔，提高压缩效率

### 2. 自适应编码方法 (`adaptive_tracking_encode.py`)

完全重构的自适应编码方法，引入以下创新：

- **运动感知重要性评分**：使用运动速度、大小和中心距离评估目标重要性
- **自适应压缩比**：根据场景复杂度和目标重要性动态调整压缩率
- **轨迹可视化**：可视化目标运动轨迹，便于分析
- **编码统计分析**：生成详细的编码统计图表，展示编码决策过程
- **重要性热力图**：使用不同颜色渐变表示目标的重要性

## 测试工具

- **`test_tracking_optimized.py`**：测试优化的检测-追踪方法
- **`test_adaptive_encode.py`**：测试自适应编码方法，可与静态方法对比
- **`compare_methods.py`**：全面比较所有方法，支持多种性能指标分析

## 使用方法

### 优化的检测-追踪方法

```bash
python object_tracking_process.py \
    --model imagenet \
    --model_path LimitNet-ImageNet \
    --input_video video.mp4 \
    --output_video output.mp4 \
    --percentage 0.5 \
    --tracker_type ByteTrack \
    --max_objects 3 \
    --background_quality 0.3
```

### 自适应编码方法

```bash
python adaptive_tracking_encode.py \
    --model imagenet \
    --model_path LimitNet-ImageNet \
    --input_video video.mp4 \
    --output_video adaptive_output.mp4 \
    --max_percentage 0.7 \
    --min_percentage 0.1
```

### 方法比较

```bash
python compare_methods.py \
    --input_video video.mp4 \
    --model imagenet \
    --model_path LimitNet-ImageNet \
    --percentages 0.1,0.3,0.5,0.7 \
    --methods saliency,tracking,optimized,adaptive \
    --output_folder ./comparison_results
```

## 可调参数

### 通用参数

- `--model`: 模型类型 (cifar/imagenet)
- `--model_path`: LimitNet模型路径
- `--input_video`: 输入视频文件
- `--output_video`: 输出视频文件
- `--percentage`: 保留的潜在变量百分比

### 优化的追踪参数

- `--tracker_type`: 追踪器类型 (CSRT/KCF/MOSSE/ByteTrack)
- `--max_objects`: 最大跟踪目标数量
- `--background_quality`: 背景区域保留比例
- `--detection_interval`: 检测间隔帧数
- `--saliency_blur`: 显著性图高斯模糊核大小
- `--quality`: 输出视频质量 (1-100)

### 自适应编码参数

- `--max_percentage`: 最大保留的潜在变量百分比
- `--min_percentage`: 最小保留的潜在变量百分比
- `--motion_weight`: 运动重要性权重
- `--size_weight`: 目标大小重要性权重
- `--center_weight`: 中心位置重要性权重
- `--i_frame_interval`: I帧间隔

## 性能对比

通过运行比较脚本，可以得到各方法在不同百分比下的性能对比，包括:

- PSNR (质量)
- 文件大小
- 处理时间
- 比特率
- 速率-失真曲线

## 总结

优化后的检测-追踪方法和自适应编码方法相较于原始方法有以下优势：

1. 更高的视频质量 (PSNR)
2. 更小的文件大小
3. 更自然的质量分配
4. 更好的视觉体验 