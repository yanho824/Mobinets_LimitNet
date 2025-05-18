# LimitNet: 渐进式、内容感知的图像与视频处理框架

欢迎使用LimitNet项目！本项目是论文 **[LimitNet: Progressive, Content-Aware Image Offloading for Extremely Weak Devices & Networks (MobiSys'2024)](https://dl.acm.org/doi/10.1145/3643832.3661856)** 的官方代码仓库。

## 项目概要

LimitNet是一个专为弱设备和低带宽网络设计的渐进式、内容感知的图像与视频压缩框架。它解决了物联网设备在使用先进视觉模型时面临的硬件限制和带宽瓶颈问题。

### 核心功能

- **渐进式数据传输**：编码器优先传输关键数据，即使数据部分丢失也能进行推理
- **内容感知压缩**：识别图像中的重要区域，保证重要内容的高质量重建
- **检测-追踪增强**：集成目标检测和追踪技术，提高视频处理效果
- **自适应编码**：根据视频内容动态调整压缩率和质量分配

### 主要模块

1. **基础LimitNet**：使用显著性检测的原始渐进式压缩
2. **检测-追踪LimitNet**：用YOLOv5检测和ByteTrack追踪替代显著性检测
3. **自适应编码LimitNet**：动态调整压缩率和质量分配的高级实现

## 技术亮点

- **高压缩率**：可减少高达83.68%的带宽使用，同时保持良好推理性能
- **低网络需求**：专为LPWAN等低带宽网络设计
- **语义理解**：保留图像的语义重要信息，而非仅基于像素统计
- **端云协同**：优化弱设备与云端之间的协作效率

## 性能对比

实验表明，与现有方法相比，LimitNet取得了显著优势：

- 在ImageNet1000上提高了14.01个百分点的准确率
- 在CIFAR100上提高了18.01个百分点的准确率
- 在COCO上将mAP@0.5提高了0.1
- 与JPEG相比，在STM32F7上的编码时间仅增加4%

## 最新功能：检测-追踪和自适应编码

我们扩展了LimitNet，引入了更强大的视频处理功能：

### 检测-追踪增强 

1. 用目标检测和追踪替代显著性检测
2. 支持多目标追踪和优先级排序
3. 径向渐变显著性图，提供自然的质量过渡
4. 背景保留机制，避免背景完全丢失

### 自适应编码技术

1. 动态压缩率控制（可在指定范围内自动调整）
2. 基于目标重要性的区域权重分配
3. 时间平滑机制，确保视频质量稳定
4. 运动感知编码，关注移动目标

### 完整的分析工具

1. 性能对比图表生成
2. 质量评估（PSNR、文件大小、处理时间）
3. 可视化显著性图和追踪结果
4. 自适应编码参数调节

## 视频演示

LimitNet可以提取输入图像的重要部分并在其渐进式比特流中优先处理这些部分。这使云端能够在任何时间点重建图像并执行推理。

<视频演示链接>

## 安装与环境配置

```bash
# 克隆仓库
git clone https://github.com/ds-kiel/LimitNet.git
cd LimitNet

# 安装依赖
pip install -r requirements.txt

# 安装YOLOv5依赖（用于检测-追踪功能）
pip install ultralytics
```

## 使用方法

### 图像压缩

```bash
# 评估单张图像
python demo.py --model_path <MODEL_PATH> --image_path <IMAGE_PATH> --percentage <PERCENTAGE>
```

### 基础视频处理

```bash
# 原始LimitNet视频处理
python process_video_frames.py --model imagenet --model_path LimitNet-ImageNet \
    --input_video video.mp4 --output_video output.mp4 --percentage 0.5
```

### 检测-追踪视频处理

```bash
# 测试检测和追踪
python test_tracking.py --input_video video.mp4 --output_video tracked.mp4

# 检测-追踪集成处理
python object_tracking_process.py --model imagenet --model_path LimitNet-ImageNet \
    --input_video video.mp4 --output_video output.mp4 --percentage 0.5 \
    --tracker_type ByteTrack --max_objects 3
```

### 自适应编码视频处理

```bash
# 自适应视频编码
python adaptive_tracking_encode.py --model imagenet --model_path LimitNet-ImageNet \
    --input_video video.mp4 --output_video adaptive_output.mp4 \
    --min_percentage 0.1 --max_percentage 0.7 --visualization
```

### 方法对比

```bash
# 对比不同方法
python compare_methods.py --input_video video.mp4 --model imagenet \
    --model_path LimitNet-ImageNet --percentages 0.1,0.3,0.5,0.7
```

## 预训练模型

您可以从以下链接下载预训练权重：

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.12516726.svg)](https://doi.org/10.5281/zenodo.12516726)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.12516726.svg)](https://zenodo.org/records/15019456)

## 显著性数据集

LimitNet使用BASNet作为教师模型训练显著性检测分支。我们使用该模型提取了ImageNet数据集子集的显著性图，可从以下链接下载：

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.12206178.svg)](https://doi.org/10.5281/zenodo.12206178)

## 项目结构

- `models/`: LimitNet模型定义
- `process_video_frames.py`: 基础视频处理
- `object_tracking_process.py`: 检测-追踪视频处理
- `adaptive_tracking_encode.py`: 自适应编码实现
- `compare_methods.py`: 方法对比工具
- `test_tracking.py`: 检测-追踪测试工具
- `README_ADAPTIVE_TRACKING.md`: 自适应追踪详细文档
- `README_optimization.md`: 优化方法详细文档

## 引用

```
@inproceedings{10.1145/3643832.3661856,
author = {Hojjat, Ali and Haberer, Janek and Zainab, Tayyaba and Landsiedel, Olaf},
title = {LimitNet: Progressive, Content-Aware Image Offloading for Extremely Weak Devices \& Networks},
year = {2024},
isbn = {9798400705816},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3643832.3661856},
doi = {10.1145/3643832.3661856},
booktitle = {Proceedings of the 22nd Annual International Conference on Mobile Systems, Applications and Services},
pages = {519–533},
numpages = {15},
location = {Minato-ku, Tokyo, Japan},
series = {MOBISYS '24}
}
```

