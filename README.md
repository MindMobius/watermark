# 视频水印添加工具

使用Python脚本批量给视频添加动态水印，支持文本和图片，可同时添加多个水印。

## 功能特性
- 支持文本水印（自定义字体、大小、颜色、透明度）
- 支持图片水印（可调节大小、透明度）
- 可选择水印运动区域和轨迹
- 预览水印添加效果
- 批量处理多个视频文件
- 提供可视化Gradio界面

## 环境要求
- Python
- Conda (推荐)
- 不使用moviepy库（导入问题）

## 安装指南
使用Conda创建环境
```
conda create -n watermark python
```
切换到环境
```bash
conda activate watermark
```
安装依赖
```bash
pip install -r requirements.txt
```
启动Gradio应用
```bash
python watermark_tool.py
```
退出环境
```bash
conda deactivate
```
