# 视频水印添加工具

使用Python脚本批量给视频添加动态水印，支持文本和图片。

## 功能特性
- 支持文本水印，支持自定义字体、字号、加粗、描边、阴影等
- 支持图片水印，支持调节透明度、大小
- 支持多种水印运动区域和轨迹
- 支持多个水印同时添加
- 预览水印添加效果
- 批量处理多个视频文件，支持文件夹选择
- 提供可视化Gradio界面
- 
- 配置预设保存和加载

## 环境要求
- Python
- Conda (推荐)

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
