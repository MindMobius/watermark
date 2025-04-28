# 视频批量添加水印工具

使用Python脚本批量给视频添加动态水印，支持文本和图片。

## 功能特性
- 支持文本水印，支持自定义字体、字号、加粗、描边、阴影等
- 支持图片水印，支持调节透明度、大小
- 支持多种水印运动区域和轨迹
- 支持多个水印同时添加
- 预览水印添加效果
- 批量处理多个视频文件，支持文件夹选择
- 提供可视化Gradio界面
- 配置预设保存和加载

## 待实现
- [ ] 字体效果编辑（字号、加粗、描边、阴影、颜色、字间距、行间距）
- [ ] 增加更多水印运动区域和轨迹
- [ ] 支持多个水印同时添加
- [ ] 支持选择视频文件夹
- [ ] 水印配置预设支持保存和加载
- [ ] 支持调节导出视频的文件路径、命名格式、分辨率、格式
- [ ] 使用GPU加速视频导出
- [ ] 导出时保留原视频音频

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
