# -*- coding: utf-8 -*- # Add this for better encoding support
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import gradio as gr
import os
import time
import math
import random
import multiprocessing
from pathlib import Path # For easier path handling

# --- Configuration ---
DEFAULT_FONT_PATH = "ali.ttf" # Example: SimHei, Arial Unicode MS, Noto Sans CJK etc. USER MUST PROVIDE if needed.
DEFAULT_OUTPUT_DIR = "output_videos"

# --- Helper Functions ---

def hex_to_rgba(hex_color, alpha=255):
    """
    Converts hex color string (#RRGGBB or #RGB) to (R, G, B, A) tuple.
    The 'alpha' parameter (0-255) determines the final transparency.
    """
    if not hex_color:
        print("Warning: Color is empty, defaulting to white.")
        hex_color = "#FFFFFF"

    hex_input = hex_color.lstrip('#')

    if not all(c in '0123456789ABCDEFabcdef' for c in hex_input):
         print(f"警告：无效的十六进制颜色字符: {hex_color}，将使用白色。")
         hex_input = "FFFFFF"

    if len(hex_input) == 3:
        hex_input = ''.join([c*2 for c in hex_input])
    elif len(hex_input) != 6:
         print(f"警告：无效的十六进制颜色长度 (必须是 3 或 6 位): #{hex_input}，将使用白色。")
         hex_input = "FFFFFF"

    try:
        r = int(hex_input[0:2], 16)
        g = int(hex_input[2:4], 16)
        b = int(hex_input[4:6], 16)
    except ValueError:
        print(f"错误: 转换十六进制颜色失败: {hex_color}，将使用白色。")
        r, g, b = 255, 255, 255

    final_alpha = max(0, min(255, int(alpha)))
    return (r, g, b, final_alpha)

def apply_opacity(img, opacity_percent):
    """Applies overall opacity to a PIL image (RGBA)."""
    if img.mode != 'RGBA':
        img = img.convert('RGBA')
    alpha_channel = img.split()[3]
    opacity_factor = max(0.0, min(1.0, opacity_percent / 100.0))
    new_alpha = alpha_channel.point(lambda p: max(0, min(255, int(p * opacity_factor))))
    img.putalpha(new_alpha)
    return img

# --- Watermark Generation ---

def create_text_watermark(
    text, font_path, font_size, color_hex, opacity_percent,
    stroke_width=0, stroke_color_hex="#000000",
    shadow_offset_x=0, shadow_offset_y=0, shadow_color_hex="#000000"
):
    """Creates a transparent PIL image with styled text (color, opacity, stroke, shadow)."""
    effective_font_path = font_path
    try:
        if not effective_font_path or not os.path.exists(effective_font_path):
             if os.path.exists(DEFAULT_FONT_PATH):
                 print(f"警告: 字体文件 '{effective_font_path}' 未找到或未提供。使用默认字体: {DEFAULT_FONT_PATH}")
                 effective_font_path = DEFAULT_FONT_PATH
             else:
                 raise ValueError(f"未找到字体文件 '{effective_font_path}' 或默认字体 '{DEFAULT_FONT_PATH}'。请上传字体或确保默认字体路径有效。")
        font = ImageFont.truetype(effective_font_path, font_size)
        print(f"使用字体: {effective_font_path}")
    except IOError as e:
        print(f"错误: 加载字体文件时出错 {effective_font_path}: {e}。尝试 Pillow 默认字体。")
        try:
            font = ImageFont.load_default()
        except Exception as e_def:
             raise ValueError(f"无法加载指定字体 '{effective_font_path}' 或 Pillow 默认字体。错误: {e_def}")

    # --- Calculate text bounding box and required image size ---
    try:
        # Base alpha calculation (applied to all color components)
        base_alpha = int(255 * (opacity_percent / 100.0))

        # Get base text dimensions
        text_bbox = font.getbbox(text)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        text_offset_y = -text_bbox[1] # Vertical offset from baseline

        # Calculate maximum offsets needed for stroke and shadow
        max_offset_x = max(0, shadow_offset_x) + stroke_width
        min_offset_x = min(0, shadow_offset_x) - stroke_width
        max_offset_y = max(0, shadow_offset_y) + stroke_width
        min_offset_y = min(0, shadow_offset_y) - stroke_width

        # Calculate final image dimensions including effects
        img_width = text_width + max_offset_x - min_offset_x
        img_height = text_height + max_offset_y - min_offset_y

        # Padding around the effects for safety
        padding = 2 # Minimal padding
        img_width += 2 * padding
        img_height += 2 * padding

        # Determine drawing origin within the padded image
        # Start drawing adjusted by negative offsets and padding
        draw_origin_x = padding - min_offset_x
        draw_origin_y = padding - min_offset_y + text_offset_y # Also include baseline offset

        # Create transparent base image
        img = Image.new('RGBA', (int(img_width), int(img_height)), (255, 255, 255, 0))
        draw = ImageDraw.Draw(img)

        # --- Calculate Colors ---
        text_color_rgba = hex_to_rgba(color_hex, alpha=base_alpha)
        shadow_color_rgba = hex_to_rgba(shadow_color_hex, alpha=base_alpha) if shadow_offset_x != 0 or shadow_offset_y != 0 else None
        stroke_color_rgba = hex_to_rgba(stroke_color_hex, alpha=base_alpha) if stroke_width > 0 else None

        # --- Draw Effects (Order: Shadow -> Stroke -> Fill) ---
        # 1. Draw Shadow (if enabled)
        if shadow_color_rgba:
            shadow_pos = (draw_origin_x + shadow_offset_x, draw_origin_y + shadow_offset_y)
            # Draw shadow without stroke first
            draw.text(shadow_pos, text, font=font, fill=shadow_color_rgba)

        # 2. Draw Main Text (potentially with stroke)
        main_pos = (draw_origin_x, draw_origin_y)
        if stroke_color_rgba and stroke_width > 0:
            # Newer Pillow versions support stroke directly
            try:
                draw.text(
                    main_pos, text, font=font, fill=text_color_rgba,
                    stroke_width=stroke_width, stroke_fill=stroke_color_rgba
                )
            except TypeError: # Older Pillow might not have stroke args
                 print("警告: 当前 Pillow 版本不支持描边参数，将跳过描边。请考虑升级 Pillow。")
                 draw.text(main_pos, text, font=font, fill=text_color_rgba)
            except Exception as draw_err:
                 print(f"绘制带描边的文本时出错: {draw_err}")
                 draw.text(main_pos, text, font=font, fill=text_color_rgba) # Fallback
        else:
            # Draw text without stroke
            draw.text(main_pos, text, font=font, fill=text_color_rgba)


        # --- Crop to actual content ---
        try:
             bbox = img.getbbox()
             if bbox:
                 img = img.crop(bbox)
             else:
                 print("警告: 水印内容完全透明或为空，生成 1x1 透明图像。")
                 img = Image.new('RGBA', (1, 1), (255, 255, 255, 0))
        except Exception as crop_e:
             print(f"警告: 裁剪文本水印时出错: {crop_e}. 使用未裁剪图像。")
             pass # Use the padded image if cropping fails

        if img.width <= 0 or img.height <= 0:
             print("警告: 创建的文本水印尺寸为零或负数，将使用1x1透明像素。")
             img = Image.new('RGBA', (1, 1), (255, 255, 255, 0))

        return img

    except Exception as e:
        print(f"错误: 创建文本水印时发生未知错误: {e}")
        # Fallback to a minimal transparent image on error
        return Image.new('RGBA', (10, 10), (255, 255, 255, 0))


def load_image_watermark(image_path, scale_percent, frame_width, frame_height, opacity_percent):
    """Loads, scales, and applies opacity to an image watermark."""
    if not image_path or not os.path.exists(image_path):
         raise ValueError(f"图片水印文件未找到或无效: {image_path}")
    try:
        img = Image.open(image_path).convert("RGBA")
    except Exception as e:
        raise ValueError(f"打开图片水印时出错 {image_path}: {e}")

    # Scaling based on frame dimension and percentage
    base_dimension = max(frame_width, frame_height)
    target_width = int(base_dimension * (scale_percent / 100.0))

    if img.width > 0:
        img_ratio = img.height / img.width
        target_height = int(target_width * img_ratio)
    else: target_height = 1 # Avoid zero height

    target_width = max(1, target_width)
    target_height = max(1, target_height)

    try:
        # Use LANCZOS for high-quality resize
        resample_filter = Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.ANTIALIAS
        img = img.resize((target_width, target_height), resample_filter)
    except Exception as resize_e:
         raise ValueError(f"缩放图片水印时出错: {resize_e}")

    # Apply Opacity
    img = apply_opacity(img, opacity_percent)
    return img


# --- Motion Calculation (WatermarkMover class remains largely the same) ---
class WatermarkMover:
    """计算水印位置基于选定的路径和区域。"""
    def __init__(self, path_type, frame_width, frame_height, watermark_width, watermark_height, margin_percent, speed_factor):
        self.path_type = path_type
        self.speed_factor = max(0.1, speed_factor)

        margin_x = int(frame_width * (margin_percent / 100.0))
        margin_y = int(frame_height * (margin_percent / 100.0))

        self.min_x = margin_x
        self.min_y = margin_y
        self.max_x = frame_width - margin_x - watermark_width
        self.max_y = frame_height - margin_y - watermark_height

        self.max_x = max(self.min_x, self.max_x) # Ensure bounds are valid
        self.max_y = max(self.min_y, self.max_y)

        self.frame_width = frame_width
        self.frame_height = frame_height
        self.watermark_width = watermark_width
        self.watermark_height = watermark_height
        self.margin_percent = margin_percent
        self.total_frames_hint = 1 # Used for cycle calculation if needed

        # Initial state for dynamic paths
        self.x = random.randint(self.min_x, self.max_x) if self.max_x > self.min_x else self.min_x
        self.y = random.randint(self.min_y, self.max_y) if self.max_y > self.min_y else self.min_y

        base_speed_x = max(1, int(frame_width * 0.005 * self.speed_factor))
        base_speed_y = max(1, int(frame_height * 0.005 * self.speed_factor))

        self.dx = random.choice([-base_speed_x, base_speed_x]) if self.max_x > self.min_x else 0
        self.dy = random.choice([-base_speed_y, base_speed_y]) if self.max_y > self.min_y else 0

        if self.dx == 0 and self.dy == 0 and (self.max_x > self.min_x or self.max_y > self.min_y):
             self.dx = base_speed_x if self.max_x > self.min_x else 0
             self.dy = base_speed_y if self.max_y > self.min_y and self.dx == 0 else 0

        self.static_x = max(0, (frame_width - watermark_width) // 2)
        self.static_y = max(0, (frame_height - watermark_height) // 2)

    def update_total_frames(self, total_frames):
        """Allow updating total frames if known late."""
        self.total_frames_hint = max(1, total_frames)

    def get_position(self, frame_index):
        """计算当前帧的位置。"""
        total_frames = self.total_frames_hint # Use the stored total frames
        if self.path_type == "静态居中":
            return self.static_x, self.static_y

        elif self.path_type == "边框反弹":
            if self.max_x <= self.min_x: self.dx = 0
            if self.max_y <= self.min_y: self.dy = 0

            self.x += self.dx
            self.y += self.dy

            bounced = False
            if self.x <= self.min_x:
                self.x = self.min_x
                self.dx = abs(self.dx) if self.dx != 0 else (random.randint(1, max(1, int(self.frame_width * 0.005 * self.speed_factor)))) if self.max_x > self.min_x else 0
                bounced = True
            elif self.x >= self.max_x:
                self.x = self.max_x
                self.dx = -abs(self.dx) if self.dx != 0 else (-random.randint(1, max(1, int(self.frame_width * 0.005 * self.speed_factor)))) if self.max_x > self.min_x else 0
                bounced = True

            if self.y <= self.min_y:
                self.y = self.min_y
                self.dy = abs(self.dy) if self.dy != 0 else (random.randint(1, max(1, int(self.frame_height * 0.005 * self.speed_factor)))) if self.max_y > self.min_y else 0
                bounced = True
            elif self.y >= self.max_y:
                self.y = self.max_y
                self.dy = -abs(self.dy) if self.dy != 0 else (-random.randint(1, max(1, int(self.frame_height * 0.005 * self.speed_factor)))) if self.max_y > self.min_y else 0
                bounced = True

            self.x = max(self.min_x, min(self.x, self.max_x))
            self.y = max(self.min_y, min(self.y, self.max_y))

            return int(self.x), int(self.y)

        elif self.path_type == "水平移动":
            progress = (frame_index % total_frames) / total_frames
            amplitude = max(0, (self.max_x - self.min_x) / 2)
            center_x = self.min_x + amplitude
            frequency = 1.0 * self.speed_factor
            current_x = center_x + amplitude * math.sin(2 * math.pi * frequency * progress)
            current_y = self.min_y + max(0, (self.max_y - self.min_y) / 2)
            return int(max(self.min_x, min(current_x, self.max_x))), int(current_y)

        elif self.path_type == "垂直移动":
            progress = (frame_index % total_frames) / total_frames
            amplitude = max(0, (self.max_y - self.min_y) / 2)
            center_y = self.min_y + amplitude
            frequency = 1.0 * self.speed_factor
            current_y = center_y + amplitude * math.sin(2 * math.pi * frequency * progress)
            current_x = self.min_x + max(0, (self.max_x - self.min_x) / 2)
            return int(current_x), int(max(self.min_y, min(current_y, self.max_y)))

        else:
            print(f"警告: 未知的移动路径 '{self.path_type}'，将使用静态居中。")
            return self.static_x, self.static_y


# --- Core Video Processing ---

def add_watermark_to_video(input_video_path, output_video_path, watermark_pil, mover):
    """将准备好的水印 PIL 图像添加到视频的每一帧。 (No Gradio Progress here)"""
    cap = cv2.VideoCapture(str(input_video_path)) # Use string path
    if not cap.isOpened():
        raise IOError(f"错误: 打开视频文件失败: {input_video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if not all([fps > 0, frame_width > 0, frame_height > 0, total_frames >= 0]):
         cap.release()
         raise ValueError(f"视频属性无效: FPS={fps}, W={frame_width}, H={frame_height}, Frames={total_frames}")

    # Update mover with actual total frames if available
    mover.update_total_frames(total_frames)

    # Ensure watermark fits (resize if necessary) - Should be done *before* mover init usually,
    # but mover is already initialized. If we resize here, bounds *might* be slightly off
    # if the watermark was initially too large. Ideally, resize happens before mover creation.
    # Let's assume watermark passed is already appropriately sized (handled in worker).

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (frame_width, frame_height)) # Use string path
    if not out.isOpened():
        cap.release()
        raise IOError(f"错误: 无法创建输出视频文件: {output_video_path}")

    frame_count = 0
    start_time = time.time()
    log_interval = max(1, int(fps * 5)) # Log every 5 seconds approx

    # Pre-check if watermark has any non-transparent pixels
    has_visible_watermark = False
    if watermark_pil.mode == 'RGBA':
        alpha_min, alpha_max = watermark_pil.getextrema()[3]
        if alpha_max > 0: # Check if maximum alpha is greater than 0
             has_visible_watermark = True
    else: # If not RGBA, assume it's visible
         has_visible_watermark = True

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        if has_visible_watermark:
            try:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                pil_frame = Image.fromarray(frame_rgb).convert("RGBA")

                wm_x, wm_y = mover.get_position(frame_count)
                wm_x = max(0, min(int(wm_x), frame_width - watermark_pil.width))
                wm_y = max(0, min(int(wm_y), frame_height - watermark_pil.height))

                # Optimized pasting: Avoid creating full overlay if possible
                # Paste directly onto the frame copy using the mask
                pil_frame.paste(watermark_pil, (wm_x, wm_y), mask=watermark_pil)

                pil_frame_out_rgb = pil_frame.convert("RGB")
                frame_out_bgr = cv2.cvtColor(np.array(pil_frame_out_rgb), cv2.COLOR_RGB2BGR)

                out.write(frame_out_bgr)

            except Exception as e:
                print(f"错误: 在帧 {frame_count} 处理水印时出错: {e}. 写入原始帧。")
                out.write(frame_bgr) # Write original frame on error
        else:
             # If watermark is fully transparent, just write the original frame
             out.write(frame_bgr)


        frame_count += 1
        if frame_count % log_interval == 0:
             elapsed = time.time() - start_time
             rate = frame_count / elapsed if elapsed > 0 else 0
             print(f"  文件 {Path(input_video_path).name}: 已处理 {frame_count} / {total_frames if total_frames > 0 else '?'} 帧 ({rate:.1f} fps)")

    cap.release()
    out.release()

    end_time = time.time()
    duration = end_time - start_time
    print(f"完成处理 {Path(input_video_path).name}。耗时: {duration:.2f} 秒")


# --- Multiprocessing Worker Function ---

def process_single_video_worker(args):
    """Worker function to process one video file."""
    (
        input_path_str, output_dir_str, watermark_params, motion_params, common_params
    ) = args

    input_path = Path(input_path_str)
    output_dir = Path(output_dir_str)
    base_filename = input_path.stem
    safe_filename_base = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in base_filename) + "_watermarked"
    output_ext = ".mp4"

    # --- Unique Filename Logic ---
    output_path = output_dir / f"{safe_filename_base}{output_ext}"
    counter = 1
    while output_path.exists():
        output_path = output_dir / f"{safe_filename_base}_{counter}{output_ext}"
        counter += 1
    # --- End Unique Filename Logic ---

    status_message = f"开始处理: {input_path.name} -> {output_path.name}\n"
    print(status_message.strip()) # Log start to console

    try:
        # --- Get Video Dimensions ---
        temp_cap = cv2.VideoCapture(str(input_path))
        if not temp_cap.isOpened():
            raise IOError(f"无法打开视频 '{input_path.name}' 以获取尺寸。")
        vid_width = int(temp_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vid_height = int(temp_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # total_vid_frames = int(temp_cap.get(cv2.CAP_PROP_FRAME_COUNT)) # Get total frames for mover
        temp_cap.release()
        if vid_width <= 0 or vid_height <= 0:
             raise ValueError(f"获取到的视频尺寸无效: {vid_width}x{vid_height}")
        status_message += f"  视频尺寸: {vid_width}x{vid_height}\n"

        # --- Prepare Watermark ---
        watermark_pil = None
        watermark_type = common_params['watermark_type']
        opacity = common_params['watermark_opacity']

        if watermark_type == "文本":
            # Unpack text params
            wp = watermark_params # shortcut
            status_message += f"  创建文本水印: '{wp['text_content'][:30]}...' 尺寸:{wp['font_size']}pt Opacity:{opacity}%\n"
            watermark_pil = create_text_watermark(
                text=wp['text_content'], font_path=wp['font_path'], font_size=wp['font_size'], color_hex=wp['text_color'],
                opacity_percent=opacity,
                stroke_width=wp['stroke_width'], stroke_color_hex=wp['stroke_color'],
                shadow_offset_x=wp['shadow_offset_x'], shadow_offset_y=wp['shadow_offset_y'], shadow_color_hex=wp['shadow_color']
            )
        elif watermark_type == "图片":
            # Unpack image params
            ip = watermark_params # shortcut
            status_message += f"  加载图片水印: 缩放:{ip['image_scale']}% Opacity:{opacity}%\n"
            watermark_pil = load_image_watermark(
                ip['image_path'], ip['image_scale'], vid_width, vid_height, opacity
            )
        else:
            raise ValueError(f"未知的内部水印类型 '{watermark_type}'")

        if watermark_pil is None or watermark_pil.width <= 0 or watermark_pil.height <= 0:
            raise ValueError("水印创建失败或尺寸无效。")
        status_message += f"  水印已生成，尺寸: {watermark_pil.width}x{watermark_pil.height}\n"

        # --- Resize watermark if it's larger than frame (rare case after scaling, but safety check) ---
        if watermark_pil.width > vid_width or watermark_pil.height > vid_height:
             print(f"警告: 水印尺寸 ({watermark_pil.width}x{watermark_pil.height}) 超出帧尺寸 ({vid_width}x{vid_height})。将调整水印大小以适应。")
             ratio = min(vid_width / watermark_pil.width, vid_height / watermark_pil.height)
             new_w = max(1, int(watermark_pil.width * ratio))
             new_h = max(1, int(watermark_pil.height * ratio))
             try:
                 resample_filter = Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.ANTIALIAS
                 watermark_pil = watermark_pil.resize((new_w, new_h), resample_filter)
                 status_message += f"  水印已调整大小为: {new_w}x{new_h}\n"
             except Exception as resize_err:
                 status_message += f"  警告: 调整水印大小时出错: {resize_err}。\n"
                 # Continue with potentially clipped watermark


        # --- Initialize Mover ---
        mp = motion_params # shortcut
        mover = WatermarkMover(
            path_type=mp['motion_path'], frame_width=vid_width, frame_height=vid_height,
            watermark_width=watermark_pil.width, watermark_height=watermark_pil.height,
            margin_percent=mp['motion_margin'], speed_factor=mp['motion_speed']
        )
        status_message += f"  初始化移动器: 类型:{mp['motion_path']}, 边距:{mp['motion_margin']}%, 速度:{mp['motion_speed']}x\n"

        # --- Add Watermark to Video ---
        add_watermark_to_video(input_path, output_path, watermark_pil, mover)

        status_message += f"  成功处理并保存到: {output_path}\n"
        return status_message, str(output_path) # Return success message and path

    except Exception as e:
        error_msg = f"错误 处理 {input_path.name} 时发生错误: {type(e).__name__}: {e}\n"
        print(f"!!! {error_msg.strip()}") # Log full error to console
        status_message += error_msg
        # Clean up partially created output file if it exists
        if output_path.exists():
             try:
                 output_path.unlink()
                 status_message += f"  已删除部分生成的输出文件: {output_path.name}\n"
             except OSError as rm_err:
                 status_message += f"  警告: 无法删除部分文件 {output_path.name}: {rm_err}\n"
        return status_message, None # Return error message and None path


# --- Gradio Interface Function ---

def process_videos_interface(
    input_videos, # List of file paths from gr.Files
    watermark_type,
    # Text params
    text_content, font_file, font_size, text_color,
    stroke_width, stroke_color, shadow_offset_x, shadow_offset_y, shadow_color,
    # Image params
    image_file, image_scale,
    # Common params
    watermark_opacity,
    # Motion params
    motion_path, motion_margin, motion_speed,
    # Output params
    output_dir_name,
    progress=gr.Progress(track_tqdm=True)): # Gradio progress

    if not input_videos:
        return "错误: 未选择任何输入视频文件。", [], None

    output_base = Path(output_dir_name if output_dir_name else DEFAULT_OUTPUT_DIR)
    try:
        output_base.mkdir(parents=True, exist_ok=True)
        print(f"输出目录: {output_base.resolve()}")
    except OSError as e:
        return f"错误: 无法创建输出目录 '{output_base}': {e}", [], None

    # --- Prepare Parameters ---
    status_messages = ["参数准备中..."]
    final_font_path = None
    watermark_params = {}
    common_params = {
        'watermark_type': watermark_type,
        'watermark_opacity': watermark_opacity,
    }

    # --- Font File Handling ---
    if watermark_type == "文本":
        if font_file is not None:
            final_font_path = font_file.name # Path from Gradio TempFile
            if not Path(final_font_path).exists():
                 status_messages.append(f"警告: 上传的字体文件路径 '{final_font_path}' 无效。尝试使用默认字体。")
                 final_font_path = None
            else:
                 status_messages.append(f"使用上传的字体: {Path(final_font_path).name}")
        else:
            status_messages.append(f"未提供字体文件。尝试使用默认字体: {DEFAULT_FONT_PATH}")

        if final_font_path is None:
             if Path(DEFAULT_FONT_PATH).exists():
                 final_font_path = DEFAULT_FONT_PATH
                 status_messages.append(f"使用默认字体: {DEFAULT_FONT_PATH}")
             else:
                 error_msg = f"错误: 文本水印需要字体，但既未提供有效字体，也找不到默认字体 '{DEFAULT_FONT_PATH}'。"
                 status_messages.append(error_msg)
                 return "\n".join(status_messages), [], None

        watermark_params = {
            'text_content': text_content, 'font_path': final_font_path, 'font_size': font_size, 'text_color': text_color,
            'stroke_width': stroke_width, 'stroke_color': stroke_color,
            'shadow_offset_x': shadow_offset_x, 'shadow_offset_y': shadow_offset_y, 'shadow_color': shadow_color,
        }
    # --- Image File Handling ---
    elif watermark_type == "图片":
        if image_file is not None:
            image_watermark_path = image_file.name # Path from Gradio TempFile
            if not Path(image_watermark_path).exists():
                 status_messages.append(f"错误: 上传的图片文件路径 '{image_watermark_path}' 无效。")
                 return "\n".join(status_messages), [], None
            else:
                 status_messages.append(f"使用上传的图片: {Path(image_watermark_path).name}")
                 watermark_params = {'image_path': image_watermark_path, 'image_scale': image_scale}
        else:
             status_messages.append(f"错误: 选择了图片水印类型，但未提供图片文件。")
             return "\n".join(status_messages), [], None

    motion_params = {
        'motion_path': motion_path, 'motion_margin': motion_margin, 'motion_speed': motion_speed,
    }

    # --- Prepare Arguments for Workers ---
    task_args = []
    for video_file in input_videos:
        input_path_str = video_file.name # Get path from Gradio file object
        task_args.append((
            input_path_str, str(output_base), watermark_params, motion_params, common_params
        ))

    # --- Process Using Multiprocessing Pool ---
    num_workers = os.cpu_count()
    status_messages.append(f"\n开始使用 {num_workers} 个 CPU 核心处理 {len(task_args)} 个视频...")
    print(f"开始使用 {num_workers} 个 CPU 核心处理 {len(task_args)} 个视频...")

    output_file_paths = []
    processed_count = 0
    total_videos = len(task_args)

    # Use try-finally to ensure pool cleanup
    pool = None
    try:
        # Use context manager for pool if Python >= 3.3
        # with multiprocessing.Pool(processes=num_workers) as pool:
        pool = multiprocessing.Pool(processes=num_workers)
        # Use imap_unordered to get results as they complete for better progress feedback
        results_iterator = pool.imap_unordered(process_single_video_worker, task_args)

        # Update progress as each video finishes
        for result in results_iterator:
            processed_count += 1
            message, output_path = result
            status_messages.append(f"\n--- 结果 ({processed_count}/{total_videos}) ---")
            status_messages.append(message.strip())
            if output_path:
                output_file_paths.append(output_path)

            # Update Gradio progress bar
            progress(processed_count / total_videos, desc=f"已处理 {processed_count}/{total_videos} 个视频")

        pool.close()
        pool.join()

    except Exception as pool_err:
        error_msg = f"!!! 多进程处理时发生严重错误: {pool_err}"
        print(error_msg)
        status_messages.append(error_msg)
    finally:
        if pool:
            pool.terminate() # Force terminate if something went wrong

    status_messages.append("\n批量处理完成。")
    final_status = "\n".join(status_messages)
    print("批量处理完成。")

    # Return status log, list of output file paths, and None for preview
    return final_status, output_file_paths, None


# --- Preview Function ---
def generate_preview(
    input_videos, # Need this to get a frame
    watermark_type,
    # Text params
    text_content, font_file, font_size, text_color,
    stroke_width, stroke_color, shadow_offset_x, shadow_offset_y, shadow_color,
    # Image params
    image_file, image_scale,
    # Common params
    watermark_opacity,
    # Motion params
    motion_path, motion_margin, motion_speed):

    if not input_videos:
        img = Image.new('RGB', (300, 100), color = 'black')
        d = ImageDraw.Draw(img)
        try: font = ImageFont.truetype("arial.ttf", 15)
        except IOError: font = ImageFont.load_default()
        d.text((10,10), "请先上传一个视频文件\n才能生成预览", fill='white', font=font)
        return img

    preview_status = ""
    try:
        input_path = Path(input_videos[0].name) # Use the first video
        preview_status += f"使用视频 '{input_path.name}' 的一帧进行预览。\n"

        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise IOError(f"无法打开视频 '{input_path.name}' 进行预览。")

        frame_to_preview = 10 # Try a slightly later frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_to_preview)
        ret, frame_bgr = cap.read()
        if not ret:
             cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
             ret, frame_bgr = cap.read()
             if not ret: raise ValueError("无法读取视频帧进行预览。")
             frame_to_preview = 0

        vid_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vid_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if vid_width <= 0 or vid_height <= 0: raise ValueError("预览时视频尺寸无效。")

        # --- Prepare Watermark ---
        watermark_pil = None
        final_font_path_preview = None
        if watermark_type == "文本":
            # Font path logic for preview
            if font_file is not None and Path(font_file.name).exists():
                 final_font_path_preview = font_file.name
            elif Path(DEFAULT_FONT_PATH).exists():
                 final_font_path_preview = DEFAULT_FONT_PATH
            if final_font_path_preview is None:
                 raise ValueError("预览需要字体，但未提供或找不到默认字体。")

            watermark_pil = create_text_watermark(
                text=text_content, font_path=final_font_path_preview, font_size=font_size, color_hex=text_color,
                opacity_percent=watermark_opacity,
                stroke_width=stroke_width, stroke_color_hex=stroke_color,
                shadow_offset_x=shadow_offset_x, shadow_offset_y=shadow_offset_y, shadow_color_hex=shadow_color
            )
            preview_status += f"创建文本水印 (字体: {Path(final_font_path_preview).name}, 尺寸: {font_size}pt, Opacity: {watermark_opacity}%)\n"

        elif watermark_type == "图片":
            if image_file is None or not Path(image_file.name).exists():
                raise ValueError("预览需要图片，但未提供或文件无效。")
            image_watermark_path_preview = image_file.name
            watermark_pil = load_image_watermark(
                image_watermark_path_preview, image_scale, vid_width, vid_height, watermark_opacity
            )
            preview_status += f"加载图片水印 (文件: {Path(image_watermark_path_preview).name}, 缩放: {image_scale}%, Opacity: {watermark_opacity}%)\n"
        else:
             raise ValueError("无效的水印类型。")

        if watermark_pil is None or watermark_pil.width <= 0 or watermark_pil.height <= 0:
             raise ValueError("预览时水印创建失败或尺寸无效。")
        preview_status += f"水印尺寸: {watermark_pil.width}x{watermark_pil.height}\n"

        # --- Initialize Mover ---
        mover = WatermarkMover(
            path_type=motion_path, frame_width=vid_width, frame_height=vid_height,
            watermark_width=watermark_pil.width, watermark_height=watermark_pil.height,
            margin_percent=motion_margin, speed_factor=motion_speed
        )
        mover.update_total_frames(total_frames) # Give mover frame count hint

        # --- Get Position & Composite ---
        wm_x, wm_y = mover.get_position(frame_to_preview)
        wm_x = max(0, min(int(wm_x), vid_width - watermark_pil.width))
        wm_y = max(0, min(int(wm_y), vid_height - watermark_pil.height))
        preview_status += f"水印位置 (帧 {frame_to_preview}): ({wm_x}, {wm_y})\n"

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_frame = Image.fromarray(frame_rgb).convert("RGBA")

        # Composite (using paste with mask)
        if watermark_pil.mode == 'RGBA' and watermark_pil.getextrema()[3][1] > 0: # Check if alpha max > 0
            pil_frame.paste(watermark_pil, (wm_x, wm_y), mask=watermark_pil)
        elif watermark_pil.mode != 'RGBA': # Assume opaque if not RGBA
            pil_frame.paste(watermark_pil, (wm_x, wm_y))


        preview_status += "预览图生成成功。"
        print(preview_status)
        return pil_frame.convert("RGB")

    except Exception as e:
        error_msg = f"生成预览时出错: {type(e).__name__}: {e}"
        print(error_msg)
        img = Image.new('RGB', (400, 150), color = 'darkred')
        d = ImageDraw.Draw(img)
        try: font = ImageFont.truetype("arial.ttf", 15)
        except IOError: font = ImageFont.load_default()
        # Wrap text
        import textwrap
        lines = textwrap.wrap(f"生成预览失败:\n{error_msg}", width=50)
        y_text = 10
        for line in lines:
            width, height = font.getsize(line) if hasattr(font, 'getsize') else (10*len(line), 15) # Basic fallback
            d.text((10, y_text), line, font=font, fill='white')
            y_text += height + 2 # Move y down for next line
        # d.text((10,10), f"生成预览失败:\n{error_msg}", fill='white', font=font)
        return img


# --- Build Gradio UI ---

# Check default font existence early
default_font_exists = Path(DEFAULT_FONT_PATH).exists()
default_font_warning = f" (默认: {Path(DEFAULT_FONT_PATH).name}{'' if default_font_exists else ' - 未找到!'})"

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎬 视频水印添加工具 (增强版)")
    gr.Markdown("为视频添加动态文本或图片水印。支持文本描边/阴影、文件名防覆盖、多核处理加速。")

    with gr.Row():
        with gr.Column(scale=3): # Input settings column slightly wider
            gr.Markdown("### 1. 输入视频")
            input_videos = gr.Files(label="上传视频文件 (可多选)", file_types=['video'], type="filepath") # Use filepath

            gr.Markdown("### 2. 水印设置")
            watermark_type = gr.Radio(["文本", "图片"], label="水印类型", value="文本")

            # --- Common Settings ---
            watermark_opacity = gr.Slider(label="水印不透明度 (%)", minimum=0, maximum=100, value=70, step=1)

            # --- Text Options ---
            with gr.Group(visible=True) as text_options:
                gr.Markdown("#### 文本水印选项")
                text_content = gr.Textbox(label="水印文字", value="在此输入水印内容", lines=2)
                font_file = gr.File(label=f"上传字体文件 (.ttf, .otf) {default_font_warning}", file_types=['.ttf', '.otf'], type="filepath") # Use filepath
                font_size = gr.Slider(label="字体大小 (pt)", minimum=8, maximum=300, value=48, step=1)
                text_color = gr.ColorPicker(label="文字颜色 (Hex)", value="#FFFFFF")
                with gr.Accordion("描边 & 阴影 (可选)", open=False):
                    stroke_width = gr.Slider(label="描边宽度 (px)", minimum=0, maximum=10, value=0, step=1)
                    stroke_color = gr.ColorPicker(label="描边颜色", value="#000000")
                    gr.Markdown("---")
                    shadow_offset_x = gr.Slider(label="阴影 X 偏移 (px)", minimum=-10, maximum=10, value=0, step=1)
                    shadow_offset_y = gr.Slider(label="阴影 Y 偏移 (px)", minimum=-10, maximum=10, value=0, step=1)
                    shadow_color = gr.ColorPicker(label="阴影颜色", value="#000000")


            # --- Image Options ---
            with gr.Group(visible=False) as image_options:
                 gr.Markdown("#### 图片水印选项")
                 image_file = gr.File(label="上传水印图片", file_types=['image'], type="filepath") # Use filepath
                 image_scale = gr.Slider(label="图片缩放比例 (%) - 相对于视频最大边长", minimum=1, maximum=50, value=10, step=1)

            # --- Motion Settings ---
            gr.Markdown("### 3. 运动设置")
            motion_path = gr.Dropdown(
                ["静态居中", "边框反弹", "水平移动", "垂直移动"],
                label="水印运动路径", value="边框反弹"
            )
            motion_margin = gr.Slider(
                label="运动区域边距 (%)", minimum=0, maximum=45, value=5, step=1
            )
            motion_speed = gr.Slider(
                label="运动速度倍率", minimum=0.1, maximum=10.0, value=1.0, step=0.1
            )

            # --- Output Settings ---
            gr.Markdown("### 4. 输出设置")
            output_dir_name = gr.Textbox(label="输出子目录名称", value=DEFAULT_OUTPUT_DIR)

        with gr.Column(scale=2): # Preview/Action column
            gr.Markdown("### 5. 水印预览")
            preview_button = gr.Button("生成预览 (使用第一个视频)")
            preview_output = gr.Image(label="效果预览", type="pil", interactive=False)

            gr.Markdown("### 6. 开始处理")
            process_button = gr.Button("🚀 开始添加水印 (多核处理)", variant="primary")

            gr.Markdown("### 状态日志")
            status_output = gr.Textbox(label="运行日志", lines=15, interactive=False, autoscroll=True)

            gr.Markdown("### 输出文件")
            output_files_display = gr.Files(label="生成的水印视频", interactive=False)


    # --- UI Logic ---
    def update_visibility(choice):
        is_text = (choice == "文本")
        return {
            text_options: gr.update(visible=is_text),
            image_options: gr.update(visible=not is_text),
        }
    watermark_type.change(update_visibility, inputs=watermark_type, outputs=[text_options, image_options])

    # Gather all inputs needed for preview and processing
    proc_inputs = [
            input_videos, watermark_type,
            # Text
            text_content, font_file, font_size, text_color,
            stroke_width, stroke_color, shadow_offset_x, shadow_offset_y, shadow_color,
            # Image
            image_file, image_scale,
            # Common
            watermark_opacity,
            # Motion
            motion_path, motion_margin, motion_speed,
            # Output
            output_dir_name
    ]
    preview_inputs = [ # Preview doesn't need output_dir_name
            input_videos, watermark_type,
            # Text
            text_content, font_file, font_size, text_color,
            stroke_width, stroke_color, shadow_offset_x, shadow_offset_y, shadow_color,
            # Image
            image_file, image_scale,
            # Common
            watermark_opacity,
            # Motion
            motion_path, motion_margin, motion_speed,
    ]

    # Connect Preview Button
    preview_button.click(
        fn=generate_preview,
        inputs=preview_inputs,
        outputs=[preview_output]
    )

    # Connect Process Button
    process_button.click(
        fn=process_videos_interface,
        inputs=proc_inputs,
        outputs=[status_output, output_files_display, preview_output] # Pass None to preview output
    )

    gr.Markdown("---")
    gr.Markdown(f"**提示:** 处理多个或长视频时，将自动使用多核 CPU 加速。确保有足够的磁盘空间。默认字体 '{DEFAULT_FONT_PATH}' {'存在' if default_font_exists else '未找到'}。")


# Launch the Gradio app
if __name__ == "__main__":
     # Required for multiprocessing pool to work correctly on some OS (like Windows)
    multiprocessing.freeze_support()

    if not default_font_exists:
        print(f"警告: 默认字体 '{DEFAULT_FONT_PATH}' 未找到。")
        print("除非通过界面上传字体文件，否则文本水印功能可能无法正常工作或显示异常。")

    print(f"系统 CPU 核心数: {os.cpu_count()}")
    print("正在启动 Gradio 界面...")
    demo.launch()