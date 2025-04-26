# -*- coding: utf-8 -*- # Add this for better encoding support
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import gradio as gr
import os
import time
import math
import random

# --- Configuration ---
# IMPORTANT: Replace with a valid path to a font file that supports Chinese characters if needed
# Or instruct users clearly they MUST upload one for Chinese text.
# SimHei is common on Windows, Arial Unicode MS, or Noto Sans CJK are good alternatives.
# Let's assume a generic name, user MUST provide if default doesn't exist/work.
DEFAULT_FONT_PATH = "msyh.ttc" # Example: Microsoft YaHei path (adjust for your system or use upload)
DEFAULT_OUTPUT_DIR = "output_videos"

# --- Helper Functions ---

def hex_to_rgba(hex_color, alpha=255):
    """
    Converts hex color string (#RRGGBB or #RGB) to (R, G, B, A) tuple.
    The 'alpha' parameter (0-255) determines the final transparency.
    """
    if not hex_color:
        # Default to white if color is empty
        print("Warning: Color is empty, defaulting to white.")
        hex_color = "#FFFFFF"
        # raise ValueError("颜色不能为空") # Original Chinese error

    hex_input = hex_color.lstrip('#')

    # Validate hex characters
    if not all(c in '0123456789ABCDEFabcdef' for c in hex_input):
         # Fallback for invalid hex? Default to white?
         print(f"警告：无效的十六进制颜色字符: {hex_color}，将使用白色。")
         hex_input = "FFFFFF"
         # raise ValueError(f"无效的十六进制颜色字符 (必须是 0-9, A-F): {hex_color}") # Original error

    # Validate hex length
    if len(hex_input) == 3:
        hex_input = ''.join([c*2 for c in hex_input])
    elif len(hex_input) != 6:
         # Fallback for invalid length? Default to white?
         print(f"警告：无效的十六进制颜色长度 (必须是 3 或 6 位): #{hex_input}，将使用白色。")
         hex_input = "FFFFFF"
        # raise ValueError(f"无效的十六进制颜色长度 (必须是 3 或 6 个字符): {hex_color}") # Original error

    try:
        r = int(hex_input[0:2], 16)
        g = int(hex_input[2:4], 16)
        b = int(hex_input[4:6], 16)
    except ValueError:
        # Handle potential conversion errors if validation somehow failed
        print(f"错误: 转换十六进制颜色失败: {hex_color}，将使用白色。")
        r, g, b = 255, 255, 255

    # Ensure alpha is within valid range
    final_alpha = max(0, min(255, int(alpha)))

    return (r, g, b, final_alpha)

def apply_opacity(img, opacity_percent):
    """Applies overall opacity to a PIL image (RGBA)."""
    if img.mode != 'RGBA':
        img = img.convert('RGBA')
    alpha = img.split()[3]
    # Adjust alpha: Scale existing alpha by the desired opacity percentage.
    # Ensure calculations use floats and result is clamped 0-255.
    opacity_factor = max(0.0, min(1.0, opacity_percent / 100.0))
    alpha = alpha.point(lambda p: max(0, min(255, int(p * opacity_factor))))
    img.putalpha(alpha)
    return img

# --- Watermark Generation ---

def create_text_watermark(text, font_path, font_size, color_hex, opacity_percent):
    """Creates a transparent PIL image with the specified text and opacity."""
    try:
        # Use default if path is None, empty, or doesn't exist and DEFAULT exists
        if not font_path or not os.path.exists(font_path):
             if os.path.exists(DEFAULT_FONT_PATH):
                 print(f"警告: 字体文件 '{font_path}' 未找到或未提供。使用默认字体: {DEFAULT_FONT_PATH}")
                 font_path = DEFAULT_FONT_PATH
             else:
                 print(f"错误: 字体文件 '{font_path}' 及默认字体 '{DEFAULT_FONT_PATH}' 均未找到。")
                 raise ValueError(f"未找到字体文件 '{font_path}' 或默认字体 '{DEFAULT_FONT_PATH}'。请上传字体或确保默认字体路径有效。")
        font = ImageFont.truetype(font_path, font_size)
        print(f"使用字体: {font_path}")
    except IOError as e:
        print(f"错误: 加载字体文件时出错 {font_path}: {e}。尝试 Pillow 默认字体。")
        try:
            font = ImageFont.load_default() # Very basic, likely no Chinese support
        except Exception as e_def:
             raise ValueError(f"无法加载指定字体 '{font_path}' 或 Pillow 默认字体。请提供有效的 .ttf 或 .otf 文件。错误: {e_def}")

    # Calculate text bounding box using the loaded font
    try:
        # Use getbbox for more accurate size calculation
        text_bbox = font.getbbox(text)
        # text_bbox is (left, top, right, bottom) relative to baseline origin (0,0)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        # Handle potential negative 'top' (descenders going below baseline)
        offset_y = -text_bbox[1] # Amount the text starts above the origin
        
        # Add padding for safety, especially around edges
        padding = 5
        img_width = text_width + 2 * padding
        img_height = text_height + 2 * padding

        # Create image: Use 0 alpha for transparent background
        img = Image.new('RGBA', (img_width, img_height), (255, 255, 255, 0))
        draw = ImageDraw.Draw(img)

        # Calculate color with desired base alpha (opacity)
        base_alpha = int(255 * (opacity_percent / 100.0)) # Opacity applies to the solid color
        text_color_rgba = hex_to_rgba(color_hex, alpha=base_alpha)

        # Draw text slightly offset within the padded area
        # Draw at (padding, padding + offset_y)
        draw.text((padding, padding + offset_y), text, font=font, fill=text_color_rgba)

        # Crop to the actual content bounds
        try:
             bbox = img.getbbox() # Find the boundary of non-transparent pixels
             if bbox:
                # Crop using the calculated bounding box
                img = img.crop(bbox)
             else: # Handle case where text might be empty or render fully transparent
                img = Image.new('RGBA', (1, 1), (255, 255, 255, 0)) # Minimal transparent image
        except Exception as crop_e:
             print(f"警告: 裁剪文本水印时出错: {crop_e}. 使用未裁剪图像。")
             pass # Keep the padded image if cropping fails

        # Double check size, ensure > 0
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

    # --- Scaling ---
    # Scale based on frame dimension and percentage
    base_dimension = max(frame_width, frame_height) # Scale relative to largest dimension
    target_width = int(base_dimension * (scale_percent / 100.0))

    # Calculate new height maintaining aspect ratio
    if img.width > 0:
        img_ratio = img.height / img.width
        target_height = int(target_width * img_ratio)
    else:
        target_height = int(base_dimension * (scale_percent / 100.0)) # Default if width is 0

    # Ensure target dimensions are at least 1x1
    target_width = max(1, target_width)
    target_height = max(1, target_height)

    # Resize using Pillow's high-quality downsampling filter
    try:
        img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)
    except AttributeError: # Handle older Pillow versions
         print("警告: Pillow 版本较旧，使用 ANTIALIAS 进行缩放。")
         img = img.resize((target_width, target_height), Image.ANTIALIAS)
    except Exception as resize_e:
         raise ValueError(f"缩放图片水印时出错: {resize_e}")


    # --- Apply Opacity ---
    img = apply_opacity(img, opacity_percent)

    return img

# --- Motion Calculation (WatermarkMover class remains largely the same) ---
class WatermarkMover:
    """计算水印位置基于选定的路径和区域。"""
    def __init__(self, path_type, frame_width, frame_height, watermark_width, watermark_height, margin_percent, speed_factor):
        self.path_type = path_type
        self.speed_factor = max(0.1, speed_factor) # 确保最小速度

        # Define motion boundaries based on margin
        margin_x = int(frame_width * (margin_percent / 100.0))
        margin_y = int(frame_height * (margin_percent / 100.0))

        self.min_x = margin_x
        self.min_y = margin_y
        # Adjust max bounds so the *entire* watermark stays within margin
        self.max_x = frame_width - margin_x - watermark_width
        self.max_y = frame_height - margin_y - watermark_height

        # Ensure bounds are valid (watermark isn't larger than allowed area)
        self.max_x = max(self.min_x, self.max_x)
        self.max_y = max(self.min_y, self.max_y)

        self.frame_width = frame_width
        self.frame_height = frame_height
        self.watermark_width = watermark_width
        self.watermark_height = watermark_height
        self.margin_percent = margin_percent # Store for potential re-init

        # Initial state for dynamic paths
        self.x = random.randint(self.min_x, self.max_x) if self.max_x > self.min_x else self.min_x
        self.y = random.randint(self.min_y, self.max_y) if self.max_y > self.min_y else self.min_y

        # Base speed (pixels per frame), adjust with speed_factor
        base_speed_x = max(1, int(frame_width * 0.005 * self.speed_factor))
        base_speed_y = max(1, int(frame_height * 0.005 * self.speed_factor))

        self.dx = random.choice([-base_speed_x, base_speed_x]) if base_speed_x > 0 else 0
        self.dy = random.choice([-base_speed_y, base_speed_y]) if base_speed_y > 0 else 0
        
        # Ensure dx/dy are not both zero if bounds allow movement
        if self.dx == 0 and self.dy == 0 and (self.max_x > self.min_x or self.max_y > self.min_y):
             self.dx = base_speed_x if self.max_x > self.min_x else 0
             self.dy = base_speed_y if self.max_y > self.min_y and self.dx == 0 else 0 # Prefer horizontal if possible

        # Static position (center)
        self.static_x = max(0, (frame_width - watermark_width) // 2)
        self.static_y = max(0, (frame_height - watermark_height) // 2)


    def get_position(self, frame_index, total_frames):
        """计算当前帧的位置。"""
        if self.path_type == "静态居中":
            return self.static_x, self.static_y

        elif self.path_type == "边框反弹":
            # Prevent movement if bounds are collapsed
            if self.max_x <= self.min_x: self.dx = 0
            if self.max_y <= self.min_y: self.dy = 0

            self.x += self.dx
            self.y += self.dy

            # Bounce off edges
            bounced = False
            if self.x <= self.min_x:
                self.x = self.min_x
                self.dx = abs(self.dx) if self.dx != 0 else (random.choice([1,-1]) if self.max_x > self.min_x else 0) # Introduce randomness if stuck
                bounced = True
            elif self.x >= self.max_x:
                self.x = self.max_x
                self.dx = -abs(self.dx) if self.dx != 0 else (random.choice([1,-1]) if self.max_x > self.min_x else 0)
                bounced = True

            if self.y <= self.min_y:
                self.y = self.min_y
                self.dy = abs(self.dy) if self.dy != 0 else (random.choice([1,-1]) if self.max_y > self.min_y else 0)
                bounced = True
            elif self.y >= self.max_y:
                self.y = self.max_y
                self.dy = -abs(self.dy) if self.dy != 0 else (random.choice([1,-1]) if self.max_y > self.min_y else 0)
                bounced = True

            # Ensure position stays within bounds if speed is high
            self.x = max(self.min_x, min(self.x, self.max_x))
            self.y = max(self.min_y, min(self.y, self.max_y))

            return int(self.x), int(self.y)

        elif self.path_type == "水平移动":
            # Simple back and forth horizontally
            if total_frames <= 0: total_frames = 1 # Avoid division by zero
            progress = (frame_index % total_frames) / total_frames # Cycle progress
            amplitude = (self.max_x - self.min_x) / 2
            center_x = self.min_x + amplitude
            # Use sine wave for smooth oscillation, adjust frequency with speed
            frequency = 1.0 * self.speed_factor # Base cycles per video * speed factor
            current_x = center_x + amplitude * math.sin(2 * math.pi * frequency * progress)
            # Fixed vertical position (center of allowed area)
            current_y = self.min_y + (self.max_y - self.min_y) / 2

            # Ensure within bounds
            current_x = max(self.min_x, min(current_x, self.max_x))
            current_y = max(self.min_y, min(current_y, self.max_y))

            return int(current_x), int(current_y)

        elif self.path_type == "垂直移动":
            # Simple back and forth vertically
            if total_frames <= 0: total_frames = 1
            progress = (frame_index % total_frames) / total_frames
            amplitude = (self.max_y - self.min_y) / 2
            center_y = self.min_y + amplitude
            frequency = 1.0 * self.speed_factor
            current_y = center_y + amplitude * math.sin(2 * math.pi * frequency * progress)
            # Fixed horizontal position (center of allowed area)
            current_x = self.min_x + (self.max_x - self.min_x) / 2

            # Ensure within bounds
            current_x = max(self.min_x, min(current_x, self.max_x))
            current_y = max(self.min_y, min(current_y, self.max_y))

            return int(current_x), int(current_y)

        else: # Default to static if path unknown
            print(f"警告: 未知的移动路径 '{self.path_type}'，将使用静态居中。")
            return self.static_x, self.static_y

# --- Core Video Processing ---

def add_watermark_to_video(input_video_path, output_video_path, watermark_pil, mover, progress=gr.Progress()):
    """将准备好的水印 PIL 图像添加到视频的每一帧。"""
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise IOError(f"错误: 打开视频文件失败: {input_video_path}")

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Validate properties
    if not all([fps > 0, frame_width > 0, frame_height > 0, total_frames >= 0]): # total_frames can be 0 for streams
         cap.release()
         raise ValueError(f"视频属性无效或无法读取: FPS={fps}, 宽度={frame_width}, 高度={frame_height}, 总帧数={total_frames}")

    # Ensure watermark fits within the frame (can happen if margin is 0 and watermark is large)
    if watermark_pil.width > frame_width or watermark_pil.height > frame_height:
         print(f"警告: 水印尺寸 ({watermark_pil.width}x{watermark_pil.height}) 超出帧尺寸 ({frame_width}x{frame_height})。将调整水印大小以适应。")
         # Resize watermark to fit frame while maintaining aspect ratio
         ratio = min(frame_width / watermark_pil.width, frame_height / watermark_pil.height)
         new_w = max(1, int(watermark_pil.width * ratio))
         new_h = max(1, int(watermark_pil.height * ratio))
         try:
            print(f"  调整水印大小为: {new_w}x{new_h}")
            watermark_pil = watermark_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
         except AttributeError:
            watermark_pil = watermark_pil.resize((new_w, new_h), Image.ANTIALIAS)
         except Exception as resize_err:
             print(f"  调整水印大小时出错: {resize_err}. 继续使用原始（可能裁剪的）水印。")
             # Keep original watermark_pil but it might get clipped.

         # Re-initialize mover with new watermark dimensions as bounds depend on it
         print("由于水印尺寸调整，重新初始化水印移动器。")
         # Need mover's original parameters
         mover.__init__(mover.path_type, frame_width, frame_height, new_w, new_h, mover.margin_percent, mover.speed_factor)


    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Common codec for .mp4
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    if not out.isOpened():
        cap.release()
        raise IOError(f"错误: 无法创建输出视频文件: {output_video_path}. 检查编解码器或路径权限。")

    frame_count = 0
    start_time = time.time()

    # Initialize progress tracking for Gradio
    progress(0, desc=f"开始处理 {os.path.basename(input_video_path)}")

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break # End of video or error reading frame

        # --- Apply watermark using Pillow ---
        # 1. Convert OpenCV frame (BGR) to PIL image (RGBA) for compositing
        try:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil_frame = Image.fromarray(frame_rgb).convert("RGBA")
        except Exception as convert_e:
            print(f"错误: 在帧 {frame_count} 转换 BGR 到 RGBA 失败: {convert_e}. 跳过此帧水印。")
            out.write(frame_bgr) # Write original frame on error
            frame_count += 1
            continue


        # 2. Get watermark position for this frame
        wm_x, wm_y = mover.get_position(frame_count, total_frames if total_frames > 0 else 1) # Avoid zero total_frames in mover

        # Ensure position is integer and clamped
        wm_x = max(0, min(int(wm_x), frame_width - watermark_pil.width))
        wm_y = max(0, min(int(wm_y), frame_height - watermark_pil.height))

        # 3. Paste watermark onto the frame using alpha compositing
        try:
            # Create a temporary RGBA layer for pasting, same size as frame, fully transparent
            # This seems unnecessary if pil_frame is already RGBA. Let's composite directly.
            # rgba_layer = Image.new('RGBA', pil_frame.size, (0,0,0,0))
            # rgba_layer.paste(watermark_pil, (wm_x, wm_y), mask=watermark_pil) # Use watermark alpha as mask
            # pil_frame_out = Image.alpha_composite(pil_frame, rgba_layer)

            # Simpler approach: Create a copy to paste onto if needed, or paste directly
            # Check if watermark has actual content (alpha > 0)
            if watermark_pil.getextrema()[3][0] < 255: # Check if any pixel is not fully transparent
                 # Create a transparent overlay the size of the frame
                 overlay = Image.new('RGBA', pil_frame.size, (255, 255, 255, 0))
                 # Paste the watermark onto the overlay
                 overlay.paste(watermark_pil, (wm_x, wm_y), mask=watermark_pil)
                 # Composite the overlay (with watermark) onto the frame
                 pil_frame_out = Image.alpha_composite(pil_frame, overlay)
            else:
                 # Watermark seems fully transparent, skip compositing
                 pil_frame_out = pil_frame

        except Exception as e:
            print(f"错误: 在帧 {frame_count} 粘贴水印时出错: {e}. 位置:({wm_x},{wm_y}), 水印尺寸: {watermark_pil.size}, 帧尺寸: {pil_frame.size}. 跳过此帧水印。")
            pil_frame_out = pil_frame # Skip pasting on error for this frame


        # 4. Convert PIL image (RGBA) back to OpenCV frame (BGR)
        try:
            # Need to convert back to RGB first before converting color space
            pil_frame_out_rgb = pil_frame_out.convert("RGB")
            frame_out_bgr = cv2.cvtColor(np.array(pil_frame_out_rgb), cv2.COLOR_RGB2BGR)
        except Exception as convert_back_e:
             print(f"错误: 在帧 {frame_count} 转换 RGBA 回 BGR 失败: {convert_back_e}. 跳过此帧。")
             # Decide how to handle: write original? skip frame?
             # Let's write the original BGR frame to avoid losing frames
             out.write(frame_bgr)
             frame_count += 1
             continue

        # --- Write the frame ---
        out.write(frame_out_bgr)

        frame_count += 1
        # Update Gradio progress
        if total_frames > 0:
             progress(frame_count / total_frames, desc=f"处理中 {frame_count}/{total_frames}")
        else:
             # Estimate progress based on time for streams? Or just show frame count.
             if frame_count % 30 == 0: # Update every second assuming ~30fps
                 elapsed = time.time() - start_time
                 progress(frame_count / (frame_count + 1), desc=f"处理中: {frame_count} 帧, {elapsed:.1f} 秒")


    # Release everything
    cap.release()
    out.release()
    # cv2.destroyAllWindows() # Not needed in script context

    end_time = time.time()
    duration = end_time - start_time
    print(f"完成处理 {input_video_path}。耗时: {duration:.2f} 秒")
    progress(1.0, desc=f"完成 {os.path.basename(input_video_path)}") # Final progress update


# --- Gradio Interface Function ---

def process_videos_interface(
    input_videos, # List of file paths from gr.Files
    watermark_type,
    text_content,
    font_file, # File object from gr.File
    font_size,
    text_color,
    image_file, # File object from gr.File
    image_scale,
    watermark_opacity, # Added opacity parameter
    motion_path,
    motion_margin,
    motion_speed,
    output_dir_name,
    progress=gr.Progress(track_tqdm=True)): # Add progress tracking

    if not input_videos:
        return "错误: 未选择任何输入视频文件。", [], None # Added None for preview output

    # Create output directory
    output_base = output_dir_name if output_dir_name else DEFAULT_OUTPUT_DIR
    if not os.path.exists(output_base):
        try:
            os.makedirs(output_base)
            print(f"创建输出目录: {output_base}")
        except OSError as e:
            return f"错误: 无法创建输出目录 '{output_base}': {e}", [], None

    status_messages = []
    output_file_paths = []

    # --- Font File Handling ---
    final_font_path = None
    if watermark_type == "文本":
        if font_file is not None:
            # Gradio File component gives a TemporaryFile object with a .name attribute for path
            final_font_path = font_file.name
            status_messages.append(f"使用上传的字体: {os.path.basename(final_font_path)}")
            if not os.path.exists(final_font_path):
                 status_messages.append(f"警告: 上传的字体文件路径 '{final_font_path}' 似乎无效。尝试使用默认字体。")
                 final_font_path = None # Fallback to check default
        else:
            status_messages.append(f"未提供字体文件。尝试使用默认字体: {DEFAULT_FONT_PATH}")

        # If no specific font path or it was invalid, try the default
        if final_font_path is None:
             if os.path.exists(DEFAULT_FONT_PATH):
                 final_font_path = DEFAULT_FONT_PATH
                 status_messages.append(f"使用默认字体: {DEFAULT_FONT_PATH}")
             else:
                 error_msg = f"错误: 文本水印需要字体，但既未提供有效字体，也找不到默认字体 '{DEFAULT_FONT_PATH}'。"
                 status_messages.append(error_msg)
                 return "\n".join(status_messages), [], None

    # --- Image File Handling ---
    image_watermark_path = None
    if watermark_type == "图片":
        if image_file is not None:
            image_watermark_path = image_file.name
            status_messages.append(f"使用上传的图片: {os.path.basename(image_watermark_path)}")
            if not os.path.exists(image_watermark_path):
                 status_messages.append(f"错误: 上传的图片文件路径 '{image_watermark_path}' 似乎无效。")
                 return "\n".join(status_messages), [], None
        else:
             status_messages.append(f"错误: 选择了图片水印类型，但未提供图片文件。")
             return "\n".join(status_messages), [], None


    # --- Process Each Video ---
    total_videos = len(input_videos)
    for i, video_file in enumerate(input_videos):
        input_path = video_file.name # Path from Gradio's TemporaryFile object
        base_filename = os.path.basename(input_path)
        filename_no_ext, _ = os.path.splitext(base_filename)
        # Sanitize filename slightly (replace spaces, etc.) if needed
        safe_filename = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in filename_no_ext)
        output_filename = f"{safe_filename}_watermarked.mp4"
        output_path = os.path.join(output_base, output_filename)

        status_messages.append(f"\n[{i+1}/{total_videos}] 开始处理: {base_filename} -> {output_filename}")
        print(f"开始处理视频 {i+1}/{total_videos}: {input_path}")

        try:
            # --- Get Video Dimensions for Watermark Prep ---
            temp_cap = cv2.VideoCapture(input_path)
            if not temp_cap.isOpened():
                raise IOError(f"无法打开视频 '{base_filename}' 以获取尺寸。")
            vid_width = int(temp_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            vid_height = int(temp_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_vid_frames = int(temp_cap.get(cv2.CAP_PROP_FRAME_COUNT)) # For mover
            temp_cap.release()

            if vid_width <= 0 or vid_height <= 0:
                 raise ValueError(f"获取到的视频尺寸无效: {vid_width}x{vid_height}")

            # --- Prepare Watermark (Specific to this video's dimensions) ---
            watermark_pil = None
            if watermark_type == "文本":
                status_messages.append(f"  创建文本水印: '{text_content[:30]}...' 尺寸:{font_size}pt 透明度:{watermark_opacity}%")
                watermark_pil = create_text_watermark(
                    text_content, final_font_path, font_size, text_color, watermark_opacity
                )
            elif watermark_type == "图片":
                 status_messages.append(f"  加载图片水印: 缩放:{image_scale}% 透明度:{watermark_opacity}%")
                 watermark_pil = load_image_watermark(
                    image_watermark_path, image_scale, vid_width, vid_height, watermark_opacity
                 )
            else:
                # This case should not be reachable if UI is set up correctly
                raise ValueError(f"内部错误：未知的水印类型 '{watermark_type}'")

            if watermark_pil is None or watermark_pil.width <= 0 or watermark_pil.height <= 0:
                raise ValueError("水印创建失败或尺寸无效。")
            status_messages.append(f"  水印已生成，尺寸: {watermark_pil.width}x{watermark_pil.height}")

            # --- Initialize Mover ---
            mover = WatermarkMover(
                path_type=motion_path,
                frame_width=vid_width,
                frame_height=vid_height,
                watermark_width=watermark_pil.width,
                watermark_height=watermark_pil.height,
                margin_percent=motion_margin,
                speed_factor=motion_speed
            )
            status_messages.append(f"  初始化移动器: 类型:{motion_path}, 边距:{motion_margin}%, 速度:{motion_speed}x")

            # --- Add Watermark to Video ---
            add_watermark_to_video(input_path, output_path, watermark_pil, mover, progress)

            status_messages.append(f"  成功处理并保存到: {output_path}")
            output_file_paths.append(output_path)

        except Exception as e:
            error_msg = f"  错误 处理 {base_filename} 时发生错误: {type(e).__name__}: {e}"
            print(error_msg) # Print detailed error to console
            status_messages.append(error_msg)
            # Clean up partially created output file if it exists
            if os.path.exists(output_path):
                 try:
                     os.remove(output_path)
                     status_messages.append(f"  已删除部分生成的输出文件: {output_path}")
                 except OSError as rm_err:
                     status_messages.append(f"  警告: 无法删除部分文件 {output_path}: {rm_err}")

    status_messages.append("\n批量处理完成。")
    # Return status log and list of output file paths
    return "\n".join(status_messages), output_file_paths, None # No preview update from main process


# --- Preview Function ---
def generate_preview(
    input_videos, # Need this to get a frame
    watermark_type,
    text_content,
    font_file,
    font_size,
    text_color,
    image_file,
    image_scale,
    watermark_opacity,
    motion_path,
    motion_margin,
    motion_speed):

    if not input_videos:
        # Return a placeholder or message image if no video selected
        # Create a simple black image with text
        img = Image.new('RGB', (300, 100), color = 'black')
        d = ImageDraw.Draw(img)
        try:
            # Try using a basic font if available
            font = ImageFont.truetype("arial.ttf", 15) # Adjust path/size if needed
        except IOError:
            font = ImageFont.load_default()
        d.text((10,10), "请先上传一个视频文件\n才能生成预览", fill='white', font=font)
        return img # Return PIL image directly

    preview_status = ""
    try:
        input_path = input_videos[0].name # Use the first video
        preview_status += f"使用视频 '{os.path.basename(input_path)}' 的第一帧进行预览。\n"

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise IOError(f"无法打开视频 '{os.path.basename(input_path)}' 进行预览。")

        # Read the first frame (or maybe a few frames in?)
        frame_to_preview = 10 # Try frame 10 to skip potential black intro
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_to_preview)
        ret, frame_bgr = cap.read()
        if not ret:
             # If frame 10 fails, try frame 0
             cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
             ret, frame_bgr = cap.read()
             if not ret:
                 raise ValueError("无法读取视频帧进行预览。")
             frame_to_preview = 0 # Used frame 0

        vid_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vid_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # Needed for mover % calc
        cap.release()

        if vid_width <= 0 or vid_height <= 0:
            raise ValueError("预览时获取的视频尺寸无效。")

        # --- Prepare Watermark using current settings ---
        watermark_pil = None
        final_font_path_preview = None # Separate font path logic for preview
        if watermark_type == "文本":
            if font_file is not None:
                 final_font_path_preview = font_file.name
                 if not os.path.exists(final_font_path_preview): final_font_path_preview = None
            if final_font_path_preview is None and os.path.exists(DEFAULT_FONT_PATH):
                 final_font_path_preview = DEFAULT_FONT_PATH
            if final_font_path_preview is None:
                 raise ValueError("预览需要字体，但未提供或找不到默认字体。")

            watermark_pil = create_text_watermark(
                text_content, final_font_path_preview, font_size, text_color, watermark_opacity
            )
            preview_status += f"创建文本水印 (字体: {os.path.basename(final_font_path_preview)}, 尺寸: {font_size}pt, 透明度: {watermark_opacity}%)\n"

        elif watermark_type == "图片":
            if image_file is None or not os.path.exists(image_file.name):
                raise ValueError("预览需要图片，但未提供或文件无效。")
            image_watermark_path_preview = image_file.name
            watermark_pil = load_image_watermark(
                image_watermark_path_preview, image_scale, vid_width, vid_height, watermark_opacity
            )
            preview_status += f"加载图片水印 (文件: {os.path.basename(image_watermark_path_preview)}, 缩放: {image_scale}%, 透明度: {watermark_opacity}%)\n"
        else:
             raise ValueError("无效的水印类型。")

        if watermark_pil is None or watermark_pil.width <= 0 or watermark_pil.height <= 0:
             raise ValueError("预览时水印创建失败或尺寸无效。")
        preview_status += f"水印尺寸: {watermark_pil.width}x{watermark_pil.height}\n"

        # --- Initialize Mover ---
        mover = WatermarkMover(
            path_type=motion_path,
            frame_width=vid_width,
            frame_height=vid_height,
            watermark_width=watermark_pil.width,
            watermark_height=watermark_pil.height,
            margin_percent=motion_margin,
            speed_factor=motion_speed
        )

        # --- Get Position for the Preview Frame ---
        # Use the frame number we actually read
        wm_x, wm_y = mover.get_position(frame_to_preview, total_frames if total_frames > 0 else 1)
        wm_x = max(0, min(int(wm_x), vid_width - watermark_pil.width))
        wm_y = max(0, min(int(wm_y), vid_height - watermark_pil.height))
        preview_status += f"水印位置 (帧 {frame_to_preview}): ({wm_x}, {wm_y})\n"

        # --- Composite onto Frame ---
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_frame = Image.fromarray(frame_rgb).convert("RGBA")

        # Composite (similar to add_watermark_to_video)
        if watermark_pil.getextrema()[3][0] < 255: # Check alpha channel
            overlay = Image.new('RGBA', pil_frame.size, (255, 255, 255, 0))
            overlay.paste(watermark_pil, (wm_x, wm_y), mask=watermark_pil)
            pil_frame_out = Image.alpha_composite(pil_frame, overlay)
        else:
            pil_frame_out = pil_frame

        preview_status += "预览图生成成功。"
        print(preview_status) # Print status to console
        return pil_frame_out.convert("RGB") # Return RGB PIL image for Gradio

    except Exception as e:
        error_msg = f"生成预览时出错: {type(e).__name__}: {e}"
        print(error_msg)
        # Create an error image
        img = Image.new('RGB', (400, 150), color = 'darkred')
        d = ImageDraw.Draw(img)
        try: font = ImageFont.truetype("arial.ttf", 15)
        except IOError: font = ImageFont.load_default()
        d.text((10,10), f"生成预览失败:\n{error_msg}", fill='white', font=font)
        return img


# --- Build Gradio UI (with Chinese text and new elements) ---

# Check for default font existence early
default_font_exists = os.path.exists(DEFAULT_FONT_PATH)
default_font_warning = f" (默认: {os.path.basename(DEFAULT_FONT_PATH)}{'' if default_font_exists else ' - 未找到!'})"

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎬 视频水印添加工具")
    gr.Markdown("为您的视频添加动态文本或图片水印，支持多种移动路径和自定义设置。")

    with gr.Row():
        with gr.Column(scale=2): # Wider column for inputs
            gr.Markdown("### 1. 输入视频")
            input_videos = gr.Files(label="上传视频文件 (可多选)", file_types=['video'], type="filepath")

            gr.Markdown("### 2. 水印设置")
            watermark_type = gr.Radio(["文本", "图片"], label="水印类型", value="文本")

            # --- Common Settings ---
            watermark_opacity = gr.Slider(label="水印不透明度 (%)", minimum=0, maximum=100, value=70, step=1)

            # --- Text Options ---
            with gr.Group(visible=True) as text_options: # Use Group for better visual separation
                gr.Markdown("#### 文本水印选项")
                text_content = gr.Textbox(label="水印文字", value="在此输入水印内容", lines=2)
                font_file = gr.File(label=f"上传字体文件 (.ttf, .otf) {default_font_warning}", file_types=['.ttf', '.otf'])
                font_size = gr.Slider(label="字体大小 (pt)", minimum=8, maximum=300, value=48, step=1)
                text_color = gr.ColorPicker(label="文字颜色 (Hex)", value="#FFFFFF")

            # --- Image Options ---
            with gr.Group(visible=False) as image_options: # Use Group
                 gr.Markdown("#### 图片水印选项")
                 image_file = gr.File(label="上传水印图片", file_types=['image'], type="filepath")
                 image_scale = gr.Slider(label="图片缩放比例 (%) - 相对于视频最大边长", minimum=1, maximum=50, value=10, step=1)

            # --- Motion Settings ---
            gr.Markdown("### 3. 运动设置")
            motion_path = gr.Dropdown(
                ["静态居中", "边框反弹", "水平移动", "垂直移动"],
                label="水印运动路径",
                value="边框反弹"
            )
            motion_margin = gr.Slider(
                label="运动区域边距 (%) - 水印活动范围距离视频边缘的百分比",
                minimum=0, maximum=45, value=5, step=1 # Max 45% margin each side prevents overlap
            )
            motion_speed = gr.Slider(
                label="运动速度倍率",
                minimum=0.1, maximum=10.0, value=1.0, step=0.1
            )

            # --- Output Settings ---
            gr.Markdown("### 4. 输出设置")
            output_dir_name = gr.Textbox(label="输出子目录名称", value=DEFAULT_OUTPUT_DIR)

        with gr.Column(scale=1): # Narrower column for preview and actions
            gr.Markdown("### 5. 水印预览")
            preview_button = gr.Button("生成预览 (使用第一个视频)")
            preview_output = gr.Image(label="效果预览", type="pil", interactive=False) # Show PIL image

            gr.Markdown("### 6. 开始处理")
            process_button = gr.Button("开始添加水印", variant="primary")

            gr.Markdown("### 状态日志")
            status_output = gr.Textbox(label="运行日志", lines=15, interactive=False)

            gr.Markdown("### 输出文件")
            # Use gr.Files to display multiple output files for download
            output_files_display = gr.Files(label="生成的水印视频", interactive=False)


    # --- UI Logic ---
    # Visibility toggle for text/image options
    def update_visibility(choice):
        is_text = (choice == "文本")
        return {
            text_options: gr.update(visible=is_text),
            image_options: gr.update(visible=not is_text),
        }
    watermark_type.change(update_visibility, inputs=watermark_type, outputs=[text_options, image_options])

    # Gather all inputs needed for preview and processing
    all_inputs = [
            input_videos, watermark_type,
            text_content, font_file, font_size, text_color, # Text inputs
            image_file, image_scale, # Image inputs
            watermark_opacity, # Opacity input
            motion_path, motion_margin, motion_speed, # Motion inputs
            output_dir_name # Output dir (needed by main process, not preview)
    ]
    # Define inputs specifically for preview (doesn't need output_dir_name)
    preview_inputs = [
            input_videos, watermark_type,
            text_content, font_file, font_size, text_color,
            image_file, image_scale, watermark_opacity,
            motion_path, motion_margin, motion_speed
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
        inputs=all_inputs,
        outputs=[status_output, output_files_display, preview_output] # Clear preview on process start? Or keep it? Let's keep it, but pass None from process func.
    )

    gr.Markdown("---")
    gr.Markdown(f"**注意:** 处理长视频可能需要较长时间。对于文本水印，如果默认字体 '{DEFAULT_FONT_PATH}' 不存在或不支持所需字符 (如中文), 请务必上传您自己的字体文件。输出视频将保存在脚本所在目录下的 '{DEFAULT_OUTPUT_DIR}' (或您指定的) 子目录中。")


# Launch the Gradio app
if __name__ == "__main__":
    # Check default font again on startup for final warning
    if not default_font_exists:
        print(f"警告: 默认字体 '{DEFAULT_FONT_PATH}' 未找到。")
        print("除非通过界面上传字体文件，否则文本水印功能可能无法正常工作或显示异常。")

    print("正在启动 Gradio 界面...")
    # Share=True allows access over local network if needed
    demo.launch()