import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import gradio as gr
import os
import time
import math
import random

# --- Configuration ---
DEFAULT_FONT_PATH = "ali.ttf" # IMPORTANT: Replace with a valid path to your font file or expect user input
DEFAULT_OUTPUT_DIR = "output_videos"

# --- Watermark Generation ---

def hex_to_rgba(hex_color, alpha=255):
    """Converts hex color string to (R, G, B, A) tuple."""
    if not hex_color.startswith('#'):
        raise ValueError("Hex color must start with #")
        
    hex_color = hex_color.lstrip('#')
    
    # Validate length and characters
    if len(hex_color) not in (3, 6):
        raise ValueError(f"Invalid hex color length (must be 3 or 6 chars after #): {hex_color}")
        
    if not all(c in '0123456789ABCDEFabcdef' for c in hex_color):
        raise ValueError(f"Invalid hex characters (must be 0-9, A-F): {hex_color}")

    if len(hex_color) == 3:
        hex_color = ''.join([c*2 for c in hex_color])
        
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return (r, g, b, alpha)

def create_text_watermark(text, font_path, font_size, color_hex):
    """Creates a transparent PIL image with the specified text."""
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        print(f"Error: Font file not found at {font_path}. Using default Pillow font.")
        try:
            # Try loading a basic default if primary fails (limited characters)
            font = ImageFont.load_default()
            # Estimate size needed (less accurate)
            # Use simple estimation or render once to get size - simpler approach here:
            # Render small text to estimate character width/height roughly
            bbox = font.getbbox("Ag") 
            estimated_width = len(text) * (bbox[2] - bbox[0]) 
            estimated_height = bbox[3] - bbox[1] + 10 # Add padding
        except Exception as e:
             raise ValueError(f"Could not load specified font '{font_path}' or default font. Please provide a valid .ttf or .otf file. Error: {e}")
        
    # Calculate text size more accurately using getbbox if font loaded
    if font_path and os.path.exists(font_path):
         # Use getbbox for better size calculation with the actual font
         try:
             # Get bounding box for the text
             text_bbox = font.getbbox(text)
             text_width = text_bbox[2] - text_bbox[0]
             text_height = text_bbox[3] - text_bbox[1]
             # Add some padding if needed, Pillow rendering might clip edges slightly sometimes
             padding = 5 
             img_width = text_width + 2 * padding
             img_height = text_height + 2 * padding
             # Create image larger than text to draw onto
             img = Image.new('RGBA', (img_width, img_height), (255, 255, 255, 0))
             draw = ImageDraw.Draw(img)
             text_color_rgba = hex_to_rgba(color_hex)
             # Draw text at position (padding, padding - text_bbox[1]) to align baseline
             draw.text((padding, padding - text_bbox[1]), text, font=font, fill=text_color_rgba)
             # Crop the image to the actual text bounds + padding for minimal size
             # Use getmask to find the non-transparent bounding box
             try:
                 bbox = img.getbbox()
                 if bbox:
                    img = img.crop(bbox)
                 else: # Handle case where text might be empty or render fully transparent
                    img = Image.new('RGBA', (1, 1), (255, 255, 255, 0)) # Minimal transparent image
             except Exception: # Fallback if getbbox fails
                 pass # Keep the padded image
             return img

         except Exception as e:
             print(f"Warning: Error calculating text bounding box: {e}. Using estimated size.")
             # Fallback to estimation if getbbox fails
             estimated_width = len(text) * font_size // 2 # Rough estimate
             estimated_height = font_size + 10
             img_width = estimated_width
             img_height = estimated_height

    else: # Default font case (already estimated size above)
        img_width = estimated_width
        img_height = estimated_height
        
    # Create image and draw text (used for default font or if getbbox failed)
    img = Image.new('RGBA', (img_width, img_height), (255, 255, 255, 0)) # Transparent background
    draw = ImageDraw.Draw(img)
    text_color_rgba = hex_to_rgba(color_hex)
    
    # Draw text at (0, 0) or centered depending on preference
    # For default font, (0,0) is simpler as baseline/bbox is less predictable
    draw.text((0, 0), text, font=font, fill=text_color_rgba) 

    # Try to crop whitespace if possible (less reliable for default font)
    try:
        bbox = img.getbbox()
        if bbox:
            img = img.crop(bbox)
        elif img.width == 0 or img.height == 0: # Handle empty image case
             img = Image.new('RGBA', (1, 1), (255, 255, 255, 0))
    except Exception:
        pass # Ignore cropping errors with default font

    return img


def load_image_watermark(image_path, scale_percent, frame_width, frame_height):
    """Loads and scales an image watermark."""
    try:
        img = Image.open(image_path).convert("RGBA")
    except FileNotFoundError:
        raise ValueError(f"Image watermark file not found: {image_path}")
    except Exception as e:
        raise ValueError(f"Error opening image watermark {image_path}: {e}")

    # Scale based on frame dimension and percentage
    base_dimension = max(frame_width, frame_height) # Scale relative to largest dimension
    target_width = int(base_dimension * (scale_percent / 100.0))

    # Calculate new height maintaining aspect ratio
    img_ratio = img.height / img.width
    target_height = int(target_width * img_ratio)

    # Ensure target dimensions are at least 1x1
    target_width = max(1, target_width)
    target_height = max(1, target_height)
    
    # Resize using Pillow's high-quality downsampling filter if possible
    try:
        img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)
    except AttributeError: # Handle older Pillow versions
         img = img.resize((target_width, target_height), Image.ANTIALIAS)
         
    return img

# --- Motion Calculation ---

class WatermarkMover:
    """Calculates watermark position based on selected path and region."""
    def __init__(self, path_type, frame_width, frame_height, watermark_width, watermark_height, margin_percent, speed_factor):
        self.path_type = path_type
        self.speed_factor = max(0.1, speed_factor) # Ensure minimum speed

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

        # Initial state for dynamic paths
        self.x = random.randint(self.min_x, self.max_x) if self.max_x > self.min_x else self.min_x
        self.y = random.randint(self.min_y, self.max_y) if self.max_y > self.min_y else self.min_y

        # Base speed (pixels per frame), adjust with speed_factor
        base_speed_x = max(1, int(frame_width * 0.005 * self.speed_factor)) 
        base_speed_y = max(1, int(frame_height * 0.005 * self.speed_factor))
        
        self.dx = random.choice([-base_speed_x, base_speed_x])
        self.dy = random.choice([-base_speed_y, base_speed_y])
        
        # Static position (center)
        self.static_x = max(0, (frame_width - watermark_width) // 2)
        self.static_y = max(0, (frame_height - watermark_height) // 2)


    def get_position(self, frame_index, total_frames):
        """Calculate position for the current frame."""
        if self.path_type == "Static":
            return self.static_x, self.static_y

        elif self.path_type == "Bouncing Box":
            self.x += self.dx
            self.y += self.dy

            # Bounce off edges
            if self.x <= self.min_x:
                self.x = self.min_x
                self.dx = abs(self.dx) # Ensure positive direction
            elif self.x >= self.max_x:
                self.x = self.max_x
                self.dx = -abs(self.dx) # Ensure negative direction

            if self.y <= self.min_y:
                self.y = self.min_y
                self.dy = abs(self.dy) # Ensure positive direction
            elif self.y >= self.max_y:
                self.y = self.max_y
                self.dy = -abs(self.dy) # Ensure negative direction
                
            # Ensure position stays within bounds if speed is high
            self.x = max(self.min_x, min(self.x, self.max_x))
            self.y = max(self.min_y, min(self.y, self.max_y))

            return int(self.x), int(self.y)

        elif self.path_type == "Horizontal":
            # Simple back and forth horizontally
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

        elif self.path_type == "Vertical":
            # Simple back and forth vertically
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
            return self.static_x, self.static_y

# --- Core Video Processing ---

def add_watermark_to_video(input_video_path, output_video_path, watermark_pil, mover):
    """Adds the prepared watermark PIL image to each frame of the video."""
    
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise IOError(f"Error opening video file: {input_video_path}")

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Ensure watermark fits within the frame (can happen if margin is 0 and watermark is large)
    if watermark_pil.width > frame_width or watermark_pil.height > frame_height:
         print(f"Warning: Watermark dimensions ({watermark_pil.width}x{watermark_pil.height}) exceed frame dimensions ({frame_width}x{frame_height}). Resizing watermark to fit.")
         # Resize watermark to fit frame while maintaining aspect ratio
         ratio = min(frame_width / watermark_pil.width, frame_height / watermark_pil.height)
         new_w = int(watermark_pil.width * ratio)
         new_h = int(watermark_pil.height * ratio)
         try:
            watermark_pil = watermark_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
         except AttributeError:
            watermark_pil = watermark_pil.resize((new_w, new_h), Image.ANTIALIAS)
         # Update mover if dimensions changed significantly? Re-init is safer.
         # This assumes mover was already initialized with potentially large wm size,
         # which might lead to incorrect bounds. Re-initializing mover here might be best.
         print("Re-initializing mover due to watermark resize.")
         mover.__init__(mover.path_type, frame_width, frame_height, new_w, new_h, mover.margin_percent, mover.speed_factor)


    # Define the codec and create VideoWriter object
    # Using 'mp4v' for MP4 output. Alternatives: 'XVID', 'MJPG' (for AVI)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    frame_count = 0
    start_time = time.time()

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break # End of video

        # --- Apply watermark using Pillow ---
        # 1. Convert OpenCV frame (BGR) to PIL image (RGBA)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_frame = Image.fromarray(frame_rgb).convert("RGBA")

        # 2. Get watermark position for this frame
        wm_x, wm_y = mover.get_position(frame_count, total_frames)
        
        # Ensure position is integer
        wm_x, wm_y = int(wm_x), int(wm_y)

        # Clamp position just in case calculation slightly exceeds bounds
        wm_x = max(0, min(wm_x, frame_width - watermark_pil.width))
        wm_y = max(0, min(wm_y, frame_height - watermark_pil.height))
        
        # 3. Paste watermark onto the frame using alpha compositing
        # Create a temporary RGBA layer for pasting
        rgba_layer = Image.new('RGBA', pil_frame.size, (0,0,0,0))
        try:
            rgba_layer.paste(watermark_pil, (wm_x, wm_y), mask=watermark_pil) # Use watermark alpha as mask
            # Composite the layer over the original frame
            pil_frame_out = Image.alpha_composite(pil_frame, rgba_layer)
        except Exception as e:
            print(f"Error pasting watermark at frame {frame_count}: {e}. Position:({wm_x},{wm_y}), WM Size: {watermark_pil.size}, Frame Size: {pil_frame.size}")
            pil_frame_out = pil_frame # Skip pasting on error for this frame

        # 4. Convert PIL image (RGBA) back to OpenCV frame (BGR)
        # Need to convert back to RGB first if output needs BGR
        pil_frame_out_rgb = pil_frame_out.convert("RGB")
        frame_out_bgr = cv2.cvtColor(np.array(pil_frame_out_rgb), cv2.COLOR_RGB2BGR)
        
        # --- Write the frame ---
        out.write(frame_out_bgr)

        frame_count += 1
        # Optional: Print progress
        if frame_count % 100 == 0 and total_frames > 0:
            elapsed = time.time() - start_time
            eta = (elapsed / frame_count) * (total_frames - frame_count) if frame_count > 0 else 0
            print(f"  Processed frame {frame_count}/{total_frames}. ETA: {eta:.2f}s")

    # Release everything
    cap.release()
    out.release()
    cv2.destroyAllWindows() # Should not be needed in script, but good practice
    end_time = time.time()
    print(f"Finished processing {input_video_path}. Time taken: {end_time - start_time:.2f}s")


# --- Gradio Interface ---

def process_videos_interface(
    input_videos, # List of file paths from gr.Files
    watermark_type,
    text_content,
    font_file, # File object from gr.File
    font_size,
    text_color,
    image_file, # File object from gr.File
    image_scale,
    motion_path,
    motion_margin,
    motion_speed,
    output_dir_name):

    if not input_videos:
        return "Error: No input video files selected.", []

    # Create default output directory if needed
    output_base = output_dir_name if output_dir_name else DEFAULT_OUTPUT_DIR
    if not os.path.exists(output_base):
        try:
            os.makedirs(output_base)
        except OSError as e:
            return f"Error: Could not create output directory '{output_base}': {e}", []
            
    status_messages = []
    output_file_paths = []
    
    # --- Prepare Watermark (once for all videos unless image scaling depends on frame size?)
    # Let's assume watermark is prepared once based on the *first* video's dimensions for scaling.
    # This is efficient but means image watermarks might look differently sized on videos with varying resolutions.
    # A better approach might be to prepare the watermark inside the loop for each video. Let's do that.
    
    watermark_pil = None # Will be prepared inside the loop

    # --- Font File Handling ---
    final_font_path = DEFAULT_FONT_PATH # Start with default
    if watermark_type == "Text":
        if font_file is not None:
            # Gradio File component gives a TemporaryFile object
            # We need its path. The .name attribute usually holds it.
            final_font_path = font_file.name 
            status_messages.append(f"Using provided font: {os.path.basename(final_font_path)}")
            # Check if the temp file path is accessible (it should be)
            if not os.path.exists(final_font_path):
                 status_messages.append(f"Warning: Provided font file path '{final_font_path}' seems invalid after upload. Trying default: {DEFAULT_FONT_PATH}")
                 final_font_path = DEFAULT_FONT_PATH # Fallback
        else:
            status_messages.append(f"No font file provided. Using default: {DEFAULT_FONT_PATH}")
            # Check if default exists
            if not os.path.exists(final_font_path):
                 status_messages.append(f"Error: Default font '{final_font_path}' not found. Cannot add text watermark.")
                 return "\n".join(status_messages), []

    # --- Image File Handling ---
    image_watermark_path = None
    if watermark_type == "Image":
        if image_file is not None:
            image_watermark_path = image_file.name
            status_messages.append(f"Using provided image: {os.path.basename(image_watermark_path)}")
            if not os.path.exists(image_watermark_path):
                 status_messages.append(f"Error: Provided image file path '{image_watermark_path}' seems invalid after upload.")
                 return "\n".join(status_messages), []
        else:
             status_messages.append(f"Error: Image watermark type selected, but no image file provided.")
             return "\n".join(status_messages), []


    # --- Process Each Video ---
    for video_file in input_videos:
        input_path = video_file.name # Path from Gradio's TemporaryFile object
        base_filename = os.path.basename(input_path)
        filename_no_ext, _ = os.path.splitext(base_filename)
        output_filename = f"{filename_no_ext}_watermarked.mp4"
        output_path = os.path.join(output_base, output_filename)

        status_messages.append(f"\nProcessing: {base_filename} -> {output_filename}")
        print(f"Starting processing for: {input_path}")

        try:
            # --- Get First Frame Dimensions for Watermark Prep ---
            # Need to open video briefly to get dimensions for scaling/layout
            temp_cap = cv2.VideoCapture(input_path)
            if not temp_cap.isOpened():
                raise IOError("Cannot open video to get dimensions.")
            vid_width = int(temp_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            vid_height = int(temp_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            temp_cap.release() # Close it immediately
            
            if vid_width <= 0 or vid_height <= 0:
                 raise ValueError(f"Invalid video dimensions obtained: {vid_width}x{vid_height}")

            # --- Prepare Watermark (Specific to this video's dimensions) ---
            if watermark_type == "Text":
                watermark_pil = create_text_watermark(text_content, final_font_path, font_size, text_color)
                status_messages.append(f"  Created text watermark ('{text_content[:20]}...')")
            elif watermark_type == "Image":
                watermark_pil = load_image_watermark(image_watermark_path, image_scale, vid_width, vid_height)
                status_messages.append(f"  Loaded and scaled image watermark.")
            else:
                raise ValueError("Invalid watermark type selected.")
                
            if watermark_pil.width <=0 or watermark_pil.height <= 0:
                raise ValueError("Watermark creation resulted in zero size image.")

            # --- Initialize Mover (Specific to this video and watermark size) ---
            mover = WatermarkMover(
                path_type=motion_path,
                frame_width=vid_width,
                frame_height=vid_height,
                watermark_width=watermark_pil.width,
                watermark_height=watermark_pil.height,
                margin_percent=motion_margin,
                speed_factor=motion_speed
            )
            status_messages.append(f"  Initialized motion: {motion_path}, Margin: {motion_margin}%, Speed: {motion_speed}x")

            # --- Add Watermark to Video ---
            add_watermark_to_video(input_path, output_path, watermark_pil, mover)
            
            status_messages.append(f"  Successfully processed and saved to: {output_path}")
            output_file_paths.append(output_path)

        except Exception as e:
            error_msg = f"  ERROR processing {base_filename}: {e}"
            print(error_msg) # Also print to console for debugging
            status_messages.append(error_msg)
            # Optionally: Clean up partially created output file?
            if os.path.exists(output_path):
                 try: os.remove(output_path)
                 except OSError: pass
                 
    status_messages.append("\nBatch processing finished.")
    # Return status log and list of output file paths
    return "\n".join(status_messages), output_file_paths


# --- Build Gradio UI ---

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎬 Video Watermark Adder Tool")
    gr.Markdown("Add dynamic text or image watermarks to your videos. Avoids using `moviepy`.")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 1. Input Videos")
            input_videos = gr.Files(label="Upload Video Files", file_types=['video'], type="filepath") # Use filepath for direct access

            gr.Markdown("### 2. Watermark Type & Content")
            watermark_type = gr.Radio(["Text", "Image"], label="Watermark Type", value="Text")

            # --- Text Options ---
            with gr.Column(visible=True) as text_options: # Start visible
                text_content = gr.Textbox(label="Watermark Text", value="Your Watermark")
                # Use gr.File for font upload
                font_file = gr.File(label=f"Upload Font File (.ttf, .otf) - Optional, uses default ({os.path.basename(DEFAULT_FONT_PATH)}) if not provided", file_types=['.ttf', '.otf'])
                font_size = gr.Slider(label="Font Size", minimum=10, maximum=200, value=48, step=1)
                text_color = gr.ColorPicker(label="Text Color", value="#FFFFFF")

            # --- Image Options ---
            with gr.Column(visible=False) as image_options: # Start hidden
                 image_file = gr.File(label="Upload Watermark Image", file_types=['image'], type="filepath")
                 image_scale = gr.Slider(label="Image Scale (%) relative to max video dimension", minimum=1, maximum=50, value=10, step=1)

            # --- Logic to show/hide options based on type ---
            def update_visibility(choice):
                return {
                    text_options: gr.update(visible=(choice == "Text")),
                    image_options: gr.update(visible=(choice == "Image")),
                }
            watermark_type.change(update_visibility, inputs=watermark_type, outputs=[text_options, image_options])

        with gr.Column(scale=1):
            gr.Markdown("### 3. Motion Settings")
            motion_path = gr.Dropdown(
                ["Static", "Bouncing Box", "Horizontal", "Vertical"],
                label="Watermark Motion Path",
                value="Bouncing Box"
            )
            motion_margin = gr.Slider(
                label="Motion Area Margin (%) - Excludes border area",
                minimum=0, maximum=45, value=5, step=1 # Max 45% margin each side prevents overlap
            )
            motion_speed = gr.Slider(
                label="Motion Speed Multiplier",
                minimum=0.5, maximum=5.0, value=1.0, step=0.1
            )
            
            gr.Markdown("### 4. Output Settings")
            output_dir_name = gr.Textbox(label="Output Subdirectory Name", value=DEFAULT_OUTPUT_DIR)

            gr.Markdown("### 5. Run Processing")
            process_button = gr.Button("Add Watermarks", variant="primary")

    with gr.Row():
         with gr.Column(scale=2):
            gr.Markdown("### Status Log")
            status_output = gr.Textbox(label="Log", lines=10, interactive=False)
         with gr.Column(scale=1):
             gr.Markdown("### Output Files")
             # Use gr.Files to display multiple output files for download
             output_files_display = gr.Files(label="Generated Video Files", interactive=False)


    # Connect button click to the processing function
    process_button.click(
        fn=process_videos_interface,
        inputs=[
            input_videos, watermark_type,
            text_content, font_file, font_size, text_color, # Text inputs
            image_file, image_scale, # Image inputs
            motion_path, motion_margin, motion_speed, # Motion inputs
            output_dir_name # Output dir
        ],
        outputs=[status_output, output_files_display] # Status log and output files
    )
    
    gr.Markdown("---")
    gr.Markdown(f"**Notes:** Processing can be slow for long videos. Ensure '{DEFAULT_FONT_PATH}' exists or provide a font file. Output videos are saved in the '{DEFAULT_OUTPUT_DIR}' (or specified) subdirectory.")


# Launch the Gradio app
if __name__ == "__main__":
    # Ensure default output dir exists for initial display text, but main function handles creation
    # if not os.path.exists(DEFAULT_OUTPUT_DIR):
    #     try: os.makedirs(DEFAULT_OUTPUT_DIR)
    #     except OSError: print(f"Warning: Could not create default output directory '{DEFAULT_OUTPUT_DIR}' on startup.")
        
    # Check for default font on startup
    if not os.path.exists(DEFAULT_FONT_PATH):
        print(f"WARNING: Default font '{DEFAULT_FONT_PATH}' not found.")
        print("Text watermarks might fail unless you provide a font file via the interface.")
        
    print("Launching Gradio Interface...")
    demo.launch()