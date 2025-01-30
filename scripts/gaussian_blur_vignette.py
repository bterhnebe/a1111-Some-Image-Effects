import modules.scripts as scripts
import gradio as gr
from modules import images, shared
from modules.processing import process_images, Processed
from modules.shared import opts, state
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageChops, ImageOps
import numpy as np
import random
import os

###############################################
# 工具函数：RGB <-> HSL 转换（给 hue/saturation/color/luminosity 混合用）
###############################################

def rgb_to_hsl(r, g, b):
    """
    r,g,b 都在 [0,1] 范围内
    返回 (h, s, l)
    h 范围 [0,360), s,l 范围 [0,1]
    """
    mx = max(r, g, b)
    mn = min(r, g, b)
    d = mx - mn
    l = (mx + mn) / 2.0

    if abs(d) < 1e-8:  # 几乎没有差异，说明是灰度
        h = 0
        s = 0
    else:
        s = d / (1 - abs(2*l - 1))
        if abs(mx - r) < 1e-8:
            h = ((g - b) / d) % 6
        elif abs(mx - g) < 1e-8:
            h = (b - r) / d + 2
        else:
            h = (r - g) / d + 4
        h *= 60.0

    return h, s, l

def hsl_to_rgb(h, s, l):
    """
    接收 (h, s, l)，其中 h in [0,360), s,l in [0,1]
    返回 (r, g, b) 也在 [0,1]
    """
    c = (1 - abs(2*l - 1)) * s
    hh = (h / 60.0) % 6
    x = c * (1 - abs(hh - int(hh//1) - 1))
    m = l - c/2

    if 0 <= hh < 1:
        r, g, b = c, x, 0
    elif 1 <= hh < 2:
        r, g, b = x, c, 0
    elif 2 <= hh < 3:
        r, g, b = 0, c, x
    elif 3 <= hh < 4:
        r, g, b = 0, x, c
    elif 4 <= hh < 5:
        r, g, b = x, 0, c
    else:
        r, g, b = c, 0, x

    return (r + m, g + m, b + m)

###############################################
# 主要脚本
###############################################

class SomeImageEffectsScript(scripts.Script):
    def __init__(self):
        super().__init__()
        self.overlay_files = []
        self.update_overlay_files()

    def update_overlay_files(self):
        # 打印当前工作目录和脚本路径，方便调试
        print(f"Current working directory: {os.getcwd()}")
        print(f"Script location: {os.path.dirname(os.path.abspath(__file__))}")
        
        # 多个可能的 overlays 目录，挨个尝试
        potential_dirs = [
            os.path.join(scripts.basedir(), "overlays"),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "overlays"),
            os.path.join(os.getcwd(), "overlays"),
        ]
        
        for overlay_dir in potential_dirs:
            print(f"Checking for overlays in: {overlay_dir}")
            if os.path.exists(overlay_dir):
                self.overlay_files = [f for f in os.listdir(overlay_dir) if self.is_image_file(f)]
                print(f"Found {len(self.overlay_files)} overlay files in {overlay_dir}")
                return
        
        print("Overlay directory not found in any of the checked locations.")

    def is_image_file(self, filename):
        image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp']
        return any(filename.lower().endswith(ext) for ext in image_extensions)

    def title(self):
        return "Advanced Image Effects"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Group():
            with gr.Accordion("Some Image Effects", open=False):
                save_original = gr.Checkbox(label="Save Original Image", value=True)
                
                with gr.Row():
                    enable_grain = gr.Checkbox(label="Enable Grain", value=False)
                    enable_vignette = gr.Checkbox(label="Enable Vignette", value=False)
                    enable_random_blur = gr.Checkbox(label="Enable Random Blur", value=False)
                    enable_color_offset = gr.Checkbox(label="Enable Color Offset", value=False)
                      
                with gr.Row():
                    grain_intensity = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, value=0.3, label="Grain Intensity")
                
                with gr.Row():
                    vignette_intensity = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, value=0.3, label="Vignette Intensity")
                    vignette_feather = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, value=0.3, label="Vignette Feather")
                    vignette_roundness = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, value=0.5, label="Vignette Roundness")
                
                with gr.Row():
                    blur_max_size = gr.Slider(minimum=0.0, maximum=0.5, step=0.05, value=0.2, label="Max Blur Size (% of image)")
                    blur_strength = gr.Slider(minimum=0.0, maximum=10.0, step=0.5, value=3.0, label="Blur Strength")
                
                with gr.Row():
                    color_offset_x = gr.Slider(minimum=-50, maximum=50, step=1, value=0, label="Color Offset X")
                    color_offset_y = gr.Slider(minimum=-50, maximum=50, step=1, value=0, label="Color Offset Y")
                
                ### 第一层 Overlay (和原本一样)
                with gr.Row():
                    enable_overlay = gr.Checkbox(label="Enable Overlay (Layer 1)", value=False)
                    overlay_file = gr.Dropdown(label="Overlay File 1", choices=self.overlay_files)
                    overlay_fit = gr.Dropdown(label="Overlay Fit 1", choices=["stretch", "fit_out"], value="stretch")
                
                with gr.Row():
                    overlay_opacity = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, value=0.5, label="Overlay Opacity 1")
                    overlay_blend_mode = gr.Dropdown(label="Overlay Blend Mode 1", choices=[
                        "normal", "multiply", "screen", "overlay", "darken", "lighten",
                        "color_dodge", "color_burn", "hard_light", "soft_light", "difference",
                        "exclusion", "hue", "saturation", "color", "luminosity"
                    ], value="normal")

                with gr.Row():
                    enable_luminosity = gr.Checkbox(label="Enable Luminosity Adjustment", value=False)
                    luminosity_factor = gr.Slider(minimum=-1.0, maximum=1.0, step=0.05, value=0, label="Luminosity Factor")

                with gr.Row():
                    enable_contrast = gr.Checkbox(label="Enable Contrast Adjustment", value=False)
                    contrast_factor = gr.Slider(minimum=0.0, maximum=2.0, step=0.05, value=1, label="Contrast Factor")

                with gr.Row():
                    enable_hue = gr.Checkbox(label="Enable Hue Adjustment", value=False)
                    hue_factor = gr.Slider(minimum=-180, maximum=180, step=1, value=0, label="Hue Shift")

                with gr.Row():
                    enable_saturation = gr.Checkbox(label="Enable Saturation Adjustment", value=False)
                    saturation_factor = gr.Slider(minimum=0.0, maximum=2.0, step=0.05, value=1, label="Saturation Factor")

            ##############################################
            # 新增的多层 Overlays (这里演示添加第2层、第3层)
            ##############################################
            with gr.Accordion("Additional Overlays (Layer 2, Layer 3)", open=False):
                # 第2层 Overlay
                with gr.Box():
                    enable_overlay2 = gr.Checkbox(label="Enable Overlay 2", value=False)
                    overlay_file2 = gr.Dropdown(label="Overlay File 2", choices=self.overlay_files)
                    overlay_fit2 = gr.Dropdown(label="Overlay Fit 2", choices=["stretch", "fit_out"], value="stretch")
                    overlay_opacity2 = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, value=0.5, label="Overlay Opacity 2")
                    overlay_blend_mode2 = gr.Dropdown(label="Overlay Blend Mode 2", choices=[
                        "normal", "multiply", "screen", "overlay", "darken", "lighten",
                        "color_dodge", "color_burn", "hard_light", "soft_light", "difference",
                        "exclusion", "hue", "saturation", "color", "luminosity"
                    ], value="normal")

                # 第3层 Overlay
                with gr.Box():
                    enable_overlay3 = gr.Checkbox(label="Enable Overlay 3", value=False)
                    overlay_file3 = gr.Dropdown(label="Overlay File 3", choices=self.overlay_files)
                    overlay_fit3 = gr.Dropdown(label="Overlay Fit 3", choices=["stretch", "fit_out"], value="stretch")
                    overlay_opacity3 = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, value=0.5, label="Overlay Opacity 3")
                    overlay_blend_mode3 = gr.Dropdown(label="Overlay Blend Mode 3", choices=[
                        "normal", "multiply", "screen", "overlay", "darken", "lighten",
                        "color_dodge", "color_burn", "hard_light", "soft_light", "difference",
                        "exclusion", "hue", "saturation", "color", "luminosity"
                    ], value="normal")

        # 注意：返回值顺序一定要跟 postprocess_image / add_effects 里解包的顺序对应
        return [
            # 前面已有的参数
            save_original,
            enable_grain, enable_vignette, enable_random_blur, enable_color_offset,
            grain_intensity, vignette_intensity, vignette_feather, vignette_roundness,
            blur_max_size, blur_strength, color_offset_x, color_offset_y,
            enable_overlay, overlay_file, overlay_fit, overlay_opacity, overlay_blend_mode,
            enable_luminosity, luminosity_factor, enable_contrast, contrast_factor,
            enable_hue, hue_factor, enable_saturation, saturation_factor,

            # 新增的第2层 Overlay
            enable_overlay2, overlay_file2, overlay_fit2, overlay_opacity2, overlay_blend_mode2,
            # 新增的第3层 Overlay
            enable_overlay3, overlay_file3, overlay_fit3, overlay_opacity3, overlay_blend_mode3
        ]

    def process(self, p, *args):
        """
        在这里把启用的效果名字放到 p.extra_generation_params 里，只是让结果信息里能显示。
        """
        # 根据上面UI的return顺序来解包：
        (
            save_original,
            enable_grain, enable_vignette, enable_random_blur, enable_color_offset,
            grain_intensity, vignette_intensity, vignette_feather, vignette_roundness,
            blur_max_size, blur_strength, color_offset_x, color_offset_y,
            enable_overlay, overlay_file, overlay_fit, overlay_opacity, overlay_blend_mode,
            enable_luminosity, luminosity_factor, enable_contrast, contrast_factor,
            enable_hue, hue_factor, enable_saturation, saturation_factor,

            enable_overlay2, overlay_file2, overlay_fit2, overlay_opacity2, overlay_blend_mode2,
            enable_overlay3, overlay_file3, overlay_fit3, overlay_opacity3, overlay_blend_mode3
        ) = args

        enabled_effects = []
        if enable_grain:       enabled_effects.append("Grain")
        if enable_vignette:    enabled_effects.append("Vignette")
        if enable_random_blur: enabled_effects.append("Random Blur")
        if enable_color_offset:enabled_effects.append("Color Offset")
        if enable_overlay:     enabled_effects.append("Overlay Layer1")
        if enable_overlay2:    enabled_effects.append("Overlay Layer2")
        if enable_overlay3:    enabled_effects.append("Overlay Layer3")
        if enable_luminosity:  enabled_effects.append("Luminosity")
        if enable_contrast:    enabled_effects.append("Contrast")
        if enable_hue:         enabled_effects.append("Hue")
        if enable_saturation:  enabled_effects.append("Saturation")
        
        if enabled_effects:
            p.extra_generation_params["Some Image Effects"] = ", ".join(enabled_effects)

    def postprocess_image(self, p, pp, *args):
        """
        出图后再执行的后处理
        """
        (
            save_original,
            enable_grain, enable_vignette, enable_random_blur, enable_color_offset,
            grain_intensity, vignette_intensity, vignette_feather, vignette_roundness,
            blur_max_size, blur_strength, color_offset_x, color_offset_y,
            enable_overlay, overlay_file, overlay_fit, overlay_opacity, overlay_blend_mode,
            enable_luminosity, luminosity_factor, enable_contrast, contrast_factor,
            enable_hue, hue_factor, enable_saturation, saturation_factor,

            enable_overlay2, overlay_file2, overlay_fit2, overlay_opacity2, overlay_blend_mode2,
            enable_overlay3, overlay_file3, overlay_fit3, overlay_opacity3, overlay_blend_mode3
        ) = args

        if hasattr(pp, 'image') and pp.image is not None:
            if save_original:
                self.save_original_image(pp.image)
            pp.image = self.add_effects(
                pp.image,
                save_original, enable_grain, enable_vignette, enable_random_blur, enable_color_offset,
                grain_intensity, vignette_intensity, vignette_feather, vignette_roundness,
                blur_max_size, blur_strength, color_offset_x, color_offset_y,
                enable_overlay, overlay_file, overlay_fit, overlay_opacity, overlay_blend_mode,
                enable_luminosity, luminosity_factor, enable_contrast, contrast_factor,
                enable_hue, hue_factor, enable_saturation, saturation_factor,

                enable_overlay2, overlay_file2, overlay_fit2, overlay_opacity2, overlay_blend_mode2,
                enable_overlay3, overlay_file3, overlay_fit3, overlay_opacity3, overlay_blend_mode3
            )
        elif hasattr(pp, 'images') and pp.images:
            for i, image in enumerate(pp.images):
                if save_original:
                    self.save_original_image(image)
                pp.images[i] = self.add_effects(
                    image,
                    save_original, enable_grain, enable_vignette, enable_random_blur, enable_color_offset,
                    grain_intensity, vignette_intensity, vignette_feather, vignette_roundness,
                    blur_max_size, blur_strength, color_offset_x, color_offset_y,
                    enable_overlay, overlay_file, overlay_fit, overlay_opacity, overlay_blend_mode,
                    enable_luminosity, luminosity_factor, enable_contrast, contrast_factor,
                    enable_hue, hue_factor, enable_saturation, saturation_factor,

                    enable_overlay2, overlay_file2, overlay_fit2, overlay_opacity2, overlay_blend_mode2,
                    enable_overlay3, overlay_file3, overlay_fit3, overlay_opacity3, overlay_blend_mode3
                )

    def add_effects(self, img,
                    save_original, enable_grain, enable_vignette, enable_random_blur, enable_color_offset,
                    grain_intensity, vignette_intensity, vignette_feather, vignette_roundness,
                    blur_max_size, blur_strength, color_offset_x, color_offset_y,
                    enable_overlay, overlay_file, overlay_fit, overlay_opacity, overlay_blend_mode,
                    enable_luminosity, luminosity_factor, enable_contrast, contrast_factor,
                    enable_hue, hue_factor, enable_saturation, saturation_factor,
                    enable_overlay2, overlay_file2, overlay_fit2, overlay_opacity2, overlay_blend_mode2,
                    enable_overlay3, overlay_file3, overlay_fit3, overlay_opacity3, overlay_blend_mode3
                   ):

        if img is None:
            print("Error: Input image is None")
            return None

        # ====== 1) 先做各种单图效果 ======
        if enable_grain:
            img = self.add_grain(img, grain_intensity)
        
        if enable_vignette:
            img = self.add_vignette(img, vignette_intensity, vignette_feather, vignette_roundness)
        
        if enable_random_blur:
            img = self.add_random_blur(img, blur_max_size, blur_strength)
        
        if enable_color_offset:
            img = self.add_color_offset(img, color_offset_x, color_offset_y)
        
        # ====== 2) 第一层 Overlay ======
        if enable_overlay and overlay_file:
            img = self.add_overlay(img, overlay_file, overlay_fit, overlay_opacity, overlay_blend_mode)
        
        # ====== 3) 亮度/对比度/色相/饱和度 ======
        if enable_luminosity:
            img = self.adjust_luminosity(img, luminosity_factor)
        
        if enable_contrast:
            img = self.adjust_contrast(img, contrast_factor)
        
        if enable_hue:
            img = self.adjust_hue(img, hue_factor)
        
        if enable_saturation:
            img = self.adjust_saturation(img, saturation_factor)

        # ====== 4) 第二层 Overlay ======
        if enable_overlay2 and overlay_file2:
            img = self.add_overlay(img, overlay_file2, overlay_fit2, overlay_opacity2, overlay_blend_mode2)

        # ====== 5) 第三层 Overlay ======
        if enable_overlay3 and overlay_file3:
            img = self.add_overlay(img, overlay_file3, overlay_fit3, overlay_opacity3, overlay_blend_mode3)
        
        return img

    ################################################
    # 各种子效果函数
    ################################################

    def add_grain(self, img, intensity):
        img_np = np.array(img)
        noise = np.random.randn(*img_np.shape) * 255 * intensity
        noisy_img = np.clip(img_np + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy_img)

    def add_vignette(self, img, intensity, feather, roundness):
        width, height = img.size
        mask = Image.new('L', (width, height), 255)
        draw = ImageDraw.Draw(mask)
        
        x_center, y_center = width // 2, height // 2
        max_radius = min(width, height) // 2
        
        for i in range(max_radius):
            alpha = int(255 * (1 - (i / max_radius) ** roundness) * intensity)
            draw.ellipse([x_center - i, y_center - i, x_center + i, y_center + i], fill=alpha)
        
        mask = mask.filter(ImageFilter.GaussianBlur(radius=max_radius * feather))
        
        enhancer = ImageEnhance.Brightness(img)
        darkened = enhancer.enhance(1 - intensity * 0.5)
        
        return Image.composite(darkened, img, mask)

    def add_random_blur(self, img, max_size, strength):
        width, height = img.size
        blur_size = int(min(width, height) * max_size)
        if blur_size < 1:
            return img  # 如果设置太小就直接不做

        x = random.randint(0, width - blur_size)
        y = random.randint(0, height - blur_size)
        
        mask = Image.new('L', (width, height), 0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse([x, y, x + blur_size, y + blur_size], fill=255)
        mask = mask.filter(ImageFilter.GaussianBlur(radius=blur_size // 4 if blur_size > 4 else 1))
        
        blurred = img.filter(ImageFilter.GaussianBlur(radius=strength))
        return Image.composite(blurred, img, mask)

    def add_color_offset(self, img, offset_x, offset_y):
        r, g, b = img.split()
        r = ImageChops.offset(r, offset_x, offset_y)
        b = ImageChops.offset(b, -offset_x, -offset_y)
        return Image.merge('RGB', (r, g, b))

    def add_overlay(self, img, overlay_file, overlay_fit, opacity, blend_mode):
        """
        加载 overlay 图片，按照给定方式贴到 img 上，然后根据 blend_mode 进行混合
        """
        if img is None:
            print("Error: Input image is None")
            return None

        potential_dirs = [
            os.path.join(scripts.basedir(), "overlays"),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "overlays"),
            os.path.join(os.getcwd(), "overlays"),
        ]
        
        overlay_path = None
        for overlay_dir in potential_dirs:
            temp_path = os.path.join(overlay_dir, overlay_file)
            if os.path.exists(temp_path):
                overlay_path = temp_path
                break

        if not overlay_path:
            print(f"Overlay file not found: {overlay_file}")
            return img

        try:
            overlay = Image.open(overlay_path).convert("RGBA")
        except Exception as e:
            print(f"Error opening overlay file: {e}")
            return img

        # 根据选项 resize overlay
        if overlay_fit == "stretch":
            overlay = overlay.resize(img.size, Image.LANCZOS)
        elif overlay_fit == "fit_out":
            img_ratio = img.width / img.height
            overlay_ratio = overlay.width / overlay.height
            if img_ratio > overlay_ratio:
                new_width = img.width
                new_height = int(new_width / overlay_ratio)
            else:
                new_height = img.height
                new_width = int(new_height * overlay_ratio)
            overlay = overlay.resize((new_width, new_height), Image.LANCZOS)
            # 居中裁剪到和原图相同大小
            left = (overlay.width - img.width) // 2
            top = (overlay.height - img.height) // 2
            right = left + img.width
            bottom = top + img.height
            overlay = overlay.crop((left, top, right, bottom))

        # 应用不透明度
        overlay = Image.blend(Image.new('RGBA', img.size, (0, 0, 0, 0)), overlay, opacity)

        # 确保都是 RGBA
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        if overlay.mode != 'RGBA':
            overlay = overlay.convert('RGBA')

        # 根据 blend_mode 混合
        if blend_mode == "multiply":
            blended = ImageChops.multiply(img, overlay)
        elif blend_mode == "screen":
            blended = ImageChops.screen(img, overlay)
        elif blend_mode == "overlay":
            blended = self.overlay_blend(img, overlay, opacity)
        elif blend_mode == "darken":
            blended = ImageChops.darker(img, overlay)
        elif blend_mode == "lighten":
            blended = ImageChops.lighter(img, overlay)
        elif blend_mode == "color_dodge":
            blended = self.color_dodge(img, overlay)
        elif blend_mode == "color_burn":
            blended = self.color_burn(img, overlay)
        elif blend_mode == "hard_light":
            blended = self.hard_light(img, overlay)
        elif blend_mode == "soft_light":
            blended = self.soft_light(img, overlay)
        elif blend_mode == "difference":
            blended = ImageChops.difference(img, overlay)
        elif blend_mode == "exclusion":
            blended = self.exclusion(img, overlay)
        elif blend_mode == "hue":
            blended = self.hue_blend(img, overlay)
        elif blend_mode == "saturation":
            blended = self.saturation_blend(img, overlay)
        elif blend_mode == "color":
            blended = self.color_blend(img, overlay)
        elif blend_mode == "luminosity":
            blended = self.luminosity_blend(img, overlay)
        else:  # normal
            blended = Image.alpha_composite(img, overlay)

        return blended.convert("RGB")

    ################################################
    # 调整亮度/对比度/色相/饱和度函数
    ################################################

    def adjust_luminosity(self, img, factor):
        return ImageEnhance.Brightness(img).enhance(1 + factor)

    def adjust_contrast(self, img, factor):
        return ImageEnhance.Contrast(img).enhance(factor)

    def adjust_hue(self, img, shift):
        # 先转HSV，移动H通道，然后再转回 RGB
        hsv = img.convert('HSV')
        h, s, v = hsv.split()
        # shift 范围[-180,180]，我们要把它映射到HSV通道 [0..255] 的变化
        def shift_hue(x):
            return (x + int(shift * 255/360)) % 256
        h = h.point(shift_hue)
        return Image.merge('HSV', (h, s, v)).convert('RGB')

    def adjust_saturation(self, img, factor):
        return ImageEnhance.Color(img).enhance(factor)

    ################################################
    # 实际在 add_overlay 里会调用的各种混合模式
    ################################################

    def overlay_blend(self, base, top, opacity=1.0):
        """
        计算 Photoshop 版 Overlay 混合模式，并支持正确的不透明度处理。
        base, top 都是 RGBA 图片，opacity 控制最终叠加的透明度。
        """

        # 转换为 NumPy 数组，归一化到 0~1
        base_np = np.array(base, dtype=np.float32) / 255.0
        top_np = np.array(top, dtype=np.float32) / 255.0

        # 计算 Overlay 模式的 R/G/B
        out_np = np.zeros_like(base_np)

        for c in range(3):  # 只处理 R/G/B 颜色通道（不处理 Alpha）
            b = base_np[:, :, c]
            t = top_np[:, :, c]

            # PS 版 Overlay 模式
            result = np.where(b < 0.5, 
                              2.0 * b * t, 
                              1.0 - 2.0 * (1.0 - b) * (1.0 - t))
            
            # 这里的 `opacity` 控制叠加透明度，而不会影响原始 Overlay 计算
            out_np[:, :, c] = (1.0 - opacity) * b + opacity * result

        # Alpha 通道：正确混合 alpha
        alpha_base = base_np[:, :, 3]
        alpha_top = top_np[:, :, 3] * opacity
        out_np[:, :, 3] = alpha_base + alpha_top * (1 - alpha_base)

        # 转换回 0~255 并返回 PIL Image
        out_np = (out_np * 255.0).clip(0, 255).astype(np.uint8)
        return Image.fromarray(out_np, mode="RGBA")

    def color_dodge(self, base, top):
        base_np = np.array(base, dtype=np.float32) / 255.0
        top_np = np.array(top, dtype=np.float32) / 255.0
        out = np.zeros_like(base_np)
        eps = 1e-5

        # Color Dodge formula: result = base / (1 - top)，若 top=1 则结果=1
        for c in range(3):
            b = base_np[:, :, c]
            t = top_np[:, :, c]
            out[:, :, c] = np.where(t < 1.0 - eps, np.minimum(1.0, b / (1.0 - t + eps)), 1.0)

        out[:, :, 3] = top_np[:, :, 3]
        out = (out * 255.0).clip(0, 255).astype(np.uint8)
        return Image.fromarray(out, 'RGBA')

    def color_burn(self, base, top):
        base_np = np.array(base, dtype=np.float32) / 255.0
        top_np = np.array(top, dtype=np.float32) / 255.0
        out = np.zeros_like(base_np)
        eps = 1e-5

        # Color Burn formula: result = 1 - (1 - base) / top
        for c in range(3):
            b = base_np[:, :, c]
            t = top_np[:, :, c]
            out[:, :, c] = np.where(t > eps, 1.0 - np.minimum(1.0, (1.0 - b) / t), 0.0)

        out[:, :, 3] = top_np[:, :, 3]
        out = (out * 255.0).clip(0, 255).astype(np.uint8)
        return Image.fromarray(out, 'RGBA')

    def hard_light(self, base, top):
        """
        Hard Light 与 Overlay 类似，但以 top 作为判断:
        if top <= 0.5: result = 2 * base * top
        else:          result = 1 - 2*(1 - base)*(1 - top)
        """
        base_np = np.array(base, dtype=np.float32) / 255.0
        top_np = np.array(top, dtype=np.float32) / 255.0
        out = np.zeros_like(base_np)

        for c in range(3):
            b = base_np[:, :, c]
            t = top_np[:, :, c]
            mask = (t <= 0.5)
            temp = np.zeros_like(b)
            temp[mask] = 2.0 * b[mask] * t[mask]
            temp[~mask] = 1.0 - 2.0*(1.0 - b[~mask])*(1.0 - t[~mask])
            out[:, :, c] = temp

        out[:, :, 3] = top_np[:, :, 3]
        out = (out * 255.0).clip(0, 255).astype(np.uint8)
        return Image.fromarray(out, 'RGBA')

    def soft_light(self, base, top):
        """
        Soft Light 公式: result = (1 - 2T)*B^2 + 2T*B
        """
        base_np = np.array(base, dtype=np.float32) / 255.0
        top_np = np.array(top, dtype=np.float32) / 255.0
        out = np.zeros_like(base_np)

        for c in range(3):
            b = base_np[:, :, c]
            t = top_np[:, :, c]
            out[:, :, c] = (1.0 - 2.0 * t) * (b**2) + 2.0 * t * b

        out[:, :, 3] = top_np[:, :, 3]
        out = (out * 255.0).clip(0, 255).astype(np.uint8)
        return Image.fromarray(out, 'RGBA')

    def exclusion(self, base, top):
        """
        Exclusion 公式: result = base + top - 2 * base * top
        """
        base_np = np.array(base, dtype=np.float32) / 255.0
        top_np = np.array(top, dtype=np.float32) / 255.0
        out = np.zeros_like(base_np)

        for c in range(3):
            b = base_np[:, :, c]
            t = top_np[:, :, c]
            out[:, :, c] = b + t - 2.0*b*t

        out[:, :, 3] = top_np[:, :, 3]
        out = (out * 255.0).clip(0, 255).astype(np.uint8)
        return Image.fromarray(out, 'RGBA')

    def hue_blend(self, base, top):
        """
        Hue 模式: 使用 top 的 Hue，base 的 Saturation & Lightness
        """
        base_np = np.array(base, dtype=np.float32)
        top_np = np.array(top, dtype=np.float32)
        out = np.zeros_like(base_np)

        height, width, _ = base_np.shape
        for i in range(height):
            for j in range(width):
                br, bg, bb, ba = base_np[i, j] / 255.0
                tr, tg, tb, ta = top_np[i, j] / 255.0

                bh, bs, bl = rgb_to_hsl(br, bg, bb)
                th, ts, tl = rgb_to_hsl(tr, tg, tb)

                # 保留 base 的 S,L，使用 top 的 H
                nh, ns, nl = (th, bs, bl)
                nr, ng, nb = hsl_to_rgb(nh, ns, nl)

                out[i, j, 0] = nr * 255.0
                out[i, j, 1] = ng * 255.0
                out[i, j, 2] = nb * 255.0
                out[i, j, 3] = ta * 255.0  # alpha 用 top 的

        return Image.fromarray(out.astype(np.uint8), mode='RGBA')

    def saturation_blend(self, base, top):
        """
        Saturation 模式: 使用 top 的 Saturation，base 的 Hue, Lightness
        """
        base_np = np.array(base, dtype=np.float32)
        top_np = np.array(top, dtype=np.float32)
        out = np.zeros_like(base_np)

        height, width, _ = base_np.shape
        for i in range(height):
            for j in range(width):
                br, bg, bb, ba = base_np[i, j] / 255.0
                tr, tg, tb, ta = top_np[i, j] / 255.0

                bh, bs, bl = rgb_to_hsl(br, bg, bb)
                th, ts, tl = rgb_to_hsl(tr, tg, tb)

                # 保留 base 的 H,L，使用 top 的 S
                nh, ns, nl = (bh, ts, bl)
                nr, ng, nb = hsl_to_rgb(nh, ns, nl)

                out[i, j, 0] = nr * 255.0
                out[i, j, 1] = ng * 255.0
                out[i, j, 2] = nb * 255.0
                out[i, j, 3] = ta * 255.0

        return Image.fromarray(out.astype(np.uint8), mode='RGBA')

    def color_blend(self, base, top):
        """
        Color 模式: 使用 top 的 Hue, Saturation；base 的 Lightness
        """
        base_np = np.array(base, dtype=np.float32)
        top_np = np.array(top, dtype=np.float32)
        out = np.zeros_like(base_np)

        height, width, _ = base_np.shape
        for i in range(height):
            for j in range(width):
                br, bg, bb, ba = base_np[i, j] / 255.0
                tr, tg, tb, ta = top_np[i, j] / 255.0

                bh, bs, bl = rgb_to_hsl(br, bg, bb)
                th, ts, tl = rgb_to_hsl(tr, tg, tb)

                # 使用 top 的 H,S，base 的 L
                nh, ns, nl = (th, ts, bl)
                nr, ng, nb = hsl_to_rgb(nh, ns, nl)

                out[i, j, 0] = nr * 255.0
                out[i, j, 1] = ng * 255.0
                out[i, j, 2] = nb * 255.0
                out[i, j, 3] = ta * 255.0

        return Image.fromarray(out.astype(np.uint8), mode='RGBA')

    def luminosity_blend(self, base, top):
        """
        Luminosity 模式: 使用 top 的 Lightness，base 的 Hue, Saturation
        """
        base_np = np.array(base, dtype=np.float32)
        top_np = np.array(top, dtype=np.float32)
        out = np.zeros_like(base_np)

        height, width, _ = base_np.shape
        for i in range(height):
            for j in range(width):
                br, bg, bb, ba = base_np[i, j] / 255.0
                tr, tg, tb, ta = top_np[i, j] / 255.0

                bh, bs, bl = rgb_to_hsl(br, bg, bb)
                th, ts, tl = rgb_to_hsl(tr, tg, tb)

                # 使用 base 的 H,S，top 的 L
                nh, ns, nl = (bh, bs, tl)
                nr, ng, nb = hsl_to_rgb(nh, ns, nl)

                out[i, j, 0] = nr * 255.0
                out[i, j, 1] = ng * 255.0
                out[i, j, 2] = nb * 255.0
                out[i, j, 3] = ta * 255.0

        return Image.fromarray(out.astype(np.uint8), mode='RGBA')

    ################################################
    # 保存原图（可选）
    ################################################

    def save_original_image(self, img):
        save_dir = (
            getattr(shared.opts, 'outdir_samples', None)
            or getattr(shared.opts, 'outdir_txt2img_samples', None)
            or getattr(shared.opts, 'outdir_img2img_samples', None)
            or getattr(shared.opts, 'outdir_extras_samples', None)
            or os.getcwd()
        )
        
        save_dir = os.path.join(save_dir, "originals")
        os.makedirs(save_dir, exist_ok=True)
        
        # 优先用当前任务名，若没有就叫 image
        job_name = shared.state.job if shared.state.job else "image"
        base_name, ext = os.path.splitext(job_name)
        if not ext:
            ext = ".png"  # 默认用PNG
        
        filename = f"{base_name}_original{ext}"
        save_path = os.path.join(save_dir, filename)
        
        try:
            img.save(save_path)
            print(f"Original image saved to: {save_path}")
        except Exception as e:
            print(f"Error saving original image: {e}")
