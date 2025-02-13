import modules.scripts as scripts
import gradio as gr
from modules import images, shared
from modules.processing import process_images, Processed
from modules.shared import opts, state
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageChops, ImageOps
import numpy as np
import random
import os
import json

###############################################
# 工具函数：RGB <-> HSL 转换（给某些混合模式用）
###############################################

def rgb_to_hsl(r, g, b):
    """
    r, g, b in [0..1], 返回 H in [0..360], S,L in [0..1]
    """
    mx = max(r, g, b)
    mn = min(r, g, b)
    d = mx - mn
    l = (mx + mn) / 2.0
    if abs(d) < 1e-8:
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
    传入 h in [0..360], s,l in [0..1],
    返回 R,G,B in [0..1]
    """
    c = (1 - abs(2*l - 1)) * s
    hh = (h / 60.0) % 6
    x = c * (1 - abs(hh - int(hh // 1) - 1))
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
# 工具函数：读取/保存预设
###############################################

PRESETS_JSON = "some_image_effects_presets.json"

def load_presets_from_json():
    if not os.path.exists(PRESETS_JSON):
        return {}
    try:
        with open(PRESETS_JSON, 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return {}

def save_presets_to_json(presets_dict):
    try:
        with open(PRESETS_JSON, 'w', encoding='utf-8') as f:
            json.dump(presets_dict, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Error saving presets to {PRESETS_JSON}: {e}")

###############################################
# 脚本主类
###############################################

class SomeImageEffectsScript(scripts.Script):
    def __init__(self):
        super().__init__()
        self.overlay_files = []
        self.update_overlay_files()

        # 预设存储
        self.presets = load_presets_from_json()

    def update_overlay_files(self):
        print(f"Current working directory: {os.getcwd()}")
        print(f"Script location: {os.path.dirname(os.path.abspath(__file__))}")
        
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
        """
        这里定义脚本的所有UI控件
        """
        with gr.Group():
            # ===================================
            # 基本功能区（保存原图、随机模糊、Vignette、Overlay等）
            # ===================================
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
                
                ### Overlay (Layer1)
                with gr.Row():
                    enable_overlay = gr.Checkbox(label="Enable Overlay (Layer 1)", value=False)
                    overlay_file = gr.Dropdown(label="Overlay File 1", choices=self.overlay_files)
                    overlay_fit = gr.Dropdown(label="Overlay Fit 1", choices=["stretch", "fit_out"], value="stretch")
                
                with gr.Row():
                    overlay_opacity = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, value=0.5, label="Overlay Opacity 1")
                    overlay_blend_mode = gr.Dropdown(
                        label="Overlay Blend Mode 1",
                        choices=[
                            "normal", "multiply", "screen", "overlay", "darken", "lighten",
                            "color_dodge", "color_burn", "hard_light", "soft_light", "difference",
                            "exclusion", "hue", "saturation", "color", "luminosity"
                        ],
                        value="normal"
                    )

            # ===================================
            # 多层 Overlays
            # ===================================
            with gr.Accordion("Additional Overlays (Layer 2, Layer 3)", open=False):
                with gr.Box():
                    enable_overlay2 = gr.Checkbox(label="Enable Overlay 2", value=False)
                    overlay_file2 = gr.Dropdown(label="Overlay File 2", choices=self.overlay_files)
                    overlay_fit2 = gr.Dropdown(label="Overlay Fit 2", choices=["stretch", "fit_out"], value="stretch")
                    overlay_opacity2 = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, value=0.5, label="Overlay Opacity 2")
                    overlay_blend_mode2 = gr.Dropdown(
                        label="Overlay Blend Mode 2",
                        choices=[
                            "normal", "multiply", "screen", "overlay", "darken", "lighten",
                            "color_dodge", "color_burn", "hard_light", "soft_light", "difference",
                            "exclusion", "hue", "saturation", "color", "luminosity"
                        ],
                        value="normal"
                    )

                with gr.Box():
                    enable_overlay3 = gr.Checkbox(label="Enable Overlay 3", value=False)
                    overlay_file3 = gr.Dropdown(label="Overlay File 3", choices=self.overlay_files)
                    overlay_fit3 = gr.Dropdown(label="Overlay Fit 3", choices=["stretch", "fit_out"], value="stretch")
                    overlay_opacity3 = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, value=0.5, label="Overlay Opacity 3")
                    overlay_blend_mode3 = gr.Dropdown(
                        label="Overlay Blend Mode 3",
                        choices=[
                            "normal", "multiply", "screen", "overlay", "darken", "lighten",
                            "color_dodge", "color_burn", "hard_light", "soft_light", "difference",
                            "exclusion", "hue", "saturation", "color", "luminosity"
                        ],
                        value="normal"
                    )

            # ===================================
            # 调整项 (Levels / Brightness / Luminosity / Contrast / Hue / Saturation)
            # ===================================
            with gr.Accordion("Adjustments (Levels / Brightness / Luminosity / Contrast / Hue / Saturation)", open=False):
                # New Brightness
                enable_new_brightness = gr.Checkbox(label="Enable Brightness", value=False)
                new_brightness_factor = gr.Slider(
                    minimum=-150, 
                    maximum=150, 
                    step=1, 
                    value=0, 
                    label="Brightness Adjustment (-150 to +150)"
                )

                # Levels
                enable_levels = gr.Checkbox(label="Enable Levels", value=False)
                levels_strength = gr.Slider(
                    minimum=0.0, maximum=2.0, step=0.01, value=1.0,
                    label="Levels Strength (0=No Effect, 1=Full, 2=Extra)"
                )

                with gr.Row():
                    enable_luminosity = gr.Checkbox(label="Enable Luminosity Adjustment", value=False)
                    luminosity_factor = gr.Slider(minimum=-1.0, maximum=1.0, step=0.05, value=0, label="Luminosity Factor")

                    enable_contrast = gr.Checkbox(label="Enable Contrast Adjustment", value=False)
                    contrast_factor = gr.Slider(minimum=0.0, maximum=2.0, step=0.05, value=1, label="Contrast Factor")

                with gr.Row():
                    enable_hue = gr.Checkbox(label="Enable Hue Adjustment", value=False)
                    hue_factor = gr.Slider(minimum=-180, maximum=180, step=1, value=0, label="Hue Shift")

                    enable_saturation = gr.Checkbox(label="Enable Saturation Adjustment", value=False)
                    saturation_factor = gr.Slider(minimum=0.0, maximum=2.0, step=0.05, value=1, label="Saturation Factor")

            # ===================================
            # 预设 (保存/加载)
            # ===================================
            with gr.Accordion("Presets (保存/加载所有设置)", open=False):
                preset_name_text = gr.Textbox(label="输入预设名称")
                save_preset_button = gr.Button("Save Current Settings as Preset")

                preset_names = list(self.presets.keys())
                presets_dropdown = gr.Dropdown(label="Select a Preset to Load", choices=preset_names, value=None)
                load_preset_button = gr.Button("Load Selected Preset")

        # ========== 定义保存&加载预设的回调 ==========
        def on_save_preset(
            preset_name,
            save_original,
            enable_grain, enable_vignette, enable_random_blur, enable_color_offset,
            grain_intensity, vignette_intensity, vignette_feather, vignette_roundness,
            blur_max_size, blur_strength, color_offset_x, color_offset_y,
            enable_overlay, overlay_file, overlay_fit, overlay_opacity, overlay_blend_mode,
            enable_overlay2, overlay_file2, overlay_fit2, overlay_opacity2, overlay_blend_mode2,
            enable_overlay3, overlay_file3, overlay_fit3, overlay_opacity3, overlay_blend_mode3,
            enable_new_brightness, new_brightness_factor,
            enable_levels, levels_strength,
            enable_luminosity, luminosity_factor,
            enable_contrast, contrast_factor,
            enable_hue, hue_factor,
            enable_saturation, saturation_factor
        ):
            preset_name = (preset_name or "").strip()
            if not preset_name:
                return gr.update(), gr.update(choices=list(self.presets.keys()))

            current_settings = {
                "save_original": save_original,
                "enable_grain": enable_grain,
                "enable_vignette": enable_vignette,
                "enable_random_blur": enable_random_blur,
                "enable_color_offset": enable_color_offset,
                "grain_intensity": grain_intensity,
                "vignette_intensity": vignette_intensity,
                "vignette_feather": vignette_feather,
                "vignette_roundness": vignette_roundness,
                "blur_max_size": blur_max_size,
                "blur_strength": blur_strength,
                "color_offset_x": color_offset_x,
                "color_offset_y": color_offset_y,

                "enable_overlay": enable_overlay,
                "overlay_file": overlay_file,
                "overlay_fit": overlay_fit,
                "overlay_opacity": overlay_opacity,
                "overlay_blend_mode": overlay_blend_mode,

                "enable_overlay2": enable_overlay2,
                "overlay_file2": overlay_file2,
                "overlay_fit2": overlay_fit2,
                "overlay_opacity2": overlay_opacity2,
                "overlay_blend_mode2": overlay_blend_mode2,

                "enable_overlay3": enable_overlay3,
                "overlay_file3": overlay_file3,
                "overlay_fit3": overlay_fit3,
                "overlay_opacity3": overlay_opacity3,
                "overlay_blend_mode3": overlay_blend_mode3,

                "enable_new_brightness": enable_new_brightness,
                "new_brightness_factor": new_brightness_factor,

                "enable_levels": enable_levels,
                "levels_strength": levels_strength,

                "enable_luminosity": enable_luminosity,
                "luminosity_factor": luminosity_factor,

                "enable_contrast": enable_contrast,
                "contrast_factor": contrast_factor,

                "enable_hue": enable_hue,
                "hue_factor": hue_factor,

                "enable_saturation": enable_saturation,
                "saturation_factor": saturation_factor,
            }

            self.presets[preset_name] = current_settings
            save_presets_to_json(self.presets)
            print(f"Preset '{preset_name}' saved.")

            new_choices = list(self.presets.keys())
            return gr.update(value=""), gr.update(choices=new_choices, value=preset_name)

        def on_load_preset(preset_name):
            if not preset_name or (preset_name not in self.presets):
                return {}

            preset = self.presets[preset_name]
            print(f"Loading preset '{preset_name}' => {preset}")

            return {
                save_original: preset["save_original"],
                enable_grain: preset["enable_grain"],
                enable_vignette: preset["enable_vignette"],
                enable_random_blur: preset["enable_random_blur"],
                enable_color_offset: preset["enable_color_offset"],
                grain_intensity: preset["grain_intensity"],
                vignette_intensity: preset["vignette_intensity"],
                vignette_feather: preset["vignette_feather"],
                vignette_roundness: preset["vignette_roundness"],
                blur_max_size: preset["blur_max_size"],
                blur_strength: preset["blur_strength"],
                color_offset_x: preset["color_offset_x"],
                color_offset_y: preset["color_offset_y"],

                enable_overlay: preset["enable_overlay"],
                overlay_file: preset["overlay_file"],
                overlay_fit: preset["overlay_fit"],
                overlay_opacity: preset["overlay_opacity"],
                overlay_blend_mode: preset["overlay_blend_mode"],

                enable_overlay2: preset["enable_overlay2"],
                overlay_file2: preset["overlay_file2"],
                overlay_fit2: preset["overlay_fit2"],
                overlay_opacity2: preset["overlay_opacity2"],
                overlay_blend_mode2: preset["overlay_blend_mode2"],

                enable_overlay3: preset["enable_overlay3"],
                overlay_file3: preset["overlay_file3"],
                overlay_fit3: preset["overlay_fit3"],
                overlay_opacity3: preset["overlay_opacity3"],
                overlay_blend_mode3: preset["overlay_blend_mode3"],

                enable_new_brightness: preset["enable_new_brightness"],
                new_brightness_factor: preset["new_brightness_factor"],

                enable_levels: preset["enable_levels"],
                levels_strength: preset["levels_strength"],

                enable_luminosity: preset["enable_luminosity"],
                luminosity_factor: preset["luminosity_factor"],

                enable_contrast: preset["enable_contrast"],
                contrast_factor: preset["contrast_factor"],

                enable_hue: preset["enable_hue"],
                hue_factor: preset["hue_factor"],

                enable_saturation: preset["enable_saturation"],
                saturation_factor: preset["saturation_factor"],
            }

        # 按钮与回调绑定
        save_preset_button.click(
            fn=on_save_preset,
            inputs=[
                preset_name_text,
                save_original,
                enable_grain, enable_vignette, enable_random_blur, enable_color_offset,
                grain_intensity, vignette_intensity, vignette_feather, vignette_roundness,
                blur_max_size, blur_strength, color_offset_x, color_offset_y,
                enable_overlay, overlay_file, overlay_fit, overlay_opacity, overlay_blend_mode,
                enable_overlay2, overlay_file2, overlay_fit2, overlay_opacity2, overlay_blend_mode2,
                enable_overlay3, overlay_file3, overlay_fit3, overlay_opacity3, overlay_blend_mode3,
                enable_new_brightness, new_brightness_factor,
                enable_levels, levels_strength,
                enable_luminosity, luminosity_factor,
                enable_contrast, contrast_factor,
                enable_hue, hue_factor,
                enable_saturation, saturation_factor
            ],
            outputs=[preset_name_text, presets_dropdown],
        )

        load_preset_button.click(
            fn=on_load_preset,
            inputs=[presets_dropdown],
            outputs=[
                save_original,
                enable_grain, enable_vignette, enable_random_blur, enable_color_offset,
                grain_intensity, vignette_intensity, vignette_feather, vignette_roundness,
                blur_max_size, blur_strength, color_offset_x, color_offset_y,
                enable_overlay, overlay_file, overlay_fit, overlay_opacity, overlay_blend_mode,
                enable_overlay2, overlay_file2, overlay_fit2, overlay_opacity2, overlay_blend_mode2,
                enable_overlay3, overlay_file3, overlay_fit3, overlay_opacity3, overlay_blend_mode3,
                enable_new_brightness, new_brightness_factor,
                enable_levels, levels_strength,
                enable_luminosity, luminosity_factor,
                enable_contrast, contrast_factor,
                enable_hue, hue_factor,
                enable_saturation, saturation_factor
            ],
        )

        # 返回所有UI控件
        return [
            save_original,
            enable_grain, enable_vignette, enable_random_blur, enable_color_offset,
            grain_intensity, vignette_intensity, vignette_feather, vignette_roundness,
            blur_max_size, blur_strength, color_offset_x, color_offset_y,
            enable_overlay, overlay_file, overlay_fit, overlay_opacity, overlay_blend_mode,
            enable_overlay2, overlay_file2, overlay_fit2, overlay_opacity2, overlay_blend_mode2,
            enable_overlay3, overlay_file3, overlay_fit3, overlay_opacity3, overlay_blend_mode3,
            enable_new_brightness, new_brightness_factor,
            enable_levels, levels_strength,
            enable_luminosity, luminosity_factor,
            enable_contrast, contrast_factor,
            enable_hue, hue_factor,
            enable_saturation, saturation_factor,
            preset_name_text, save_preset_button, presets_dropdown, load_preset_button
        ]

    def process(self, p, *args):
        """
        在这里把启用的效果名字放到 p.extra_generation_params 里，以便在生成信息里显示
        """
        (
            save_original,
            enable_grain, enable_vignette, enable_random_blur, enable_color_offset,
            grain_intensity, vignette_intensity, vignette_feather, vignette_roundness,
            blur_max_size, blur_strength, color_offset_x, color_offset_y,
            enable_overlay, overlay_file, overlay_fit, overlay_opacity, overlay_blend_mode,
            enable_overlay2, overlay_file2, overlay_fit2, overlay_opacity2, overlay_blend_mode2,
            enable_overlay3, overlay_file3, overlay_fit3, overlay_opacity3, overlay_blend_mode3,
            enable_new_brightness, new_brightness_factor,
            enable_levels, levels_strength,
            enable_luminosity, luminosity_factor,
            enable_contrast, contrast_factor,
            enable_hue, hue_factor,
            enable_saturation, saturation_factor,
            preset_name_text, save_preset_button, presets_dropdown, load_preset_button
        ) = args

        enabled_effects = []
        if enable_grain:       enabled_effects.append("Grain")
        if enable_vignette:    enabled_effects.append("Vignette")
        if enable_random_blur: enabled_effects.append("Random Blur")
        if enable_color_offset:enabled_effects.append("Color Offset")
        if enable_overlay:     enabled_effects.append("Overlay L1")
        if enable_overlay2:    enabled_effects.append("Overlay L2")
        if enable_overlay3:    enabled_effects.append("Overlay L3")
        if enable_new_brightness: enabled_effects.append("Brightness (Curves)")
        if enable_levels:      enabled_effects.append("Levels")
        if enable_luminosity:  enabled_effects.append("Luminosity")
        if enable_contrast:    enabled_effects.append("Contrast")
        if enable_hue:         enabled_effects.append("Hue")
        if enable_saturation:  enabled_effects.append("Saturation")

        if enabled_effects:
            p.extra_generation_params["Some Image Effects"] = ", ".join(enabled_effects)

    def postprocess_image(self, p, pp, *args):
        """
        出图后再执行后处理
        """
        (
            save_original,
            enable_grain, grain_intensity,
            enable_vignette, vignette_intensity, vignette_feather, vignette_roundness,
            enable_random_blur, blur_max_size, blur_strength,
            enable_color_offset, color_offset_x, color_offset_y,
            enable_overlay, overlay_file, overlay_fit, overlay_opacity, overlay_blend_mode,
            enable_overlay2, overlay_file2, overlay_fit2, overlay_opacity2, overlay_blend_mode2,
            enable_overlay3, overlay_file3, overlay_fit3, overlay_opacity3, overlay_blend_mode3,
            enable_new_brightness, new_brightness_factor,
            enable_levels, levels_strength,
            enable_luminosity, luminosity_factor,
            enable_contrast, contrast_factor,
            enable_hue, hue_factor,
            enable_saturation, saturation_factor,
            preset_name_text, save_preset_button, presets_dropdown, load_preset_button
        ) = args

        # 处理单图或多图
        if hasattr(pp, 'image') and pp.image is not None:
            if save_original:
                self.save_original_image(pp.image)
            pp.image = self.add_effects(
                pp.image,
                enable_grain, grain_intensity,
                enable_vignette, vignette_intensity, vignette_feather, vignette_roundness,
                enable_random_blur, blur_max_size, blur_strength,
                enable_color_offset, color_offset_x, color_offset_y,
                enable_overlay, overlay_file, overlay_fit, overlay_opacity, overlay_blend_mode,
                enable_overlay2, overlay_file2, overlay_fit2, overlay_opacity2, overlay_blend_mode2,
                enable_overlay3, overlay_file3, overlay_fit3, overlay_opacity3, overlay_blend_mode3,
                enable_new_brightness, new_brightness_factor,
                enable_levels, levels_strength,
                enable_luminosity, luminosity_factor,
                enable_contrast, contrast_factor,
                enable_hue, hue_factor,
                enable_saturation, saturation_factor
            )
        elif hasattr(pp, 'images') and pp.images:
            for i, image in enumerate(pp.images):
                if save_original:
                    self.save_original_image(image)
                pp.images[i] = self.add_effects(
                    image,
                    enable_grain, grain_intensity,
                    enable_vignette, vignette_intensity, vignette_feather, vignette_roundness,
                    enable_random_blur, blur_max_size, blur_strength,
                    enable_color_offset, color_offset_x, color_offset_y,
                    enable_overlay, overlay_file, overlay_fit, overlay_opacity, overlay_blend_mode,
                    enable_overlay2, overlay_file2, overlay_fit2, overlay_opacity2, overlay_blend_mode2,
                    enable_overlay3, overlay_file3, overlay_fit3, overlay_opacity3, overlay_blend_mode3,
                    enable_new_brightness, new_brightness_factor,
                    enable_levels, levels_strength,
                    enable_luminosity, luminosity_factor,
                    enable_contrast, contrast_factor,
                    enable_hue, hue_factor,
                    enable_saturation, saturation_factor
                )

    def add_effects(
        self,
        img,
        enable_grain, grain_intensity,
        enable_vignette, vignette_intensity, vignette_feather, vignette_roundness,
        enable_random_blur, blur_max_size, blur_strength,
        enable_color_offset, color_offset_x, color_offset_y,
        enable_overlay, overlay_file, overlay_fit, overlay_opacity, overlay_blend_mode,
        enable_overlay2, overlay_file2, overlay_fit2, overlay_opacity2, overlay_blend_mode2,
        enable_overlay3, overlay_file3, overlay_fit3, overlay_opacity3, overlay_blend_mode3,
        enable_new_brightness, new_brightness_factor,
        enable_levels, levels_strength,
        enable_luminosity, luminosity_factor,
        enable_contrast, contrast_factor,
        enable_hue, hue_factor,
        enable_saturation, saturation_factor
    ):
        if img is None:
            return None

        # 1) 基础效果
        if enable_grain:
            img = self.add_grain(img, grain_intensity)
        if enable_vignette:
            img = self.add_vignette(img, vignette_intensity, vignette_feather, vignette_roundness)
        if enable_random_blur:
            img = self.add_random_blur(img, blur_max_size, blur_strength)
        if enable_color_offset:
            img = self.add_color_offset(img, color_offset_x, color_offset_y)

        # 2) Overlay(s)
        if enable_overlay and overlay_file:
            img = self.add_overlay(img, overlay_file, overlay_fit, overlay_opacity, overlay_blend_mode)
        if enable_overlay2 and overlay_file2:
            img = self.add_overlay(img, overlay_file2, overlay_fit2, overlay_opacity2, overlay_blend_mode2)
        if enable_overlay3 and overlay_file3:
            img = self.add_overlay(img, overlay_file3, overlay_fit3, overlay_opacity3, overlay_blend_mode3)

        # 3) 调整 (Brightness / Levels / Luminosity / Contrast / Hue / Saturation)
        if enable_new_brightness and abs(new_brightness_factor) > 1e-6:
            img = self.adjust_brightness_new(img, new_brightness_factor)

        if enable_levels:
            img = self.apply_levels_with_strength(img, strength=levels_strength)

        if enable_luminosity:
            img = self.adjust_luminosity(img, luminosity_factor)

        if enable_contrast:
            img = self.adjust_contrast(img, contrast_factor)

        if enable_hue:
            img = self.adjust_hue(img, hue_factor)

        if enable_saturation:
            img = self.adjust_saturation(img, saturation_factor)

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
            return img
        x = random.randint(0, width - blur_size)
        y = random.randint(0, height - blur_size)
        mask = Image.new('L', (width, height), 0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse([x, y, x + blur_size, y + blur_size], fill=255)
        mask = mask.filter(ImageFilter.GaussianBlur(radius=blur_size // 4 if blur_size>4 else 1))
        blurred = img.filter(ImageFilter.GaussianBlur(radius=strength))
        return Image.composite(blurred, img, mask)

    def add_color_offset(self, img, offset_x, offset_y):
        r, g, b = img.split()
        r = ImageChops.offset(r, offset_x, offset_y)
        b = ImageChops.offset(b, -offset_x, -offset_y)
        return Image.merge('RGB', (r, g, b))

    def add_overlay(self, img, overlay_file, overlay_fit, opacity, blend_mode):
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
            left = (overlay.width - img.width) // 2
            top = (overlay.height - img.height) // 2
            right = left + img.width
            bottom = top + img.height
            overlay = overlay.crop((left, top, right, bottom))

        overlay = Image.blend(Image.new('RGBA', img.size, (0, 0, 0, 0)), overlay, opacity)

        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        if overlay.mode != 'RGBA':
            overlay = overlay.convert('RGBA')

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
    # ============== 调整函数 =======================
    ################################################

    def adjust_luminosity(self, img, factor):
        return ImageEnhance.Brightness(img).enhance(1 + factor)

    def adjust_contrast(self, img, factor):
        return ImageEnhance.Contrast(img).enhance(factor)

    def adjust_hue(self, img, shift):
        """
        shift: [-180..180]
        """
        hsv = img.convert('HSV')
        h, s, v = hsv.split()

        def shift_hue(x):
            return (x + int(shift * 255/360)) % 256

        h = h.point(shift_hue)
        return Image.merge('HSV', (h, s, v)).convert('RGB')

    def adjust_saturation(self, img, factor):
        return ImageEnhance.Color(img).enhance(factor)

    # =======================================
    # 用 Curves 来做 Brightness（保纯黑纯白）
    # =======================================
    def adjust_brightness_new(self, img, factor):
        if factor == 0:
            return img
        if img.mode != 'RGB':
            img = img.convert("RGB")
        
        arr = np.array(img, dtype=np.float32) / 255.0
        scaled_factor = factor / 150.0  # 将-150~+150映射到-1.0~+1.0
        
        # 应用非线性调整曲线
        # 使用三次贝塞尔曲线调整，增强中间调响应
        adjusted = arr + scaled_factor * (arr ** 2) * (1.0 - arr) * 4.0
        
        # 确保数值在有效范围内
        adjusted = np.clip(adjusted, 0.0, 1.0)
        
        return Image.fromarray((adjusted * 255).astype(np.uint8), "RGB")

    # 色阶(多一步strength插值)
    def apply_levels_with_strength(self, img, strength=1.0,
                                   in_black=0, in_white=255, in_gamma=1.60,
                                   out_black=0, out_white=255):
        """
        修复后的色阶强度混合实现，并增强暗部饱和度
        """
        img_levels = self.apply_levels(img, in_black, in_white, in_gamma, out_black, out_white)
        
        if abs(strength - 1.0) > 1e-6:
            if strength < 1.0:  # 与原图混合
                return Image.blend(img, img_levels, strength)
            else:  # 超过1时的特殊处理
                return Image.blend(img_levels, img, 2 - strength)
        return img_levels

    def apply_levels(self, img, in_black, in_white, in_gamma, out_black, out_white):
        if img.mode != "RGB":
            img = img.convert("RGB")
        arr = np.array(img, dtype=np.float32)
        arr /= 255.0
        
        # 输入黑/白映射
        ib = in_black / 255.0
        iw = in_white / 255.0
        iw = max(iw, ib + 1e-5)  # 避免除零
        arr = (arr - ib) / (iw - ib)
        arr = np.clip(arr, 0.0, 1.0)
        
        # Gamma校正
        arr **= (1.0 / in_gamma)
        
        # 输出黑/白映射
        ob = out_black / 255.0
        ow = out_white / 255.0
        arr = arr * (ow - ob) + ob
        arr = np.clip(arr, 0.0, 1.0)
        
        # 增强暗部饱和度
        # 计算亮度（使用加权平均）
        Y = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
        Y_expanded = Y[..., np.newaxis]  # 扩展维度以匹配颜色通道
        
        # 动态饱和度增益：暗部增强更多（示例系数0.6，可根据需求调整）
        saturation_factor = 1.0 + (1 - Y_expanded) * 0.6
        
        # 应用饱和度调整
        arr = Y_expanded + (arr - Y_expanded) * saturation_factor
        arr = np.clip(arr, 0.0, 1.0)
        
        # 转换回PIL格式
        return Image.fromarray((arr * 255).astype(np.uint8), "RGB")

    ################################################
    # 各种混合模式
    ################################################

    def overlay_blend(self, base, top, opacity=1.0):
        base_np = np.array(base, dtype=np.float32)/255.0
        top_np = np.array(top, dtype=np.float32)/255.0
        out_np = np.zeros_like(base_np)
        for c in range(3):
            b = base_np[:, :, c]
            t = top_np[:, :, c]
            result = np.where(b<0.5, 2.0*b*t, 1.0-2.0*(1.0-b)*(1.0-t))
            out_np[:, :, c] = (1.0 - opacity)*b + opacity*result
        alpha_base = base_np[:, :, 3]
        alpha_top = top_np[:, :, 3]*opacity
        out_np[:, :, 3] = alpha_base + alpha_top*(1 - alpha_base)
        out_np = (out_np*255.0).clip(0,255).astype(np.uint8)
        return Image.fromarray(out_np, mode='RGBA')

    def color_dodge(self, base, top):
        base_np = np.array(base, dtype=np.float32)/255.0
        top_np = np.array(top, dtype=np.float32)/255.0
        out = np.zeros_like(base_np)
        eps = 1e-5
        for c in range(3):
            b = base_np[:, :, c]
            t = top_np[:, :, c]
            out[:, :, c] = np.where(t<1.0-eps, np.minimum(1.0, b/(1.0-t+eps)), 1.0)
        out[:, :, 3] = top_np[:, :, 3]
        out = (out*255.0).clip(0,255).astype(np.uint8)
        return Image.fromarray(out, 'RGBA')

    def color_burn(self, base, top):
        base_np = np.array(base, dtype=np.float32)/255.0
        top_np = np.array(top, dtype=np.float32)/255.0
        out = np.zeros_like(base_np)
        eps = 1e-5
        for c in range(3):
            b = base_np[:, :, c]
            t = top_np[:, :, c]
            out[:, :, c] = np.where(t>eps, 1.0 - np.minimum(1.0, (1.0-b)/t), 0.0)
        out[:, :, 3] = top_np[:, :, 3]
        out = (out*255.0).clip(0,255).astype(np.uint8)
        return Image.fromarray(out, 'RGBA')

    def hard_light(self, base, top):
        base_np = np.array(base, dtype=np.float32)/255.0
        top_np = np.array(top, dtype=np.float32)/255.0
        out = np.zeros_like(base_np)
        for c in range(3):
            b = base_np[:, :, c]
            t = top_np[:, :, c]
            mask = (t<=0.5)
            temp = np.zeros_like(b)
            temp[mask] = 2.0*b[mask]*t[mask]
            temp[~mask] = 1.0 - 2.0*(1.0-b[~mask])*(1.0-t[~mask])
            out[:, :, c] = temp
        out[:, :, 3] = top_np[:, :, 3]
        out = (out*255.0).clip(0,255).astype(np.uint8)
        return Image.fromarray(out, 'RGBA')

    def soft_light(self, base, top):
        base_np = np.array(base, dtype=np.float32)/255.0
        top_np = np.array(top, dtype=np.float32)/255.0
        out = np.zeros_like(base_np)
        for c in range(3):
            b = base_np[:, :, c]
            t = top_np[:, :, c]
            out[:, :, c] = (1.0 - 2.0*t)*(b**2) + 2.0*t*b
        out[:, :, 3] = top_np[:, :, 3]
        out = (out*255.0).clip(0,255).astype(np.uint8)
        return Image.fromarray(out, 'RGBA')

    def exclusion(self, base, top):
        base_np = np.array(base, dtype=np.float32)/255.0
        top_np = np.array(top, dtype=np.float32)/255.0
        out = np.zeros_like(base_np)
        for c in range(3):
            b = base_np[:, :, c]
            t = top_np[:, :, c]
            out[:, :, c] = b + t - 2.0*b*t
        out[:, :, 3] = top_np[:, :, 3]
        out = (out*255.0).clip(0,255).astype(np.uint8)
        return Image.fromarray(out, 'RGBA')

    def hue_blend(self, base, top):
        base_np = np.array(base, dtype=np.float32)
        top_np = np.array(top, dtype=np.float32)
        out = np.zeros_like(base_np)
        h, w, _ = base_np.shape
        for i in range(h):
            for j in range(w):
                br, bg, bb, ba = base_np[i,j]/255.0
                tr, tg, tb, ta = top_np[i,j]/255.0
                bh, bs, bl = rgb_to_hsl(br, bg, bb)
                th, ts, tl = rgb_to_hsl(tr, tg, tb)
                nh, ns, nl = th, bs, bl  # 用 top 的 hue
                rr, gg, bb = hsl_to_rgb(nh, ns, nl)
                out[i,j] = (rr*255, gg*255, bb*255, ta*255)
        return Image.fromarray(out.astype(np.uint8), 'RGBA')

    def saturation_blend(self, base, top):
        base_np = np.array(base, dtype=np.float32)
        top_np = np.array(top, dtype=np.float32)
        out = np.zeros_like(base_np)
        h, w, _ = base_np.shape
        for i in range(h):
            for j in range(w):
                br, bg, bb, ba = base_np[i,j]/255.0
                tr, tg, tb, ta = top_np[i,j]/255.0
                bh, bs, bl = rgb_to_hsl(br, bg, bb)
                th, ts, tl = rgb_to_hsl(tr, tg, tb)
                nh, ns, nl = bh, ts, bl  # 用 top 的 saturation
                rr, gg, bb = hsl_to_rgb(nh, ns, nl)
                out[i,j] = (rr*255, gg*255, bb*255, ta*255)
        return Image.fromarray(out.astype(np.uint8), 'RGBA')

    def color_blend(self, base, top):
        base_np = np.array(base, dtype=np.float32)
        top_np = np.array(top, dtype=np.float32)
        out = np.zeros_like(base_np)
        h, w, _ = base_np.shape
        for i in range(h):
            for j in range(w):
                br, bg, bb, ba = base_np[i,j]/255.0
                tr, tg, tb, ta = top_np[i,j]/255.0
                bh, bs, bl = rgb_to_hsl(br, bg, bb)
                th, ts, tl = rgb_to_hsl(tr, tg, tb)
                nh, ns, nl = th, ts, bl  # 用 top 的 hue + saturation
                rr, gg, bb = hsl_to_rgb(nh, ns, nl)
                out[i,j] = (rr*255, gg*255, bb*255, ta*255)
        return Image.fromarray(out.astype(np.uint8), 'RGBA')

    def luminosity_blend(self, base, top):
        base_np = np.array(base, dtype=np.float32)
        top_np = np.array(top, dtype=np.float32)
        out = np.zeros_like(base_np)
        h, w, _ = base_np.shape
        for i in range(h):
            for j in range(w):
                br, bg, bb, ba = base_np[i,j]/255.0
                tr, tg, tb, ta = top_np[i,j]/255.0
                bh, bs, bl = rgb_to_hsl(br, bg, bb)
                th, ts, tl = rgb_to_hsl(tr, tg, tb)
                nh, ns, nl = bh, bs, tl  # 用 top 的 lightness
                rr, gg, bb = hsl_to_rgb(nh, ns, nl)
                out[i,j] = (rr*255, gg*255, bb*255, ta*255)
        return Image.fromarray(out.astype(np.uint8), 'RGBA')

    ################################################
    # 保存原图
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
        
        job_name = shared.state.job if shared.state.job else "image"
        base_name, ext = os.path.splitext(job_name)
        if not ext:
            ext = ".png"
        
        filename = f"{base_name}_original{ext}"
        save_path = os.path.join(save_dir, filename)
        
        try:
            img.save(save_path)
            print(f"Original image saved to: {save_path}")
        except Exception as e:
            print(f"Error saving original image: {e}")
