import os.path

import gradio as gr
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from diffusers.utils import logging
from scripts.safety_checker import StableDiffusionSafetyChecker
from transformers import AutoFeatureExtractor

from modules import scripts
from modules.scripts import PostprocessImageArgs

logger = logging.get_logger(__name__)

safety_model_id = "/home/ucloud/stable-diffusion-webui/models/huggingface/nsfw_detector/"
safety_feature_extractor = None
safety_checker = None

warning_image = os.path.join("extensions", "stable-diffusion-webui-nsfw-filter", "warning", "warning.png")
currnt_path = os.path.dirname(os.path.abspath(__file__)) + "/"


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def add_watermark(img, watermark_text):
    width, height = img.size
    print(f'image size: {width}x{height}')

    def get_offset(text_fontsize):
        return text_fontsize * 0.8

    def get_fontsize(w, h):
        min_len = min(w, h)
        return ((min_len * 10) / 512) + 1

    fontsize = get_fontsize(width, height)
    font = ImageFont.truetype(currnt_path + 'SourceHanSansCN-Normal.otf', int(fontsize))
    content_len = font.getlength(watermark_text)
    bbox = font.getbbox(watermark_text)
    txt_height = bbox[3] - bbox[1]

    x = width - content_len - get_offset(fontsize)
    y = height - txt_height - get_offset(fontsize)

    blurred = Image.new("RGBA", (width, height))
    draw = ImageDraw.Draw(blurred)
    draw.text((x, y), watermark_text, font=font, fill=(0, 0, 0, 40))
    blurred = blurred.filter(ImageFilter.BoxBlur(4))
    img.paste(blurred, (0, 0), blurred)

    draw = ImageDraw.Draw(img)
    draw.text((x, y), watermark_text, font=font, fill=(245, 245, 245, 200))

    return img


# check and replace nsfw content
def check_safety(x_image, safety_checker_adj: float):
    global safety_feature_extractor, safety_checker

    if safety_feature_extractor is None:
        safety_feature_extractor = AutoFeatureExtractor.from_pretrained(safety_model_id)
        safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_model_id)

    safety_checker_input = safety_feature_extractor(numpy_to_pil(x_image), return_tensors="pt")
    x_checked_image, has_nsfw_concept = safety_checker(
        images=x_image,
        clip_input=safety_checker_input.pixel_values,
        safety_checker_adj=safety_checker_adj,  # customize adjustment
    )

    return x_checked_image, has_nsfw_concept


def censor_batch(x, safety_checker_adj: float):
    x_samples_ddim_numpy = x.cpu().permute(0, 2, 3, 1).numpy()
    x_checked_image, has_nsfw_concept = check_safety(x_samples_ddim_numpy, safety_checker_adj)
    x = torch.from_numpy(x_checked_image).permute(0, 3, 1, 2)

    index = 0
    for unsafe_value in has_nsfw_concept:
        try:
            if unsafe_value is True:
                hwc = x.shape
                y = Image.open(warning_image).convert("RGB").resize((hwc[3], hwc[2]))
                y = (np.array(y) / 255.0).astype("float32")
                y = torch.from_numpy(y)
                y = torch.unsqueeze(y, 0).permute(0, 3, 1, 2)
                x[index] = y
            index += 1
        except Exception as e:
            logger.warning(e)
            index += 1

    return x


class NsfwCheckScript(scripts.Script):

    def title(self):
        return "NSFW check"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def postprocess_batch(self, p, *args, **kwargs):
        """
        Args:
            p:
            *args:
                args[0]: enable_nsfw_filer. True: NSFW filter enabled; False: NSFW filter disabled
                args[1]: safety_checker_adj
            **kwargs:
        Returns:
            images
        """

        images = kwargs['images']

        if args[0] is True:
            images[:] = censor_batch(images, args[1])[:]

    def postprocess_image(self, p, pp: PostprocessImageArgs, *args):
        if len(args) == 3 and args[2] is not None and args[2] != "":
            pp.image = add_watermark(pp.image, args[2])

    def ui(self, is_img2img):
        enable_nsfw_filer = gr.Checkbox(label='Enable NSFW filter',
                                        value=False,
                                        elem_id=self.elem_id("enable_nsfw_filer"))
        safety_checker_adj = gr.Slider(label="Safety checker adjustment",
                                       minimum=-0.5, maximum=0.5, value=0.0, step=0.001,
                                       elem_id=self.elem_id("safety_checker_adj"))
        watermark = gr.Textbox(label="Watermark text", default="",
                               elem_id=self.elem_id("watermark"))
        return [enable_nsfw_filer, safety_checker_adj, watermark]
