"""
weights for RealESRGAN_x4plus.pth, openai/clip-vit-large-patch14 and stable-diffusion sd-v1-4.ckpt are downloaded to ./weights
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P ./weights
in ./stable-diffusion/ldm/modules/encoders/modules.py, load from local weights (local_files_only=True) for FrozenCLIPEmbedder()
"""
import os
import random
import sys
import gc
import shutil
import numpy as np
from PIL import Image, ImageDraw
from omegaconf import OmegaConf
from tqdm import tqdm, trange
import torch
from torch import autocast
from pytorch_lightning import seed_everything
import cv2
from einops import rearrange, repeat
from basicsr.archs.rrdbnet_arch import RRDBNet
from cog import BasePredictor, Path, Input

sys.path.append("./Real-ESRGAN")
from realesrgan.utils import RealESRGANer

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler


class Predictor(BasePredictor):
    def setup(self):

        config_file = "configs/stable-diffusion/v1-inference.yaml"
        checkpoint = "weights/sd-v1-4.ckpt"

        config = OmegaConf.load(config_file)
        self.model = load_model_from_config(config, checkpoint)
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.model = self.model.to(self.device)

        self.ddim_sampler = DDIMSampler(self.model)
        self.plms_sampler = DDIMSampler(self.model)

        realesrgan_model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=4,
        )
        realesrgan_model_path = "weights/RealESRGAN_x4plus.pth"
        self.realesrgan_upsampler = RealESRGANer(
            scale=4,
            model_path=realesrgan_model_path,
            model=realesrgan_model,
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=True,
        )

    def predict(
        self,
        prompt: str = Input(
            default="female cyborg assimilated by alien fungus, intricate Three-point lighting portrait, by Ching Yeh and Greg Rutkowski, detailed cyberpunk in the style of GitS 1995",
            description="The prompt to render.",
        ),
        scale: float = Input(
            default=7.5,
            description="Unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty)).",
        ),
        steps: int = Input(
            default=50,
            description="Number of sampling steps.",
        ),
        seed: int = Input(
            description="The seed (for reproducible sampling).",
            default=None,
        ),
    ) -> Path:

        if seed is None:
            seed = random.randint(0, 2**32 - 1)
        print(
            "Using seed {}. Enter this in 'seed' if you want to produce the same output again!".format(
                seed
            )
        )

        seed_everything(seed)

        # use the default setting for the following params for the demo
        n_iter = 1  # sample this often
        C = 4  # latent channels
        f = 8  # downsampling factor, most often 8 or 16
        H = 512  # image height, in pixel space
        W = 512  # image width, in pixel space
        strength = 0.3  # strength for noising/unnoising
        precision_scope = autocast

        # default setting for the upscaling
        detail_scale = 10  # unconditional guidance scale when detailing: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))
        detail_steps = 150  # number of sampling steps when detailing
        gobig_overlap = 128  # overlap size for GOBIG
        passes = 1  # number of upscales/details

        outpath = "cog_temp_out"
        if os.path.exists(outpath):
            shutil.rmtree(outpath)
        sample_path = os.path.join(outpath, "samples")
        os.makedirs(sample_path, exist_ok=True)

        batch_size = 1
        base_count = len(os.listdir(sample_path))
        data = [batch_size * [prompt]]

        generated = []  # only do detailing, using these base filenames in output dir
        with torch.inference_mode():
            with precision_scope("cuda"):
                with self.model.ema_scope():
                    for _ in trange(n_iter, desc="Sampling"):
                        for prompts in tqdm(data, desc="data"):
                            uc = None
                            if scale != 1.0:
                                uc = self.model.get_learned_conditioning(
                                    batch_size * [""]
                                )
                            if isinstance(prompts, tuple):
                                prompts = list(prompts)
                            c = self.model.get_learned_conditioning(prompts)
                            shape = [C, H // f, W // f]
                            samples_ddim, _ = self.plms_sampler.sample(
                                S=steps,
                                conditioning=c,
                                batch_size=batch_size,
                                shape=shape,
                                verbose=False,
                                unconditional_guidance_scale=scale,
                                unconditional_conditioning=uc,
                                eta=0,
                                x_T=None,
                            )

                            x_samples_ddim = self.model.decode_first_stage(samples_ddim)
                            x_samples_ddim = torch.clamp(
                                (x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0
                            )
                            x_samples_ddim = (
                                x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()
                            )

                            x_checked_image = x_samples_ddim

                            x_checked_image_torch = torch.from_numpy(
                                x_checked_image
                            ).permute(0, 3, 1, 2)

                            for x_sample in x_checked_image_torch:
                                x_sample = 255.0 * rearrange(
                                    x_sample.cpu().numpy(), "c h w -> h w c"
                                )
                                img = Image.fromarray(x_sample.astype(np.uint8))
                                output_path = os.path.join(
                                    sample_path, f"{base_count:05}.png"
                                )
                                img.save(output_path)
                                generated.append(f"{base_count:05}")
                                base_count += 1

        torch.cuda.empty_cache()
        gc.collect()

        out_path = f"/tmp/out.png"

        for base_filename in generated:
            for _ in trange(passes, desc="Passes"):
                # realesrgan2x(realesrgan, os.path.join(sample_path, f"{base_filename}.png"), os.path.join(sample_path, f"{base_filename}u.png"))

                realesrgan_enhance(
                    os.path.join(sample_path, f"{base_filename}.png"),
                    os.path.join(sample_path, f"{base_filename}u.png"),
                    self.realesrgan_upsampler,
                )
                base_filename = f"{base_filename}u"

                source_image = Image.open(
                    os.path.join(sample_path, f"{base_filename}.png")
                )
                og_size = (H, W)
                slices, _ = grid_slice(source_image, gobig_overlap, og_size, False)

                betterslices = []
                for _, chunk_w_coords in tqdm(enumerate(slices), "Slices"):
                    chunk, coord_x, coord_y = chunk_w_coords
                    init_image = convert_pil_img(chunk).to(self.device)
                    init_image = repeat(init_image, "1 ... -> b ...", b=batch_size)
                    init_latent = self.model.get_first_stage_encoding(
                        self.model.encode_first_stage(init_image)
                    )  # move to latent space

                    self.ddim_sampler.make_schedule(
                        ddim_num_steps=detail_steps, ddim_eta=0, verbose=False
                    )

                    assert (
                        0.0 <= strength <= 1.0
                    ), "can only work with strength in [0.0, 1.0]"
                    t_enc = int(strength * detail_steps)

                    with torch.inference_mode():
                        with precision_scope("cuda"):
                            with self.model.ema_scope():
                                for prompts in tqdm(data, desc="data"):
                                    uc = None
                                    if detail_scale != 1.0:
                                        uc = self.model.get_learned_conditioning(
                                            batch_size * [""]
                                        )
                                    if isinstance(prompts, tuple):
                                        prompts = list(prompts)
                                    c = self.model.get_learned_conditioning(prompts)

                                    # encode (scaled latent)
                                    z_enc = self.ddim_sampler.stochastic_encode(
                                        init_latent,
                                        torch.tensor([t_enc] * batch_size).to(
                                            self.device
                                        ),
                                    )
                                    # decode it
                                    samples = self.ddim_sampler.decode(
                                        z_enc,
                                        c,
                                        t_enc,
                                        unconditional_guidance_scale=detail_scale,
                                        unconditional_conditioning=uc,
                                    )

                                    x_samples = self.model.decode_first_stage(samples)
                                    x_samples = torch.clamp(
                                        (x_samples + 1.0) / 2.0, min=0.0, max=1.0
                                    )

                                    for x_sample in x_samples:
                                        x_sample = 255.0 * rearrange(
                                            x_sample.cpu().numpy(), "c h w -> h w c"
                                        )
                                        resultslice = Image.fromarray(
                                            x_sample.astype(np.uint8)
                                        ).convert("RGBA")
                                        betterslices.append(
                                            (resultslice.copy(), coord_x, coord_y)
                                        )

                alpha = Image.new("L", og_size, color=0xFF)
                alpha_gradient = ImageDraw.Draw(alpha)
                a = 0
                i = 0
                overlap = gobig_overlap
                shape = (og_size, (0, 0))
                while i < overlap:
                    alpha_gradient.rectangle(shape, fill=a)
                    a += 4
                    i += 1
                    shape = ((og_size[0] - i, og_size[1] - i), (i, i))
                mask = Image.new("RGBA", og_size, color=0)
                mask.putalpha(alpha)
                finished_slices = []
                for betterslice, x, y in betterslices:
                    finished_slice = addalpha(betterslice, mask)
                    finished_slices.append((finished_slice, x, y))
                # # Once we have all our images, use grid_merge back onto the source, then save
                final_output = grid_merge(
                    source_image.convert("RGBA"), finished_slices
                ).convert("RGB")
                final_output.save(os.path.join(sample_path, f"{base_filename}d.png"))
                base_filename = f"{base_filename}d"

                torch.cuda.empty_cache()
                gc.collect()

            # put_watermark(final_output, wm_encoder)
            final_output.save(os.path.join(sample_path, f"{base_filename}.png"))
            final_output.save(out_path)

        return Path(out_path)


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def realesrgan_enhance(img, out, upsampler):
    img = cv2.imread(str(img), cv2.IMREAD_UNCHANGED)
    h, w = img.shape[0:2]
    if h < 300:
        img = cv2.resize(img, (w * 2, h * 2), interpolation=cv2.INTER_LANCZOS4)

    try:
        scale = 4
        output, _ = upsampler.enhance(img, outscale=scale)

        cv2.imwrite(out, output)
        final_output = Image.open(out)
        final_output = final_output.resize(
            (int(final_output.size[0] / 2), int(final_output.size[1] / 2)),
            get_resampling_mode(),
        )
        final_output.save(out)

    except RuntimeError as error:
        print("Error", error)
        print(
            'If you encounter CUDA out of memory, try to set "tile" to a smaller size, e.g., 400.'
        )


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def convert_pil_img(image):
    w, h = image.size
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=Image.Resampling.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.0 * image - 1.0


def addalpha(im, mask):
    imr, img, imb, ima = im.split()
    mmr, mmg, mmb, mma = mask.split()
    im = Image.merge(
        "RGBA", [imr, img, imb, mma]
    )  # we want the RGB from the original, but the transparency from the mask
    return im


# Alternative method composites a grid of images at the positions provided
def grid_merge(source, slices):
    source.convert("RGBA")
    for slice, posx, posy in slices:  # go in reverse to get proper stacking
        source.alpha_composite(slice, (posx, posy))
    return source


def grid_coords(target, original, overlap):
    # generate a list of coordinate tuples for our sections, in order of how they'll be rendered
    # target should be the size for the gobig result, original is the size of each chunk being rendered
    center = []
    target_x, target_y = target
    center_x = int(target_x / 2)
    center_y = int(target_y / 2)
    original_x, original_y = original
    x = center_x - int(original_x / 2)
    y = center_y - int(original_y / 2)
    center.append((x, y))  # center chunk
    uy = y  # up
    uy_list = []
    dy = y  # down
    dy_list = []
    lx = x  # left
    lx_list = []
    rx = x  # right
    rx_list = []
    while uy > 0:  # center row vertical up
        uy = uy - original_y + overlap
        uy_list.append((lx, uy))
    while (dy + original_y) <= target_y:  # center row vertical down
        dy = dy + original_y - overlap
        dy_list.append((rx, dy))
    while lx > 0:
        lx = lx - original_x + overlap
        lx_list.append((lx, y))
        uy = y
        while uy > 0:
            uy = uy - original_y + overlap
            uy_list.append((lx, uy))
        dy = y
        while (dy + original_y) <= target_y:
            dy = dy + original_y - overlap
            dy_list.append((lx, dy))
    while (rx + original_x) <= target_x:
        rx = rx + original_x - overlap
        rx_list.append((rx, y))
        uy = y
        while uy > 0:
            uy = uy - original_y + overlap
            uy_list.append((rx, uy))
        dy = y
        while (dy + original_y) <= target_y:
            dy = dy + original_y - overlap
            dy_list.append((rx, dy))
    # calculate a new size that will fill the canvas, which will be optionally used in grid_slice and go_big
    last_coordx, last_coordy = dy_list[-1:][0]
    render_edgey = last_coordy + original_y  # outer bottom edge of the render canvas
    render_edgex = last_coordx + original_x  # outer side edge of the render canvas
    scalarx = render_edgex / target_x
    scalary = render_edgey / target_y
    if scalarx <= scalary:
        new_edgex = int(target_x * scalarx)
        new_edgey = int(target_y * scalarx)
    else:
        new_edgex = int(target_x * scalary)
        new_edgey = int(target_y * scalary)
    # now put all the chunks into one master list of coordinates (essentially reverse of how we calculated them so that the central slices will be on top)
    result = []
    for coords in dy_list[::-1]:
        result.append(coords)
    for coords in uy_list[::-1]:
        result.append(coords)
    for coords in rx_list[::-1]:
        result.append(coords)
    for coords in lx_list[::-1]:
        result.append(coords)
    result.append(center[0])
    return result, (new_edgex, new_edgey)


def get_resampling_mode():
    try:
        from PIL import __version__, Image

        major_ver = int(__version__.split(".")[0])
        if major_ver >= 9:
            return Image.Resampling.LANCZOS
        else:
            return Image.LANCZOS
    except Exception as ex:
        return 1  # 'Lanczos' irrespective of version.


# Chop our source into a grid of images that each equal the size of the original render
def grid_slice(source, overlap, og_size, maximize=False):
    width, height = og_size  # size of the slices to be rendered
    coordinates, new_size = grid_coords(source.size, og_size, overlap)
    if maximize == True:
        source = source.resize(
            new_size, get_resampling_mode()
        )  # minor concern that we're resizing twice
        coordinates, new_size = grid_coords(
            source.size, og_size, overlap
        )  # re-do the coordinates with the new canvas size
    # loc_width and loc_height are the center point of the goal size, and we'll start there and work our way out
    slices = []
    for coordinate in coordinates:
        x, y = coordinate
        slices.append(((source.crop((x, y, x + width, y + height))), x, y))
    global slices_todo
    slices_todo = len(slices) - 1
    return slices, new_size
