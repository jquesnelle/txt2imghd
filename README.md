# txt2imghd

[![Replicate](https://replicate.com/cjwbw/stable-diffusion-high-resolution/badge)](https://replicate.com/cjwbw/stable-diffusion-high-resolution)

txt2imghd is a port of the GOBIG mode from [progrockdiffusion](https://github.com/lowfuel/progrockdiffusion) applied to [Stable Diffusion](https://github.com/CompVis/stable-diffusion), with [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) as the upscaler. It creates detailed, higher-resolution images by first generating an image from a prompt, upscaling it, and then running img2img on smaller pieces of the upscaled image, and blending the result back into the original image.

txt2imghd with default settings has the same VRAM requirements as regular Stable Diffusion, although generation of the detailed images will take longer.

## Installation

1. Have a working repository of [Stable Diffusion](https://raw.githubusercontent.com/CompVis/stable-diffusion)
2. Copy `txt2imghd.py` into `scripts/`
3. Download the appropriate [release of Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN/releases) (the respective `realesrgan-ncnn-vulkan` .zip for your OS) and unzip it into the root of your Stable Diffusion repository

## Running

txt2imghd has most of the same parameters as txt2img. `ddim_steps` has been renamed to `steps`. The `strength` parameter controls how much detailing to do (between 0.0-1.0). If no `prompt` is given on the command line, the program will ask for it as input.

```sh
python scripts/txt2imghd.py
```

txt2imghd will output three images: the original Stable Diffusion image, the upscaled version (denoted by a `u` suffix), and the detailed version (denoted by the `ud` suffix).

If you're running into issues with [WatermarkEncoder](https://pypi.org/project/invisible-watermark/), install WatermarkEncoder in your ldm environment with
`pip install invisible-watermark`

### Optional Parameters

A selection of useful parameters to be appended after `python scripts/txt2imghd.py`:

`--prompt` the prompt to render (in quotes), examples  [below](#example-images--promts)

`--img` only do detailing, using the path to an existing image (image will also be copied to output dir)

`--generated` only do detailing, on a an image in the output folder, using the image's index (example "00003")

`--n_iter 25` number of images to generate\
*default = 1*

`--gobig_overlap` overlap size for GOBIG\
*default = 128*

`--detail_steps` number of sampling steps when detailing\
*default = 150*

`--wm` watermark text using WatermarkEncoder\
*default = "txt2imghd"*

`--passes` number of upscaling/detailing passes\
*default = 1* 

`--strength` strength for noising/unnoising. 1.0 corresponds to full destruction of information in init image (especially useful when using an existing image)\
*default = 0.3*

## Example images / Promts

[old harbour, tone mapped, shiny, intricate, cinematic lighting, highly detailed, digital painting, artstation, concept art, smooth, sharp focus, illustration, art by terry moore and greg rutkowski and alphonse mucha](gallery/00005ud.png)

[55mm closeup hand photo of a breathtaking majestic beautiful armored redhead woman mage holding a tiny ball of fire in her hand on a snowy night in the village. zoom on the hand. focus on hand. dof. bokeh. art by greg rutkowski and luis royo. ultra reallistic. extremely detailed. nikon d850. cinematic postprocessing.](gallery/00030ud.png)

[a humanoid armored futuristic cybernetic samurai with glowing neon decals, award winning photograph, close up, focused trending on artstation, octane render, portrait, hyperrealistic, ultra detailed, photograph](gallery/00068ud.png)

[(painting of girl from behind looking a fleet of imperial ships in the sky, in a meadow of flowers. ) by donato giancola and Eddie Mendoza,  elegant, dynamic lighting, beautiful, poster, trending on artstation, poster, anato finnstark, wallpaper, 4 k, award winning, digital art, imperial colors, fantastic view](gallery/00091ud.png)

[concept art of a far-future city, key visual, summer day, highly detailed, digital painting, artstation, concept art, sharp focus, in harmony with nature, streamlined, by makoto shinkai and akihiko yoshida and hidari and wlop](gallery/00124ud.png)

["female cyborg assimilated by alien fungus", intricate Three-point lighting portrait, by Ching Yeh and Greg Rutkowski, detailed cyberpunk in the style of GitS 1995](gallery/00155ud.png)
