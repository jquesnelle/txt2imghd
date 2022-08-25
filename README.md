# txt2imghd

txt2imghd is a port of the GOBIG mode from [progrockdiffusion](https://github.com/lowfuel/progrockdiffusion) applied to [Stable Diffusion](https://raw.githubusercontent.com/CompVis/stable-diffusion), with [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) as the upscaler. It creates detailed, higher-resolution images by first generating an image from a prompt, upscaling it, and then running img2img on smaller pieces of the upscaled image, and blending the result back into the original image.

txt2imghd with default settings has the same VRAM requirements as regular Stable Diffusion, although generation of the detailed images will take longer.

## Installation

1. Have a working repository of [Stable Diffusion](https://raw.githubusercontent.com/CompVis/stable-diffusion)
2. Copy `txt2imghd.py` into `scripts/`
3. Download the appropriate [release of Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN/releases) and unzip it into the root of your Stable Diffusion repository

## Running

txt2imghd has most of the same parameters as txt2img. `ddim_steps` has been renamed to `steps`. The `strength` parameter controls how much detailing to do (between 0.0-1.0). If no `prompt` is given on the command line, the program will ask for it as input.

```sh
python scripts/txt2imghd.py
```

txt2imghd will output three images: the original Stable Diffusion image, the upscaled version (denoted by a `u` suffix), and the detailed version (denoted by the `ud` suffix).

## Example images

[old harbour, tone mapped, shiny, intricate, cinematic lighting, highly detailed, digital painting, artstation, concept art, smooth, sharp focus, illustration, art by terry moore and greg rutkowski and alphonse mucha](gallery/00005ud.png)

[55mm closeup hand photo of a breathtaking majestic beautiful armored redhead woman mage holding a tiny ball of fire in her hand on a snowy night in the village. zoom on the hand. focus on hand. dof. bokeh. art by greg rutkowski and luis royo. ultra reallistic. extremely detailed. nikon d850. cinematic postprocessing.](gallery/00030ud.png)

[a humanoid armored futuristic cybernetic samurai with glowing neon decals, award winning photograph, close up, focused trending on artstation, octane render, portrait, hyperrealistic, ultra detailed, photograph](gallery/00068ud.png)

[(painting of girl from behind looking a fleet of imperial ships in the sky, in a meadow of flowers. ) by donato giancola and Eddie Mendoza,  elegant, dynamic lighting, beautiful, poster, trending on artstation, poster, anato finnstark, wallpaper, 4 k, award winning, digital art, imperial colors, fantastic view](gallery/00091ud.png)

[concept art of a far-future city, key visual, summer day, highly detailed, digital painting, artstation, concept art, sharp focus, in harmony with nature, streamlined, by makoto shinkai and akihiko yoshida and hidari and wlop](gallery/00124ud.png)

["female cyborg assimilated by alien fungus", intricate Three-point lighting portrait, by Ching Yeh and Greg Rutkowski, detailed cyberpunk in the style of GitS 1995](gallery/00155ud.png)
