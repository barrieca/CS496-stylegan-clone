import os
import pickle
import numpy as np
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
import config
import sys

# ----------------------------------------------------------------------------
# Helpers for loading and using pre-trained generators.

url_ffhq = 'cache/karras2019stylegan-ffhq-1024x1024.pkl'  # 'https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ'
#url_celebahq = 'karras2019stylegan-celebahq-1024x1024.pkl'  # 'https://drive.google.com/uc?id=1MGqJl28pN4t7SAtSrPdSRJSQJqahkzUf'

synthesis_kwargs = dict(output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True), minibatch_size=8)

_Gs_cache = dict()


def load_Gs(url):
    if url not in _Gs_cache:
        # with dnnlib.util.open_url(url, cache_dir=config.cache_dir) as f:
        with open(url, 'rb') as f:
            _G, _D, Gs = pickle.load(f)
        _Gs_cache[url] = Gs
    return _Gs_cache[url]


def style_mixing_figure(png, Gs, w, h, src_seeds, dst_dlatents, style_ranges):
    print(png)
    src_latents = np.stack(np.random.RandomState(seed).randn(Gs.input_shape[1]) for seed in src_seeds)
    src_dlatents = Gs.components.mapping.run(src_latents, None)  # [seed, layer, component]
    src_images = Gs.components.synthesis.run(src_dlatents, randomize_noise=False, **synthesis_kwargs)
    dst_images = Gs.components.synthesis.run(dst_dlatents, randomize_noise=False, **synthesis_kwargs)

    canvas = PIL.Image.new('RGB', (w * (len(src_seeds) + 1), h * (3 + 1)), 'white')
    for col, src_image in enumerate(list(src_images)):
        canvas.paste(PIL.Image.fromarray(src_image, 'RGB'), ((col + 1) * w, 0))
    for row, dst_image in enumerate(list(dst_images)):
        canvas.paste(PIL.Image.fromarray(dst_image, 'RGB'), (0, (row + 1) * h))
        row_dlatents = np.stack([dst_dlatents[row]] * len(src_seeds))
        row_dlatents[:, style_ranges[row]] = src_dlatents[:, style_ranges[row]]
        row_images = Gs.components.synthesis.run(row_dlatents, randomize_noise=False, **synthesis_kwargs)
        for col, image in enumerate(list(row_images)):
            canvas.paste(PIL.Image.fromarray(image, 'RGB'), ((col + 1) * w, (row + 1) * h))
    canvas.save(png)


def main():
    tflib.init_tf()
    latent_code_file_name = 'latent_codes/' + sys.argv[1]
    style_mixing_file_name = sys.argv[2]
    w_latent = np.load(latent_code_file_name)
    w_latent = np.tile(w_latent, (3, 1, 1))

    # draw_style_mixing_figure(os.path.join(config.result_dir, 'figure04-style-mixing.png'), load_Gs(url_ffhq), w=1024,
    #                          h=1024, src_seeds=[639, 701, 687, 615, 2268], dst_seeds=[888, 829, 1898, 1733, 1614, 845],
    #                          style_ranges=[range(0, 4)] * 3 + [range(4, 8)] * 2 + [range(8, 18)])

    style_mixing_figure(os.path.join(config.result_dir, style_mixing_file_name), load_Gs(url_ffhq), w=1024,
                           h=1024, src_seeds=[639, 701, 687, 615, 2268], dst_dlatents=w_latent,
                           style_ranges=[range(0, 4)] + [range(4, 8)] + [range(8, 18)])

    # ----------------------------------------------------------------------------


if __name__ == "__main__":
    main()
