"""Neural style transfer, revisited (HW4).

What changed from dpl4_*.py, and why:
- Start from the content photo instead of white noise, so the result keeps the
  scene and does not carry leftover noise.
- Keep the photo's aspect ratio instead of squashing it to 512x512.
- Content from one deep layer (conv4_2), style from the first conv of each
  block (conv1_1 ... conv5_1), as in Gatys et al. The old setup used ten style
  layers and two shallow content layers, which let style wipe out the scene.
- Gram matrices are normalized per layer, so every style layer counts equally.
- A small total-variation term smooths pixel-level speckle.
- Coarse to fine: optimize at a small size, then upsample and refine, which
  gives larger brush strokes and a sharper final image.
- Frames are saved straight from the tensor (no matplotlib axes) for the GIF.

Usage (from this folder; the settings used for the README):
  python nst_improved.py --content data/images/content/content_path.jpg --style data/images/style/Zodiac.jpg --out out/zodiac --style-weight 3e6
  python nst_improved.py --content data/images/content/sea-steps.jpg --style "data/images/style/The Starry Night.jpg" --out out/starry --style-weight 1e7
"""
import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps
import torchvision.transforms.functional as TF
from torchvision.models import vgg19, VGG19_Weights

p = argparse.ArgumentParser()
p.add_argument('--content', required=True)
p.add_argument('--style', required=True)
p.add_argument('--out', required=True)
p.add_argument('--sizes', default='384,768', help='long side for each scale, coarse to fine')
p.add_argument('--steps', default='300,200', help='L-BFGS evaluations for each scale')
p.add_argument('--style-weight', type=float, default=1e6)
p.add_argument('--content-weight', type=float, default=1.0)
p.add_argument('--tv-weight', type=float, default=1e-5)
p.add_argument('--style-scale', type=float, default=1.0, help='style image size relative to the content long side')
p.add_argument('--frame-every', type=int, default=10)
args = p.parse_args()

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs(os.path.join(args.out, 'frames'), exist_ok=True)

MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)

# VGG19 conv indices (1-based, counting conv layers only)
CONTENT_LAYERS = {10}            # conv4_2
STYLE_LAYERS = {1, 3, 5, 9, 13}  # conv1_1, conv2_1, conv3_1, conv4_1, conv5_1
LAST = max(CONTENT_LAYERS | STYLE_LAYERS)

vgg = vgg19(weights=VGG19_Weights.DEFAULT).features.to(dev).eval()
for q in vgg.parameters():
    q.requires_grad_(False)


def features(x):
    """Content and style activations for an image in [0, 1]."""
    x = (x - MEAN) / STD
    content, style, n = {}, {}, 0
    for layer in vgg:
        x = layer(x) if not isinstance(layer, nn.ReLU) else F.relu(x)
        if isinstance(layer, nn.Conv2d):
            n += 1
            # take activations after the ReLU that follows, so note the index here
            pending = n
        elif isinstance(layer, nn.ReLU):
            if pending in CONTENT_LAYERS:
                content[pending] = x
            if pending in STYLE_LAYERS:
                style[pending] = x
            if pending == LAST:
                break
    return content, style


def gram(f):
    b, c, h, w = f.shape
    f = f.view(b, c, h * w)
    return f @ f.transpose(1, 2) / (c * h * w)


def load(path, long_side):
    im = ImageOps.exif_transpose(Image.open(path)).convert('RGB')
    s = long_side / max(im.size)
    im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
    return TF.to_tensor(im).unsqueeze(0).to(dev)


def save(t, path):
    TF.to_pil_image(t.detach().clamp(0, 1).squeeze(0).cpu()).save(path)


sizes = [int(v) for v in args.sizes.split(',')]
steps = [int(v) for v in args.steps.split(',')]
frame_no = 0
img = None

for scale, (size, n_steps) in enumerate(zip(sizes, steps)):
    content_img = load(args.content, size)
    style_img = load(args.style, int(size * args.style_scale))
    with torch.no_grad():
        c_target, _ = features(content_img)
        _, s_feats = features(style_img)
        s_target = {k: gram(v) for k, v in s_feats.items()}

    if img is None:
        img = content_img.clone()
        save(img, os.path.join(args.out, 'frames', f'{frame_no:04d}.png'))
        frame_no += 1
    else:
        img = F.interpolate(img.detach(), size=content_img.shape[-2:], mode='bicubic', align_corners=False)
    img = img.clamp(0, 1).requires_grad_(True)
    opt = torch.optim.LBFGS([img], max_iter=n_steps, history_size=50, line_search_fn='strong_wolfe')
    it = [0]

    def closure():
        opt.zero_grad()
        c, s = features(img)
        lc = sum(F.mse_loss(c[k], c_target[k]) for k in CONTENT_LAYERS)
        ls = sum(F.mse_loss(gram(s[k]), s_target[k]) for k in STYLE_LAYERS) / len(STYLE_LAYERS)
        tv = (img[..., 1:, :] - img[..., :-1, :]).abs().mean() + (img[..., :, 1:] - img[..., :, :-1]).abs().mean()
        loss = args.content_weight * lc + args.style_weight * ls + args.tv_weight * tv * img.numel()
        loss.backward()
        it[0] += 1
        global frame_no
        if it[0] % args.frame_every == 0 and scale == 0:
            save(img, os.path.join(args.out, 'frames', f'{frame_no:04d}.png'))
            frame_no += 1
        if it[0] % 50 == 0:
            print(f'scale {size}: step {it[0]:4d}  content {lc.item():.3f}  style {ls.item() * args.style_weight:.3f}  tv {tv.item():.4f}', flush=True)
        return loss

    opt.step(closure)
    with torch.no_grad():
        img.clamp_(0, 1)
    save(img, os.path.join(args.out, f'result_{size}.png'))

save(img, os.path.join(args.out, 'result.png'))
save(img, os.path.join(args.out, 'frames', f'{frame_no:04d}.png'))
print('done', os.path.join(args.out, 'result.png'))
