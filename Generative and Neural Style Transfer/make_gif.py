"""Content + Style = result triptych GIF from a run's frames.

Opens on the final result (held), then plays from the photo to the result, so a
paused GIF still shows the finished image.
Usage: python make_gif.py <run_dir> <content> <style> "<style label>" <out.gif> [steps_per_frame]
"""
import glob
import sys

from PIL import Image, ImageDraw, ImageFont, ImageOps

run, content, style, label, out = sys.argv[1:6]
per = int(sys.argv[6]) if len(sys.argv) > 6 else 10
stride = int(sys.argv[7]) if len(sys.argv) > 7 else 1
H = 300
F = 'C:/Windows/Fonts/'
font = ImageFont.truetype(F + 'arial.ttf', 17)
sym = ImageFont.truetype(F + 'arialbd.ttf', 34)


def fit(im):
    im = ImageOps.exif_transpose(im).convert('RGB')
    return im.resize((round(im.width * H / im.height), H), Image.LANCZOS)


c, s = fit(Image.open(content)), fit(Image.open(style))
frames = sorted(glob.glob(f'{run}/frames/*.png'))
res_w = c.width
gap, sw, cap = 14, 46, 34
W = c.width + s.width + res_w + 2 * sw + 2 * gap


def panel(step_img, caption):
    im = Image.new('RGB', (W, H + cap), 'white')
    d = ImageDraw.Draw(im)
    x = gap
    for pic, text in [(c, 'Content'), (s, f'Style ({label})')]:
        im.paste(pic, (x, 0))
        d.text((x + pic.width / 2, H + 8), text, font=font, fill=(30, 30, 30), anchor='mt')
        x += pic.width
        d.text((x + sw / 2, H / 2), '+' if text == 'Content' else '=', font=sym, fill=(60, 64, 80), anchor='mm')
        x += sw
    r = step_img.convert('RGB').resize((res_w, H), Image.LANCZOS)
    im.paste(r, (x, 0))
    d.text((x + res_w / 2, H + 8), caption, font=font, fill=(30, 30, 30), anchor='mt')
    return im


seq = []
last = len(frames) - 1
for i, f in enumerate(frames):
    if i not in (0, last) and i % stride:
        continue
    if i == 0:
        cap_txt = 'Step 0 (the photo)'
    elif i == last:
        cap_txt = 'Result'
    else:
        cap_txt = f'Step {i * per}'
    seq.append(panel(Image.open(f), cap_txt))

final = seq[-1]
order = [final] + seq[:-1]
dur = [1800] + [500] + [90 * stride] * (len(seq) - 2)
# One palette for every frame: the content and style panels then encode
# identically in each frame, and the GIF only stores the changing result panel.
ref = final.quantize(256, method=Image.Quantize.MEDIANCUT)
pal = [im.quantize(palette=ref) for im in order]
pal[0].save(out, save_all=True, append_images=pal[1:], duration=dur, loop=0, optimize=True)
final.save(out.replace('.gif', '.png'))
print(out, len(order), 'frames')
