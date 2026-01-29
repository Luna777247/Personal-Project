import os
import json
random = __import__('random')
from PIL import Image, ImageDraw, ImageFont


def load_split(split_name='val', processed_dir='data/processed'):
    split_file = os.path.join(processed_dir, f'{split_name}_captions.json')
    if not os.path.exists(split_file):
        raise FileNotFoundError(f"Split file not found: {split_file}")
    with open(split_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def make_montage(items, out_path, cols=3, thumb_size=(224, 224), font_path=None):
    rows = (len(items) + cols - 1) // cols
    pad = 8
    caption_h = 40
    w = cols * (thumb_size[0] + pad) + pad
    h = rows * (thumb_size[1] + caption_h + pad) + pad

    montage = Image.new('RGB', (w, h), color=(255, 255, 255))
    draw = ImageDraw.Draw(montage)

    try:
        font = ImageFont.truetype(font_path or 'arial.ttf', 14)
    except Exception:
        font = ImageFont.load_default()

    for idx, (img_path, caption) in enumerate(items):
        col = idx % cols
        row = idx // cols
        x = pad + col * (thumb_size[0] + pad)
        y = pad + row * (thumb_size[1] + caption_h + pad)

        try:
            img = Image.open(img_path).convert('RGB')
            img = img.resize(thumb_size)
        except Exception:
            img = Image.new('RGB', thumb_size, color=(200, 200, 200))

        montage.paste(img, (x, y))

        # draw caption (wrap)
        cap_lines = []
        words = caption.split()
        line = ''
        for w in words:
            if len(line + ' ' + w) > 30:
                cap_lines.append(line.strip())
                line = w
            else:
                line = (line + ' ' + w).strip()
        if line:
            cap_lines.append(line)

        for i, cl in enumerate(cap_lines[:3]):
            draw.text((x + 4, y + thumb_size[1] + 2 + i * 14), cl, fill=(0, 0, 0), font=font)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    montage.save(out_path)
    print(f"Saved preview montage to {out_path}")


def preview(split='val', n=6):
    data = load_split(split)
    image_dir = os.path.join('data/processed', 'images')

    items = []
    for img_id, caps in data.items():
        img_path = os.path.join(image_dir, f"{img_id}.jpg")
        caption = caps[0] if isinstance(caps, list) and caps else ''
        if os.path.exists(img_path):
            items.append((img_path, caption))

    if not items:
        print('No items found in processed images for preview')
        return

    sample = random.sample(items, min(n, len(items)))
    out = os.path.join('results', f'preview_{split}.png')
    make_montage(sample, out_path=out)


if __name__ == '__main__':
    preview(split='val', n=6)
