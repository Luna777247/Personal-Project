import os
from PIL import Image
import random

def create_preview(processed_images_dir='data/processed/images', out_file='results/preview_sample.jpg', sample_count=16):
    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    all_images = [os.path.join(processed_images_dir, f) for f in os.listdir(processed_images_dir) if f.lower().endswith(('.jpg', '.png'))]
    sample = random.sample(all_images, min(sample_count, len(all_images)))

    # create a simple grid montage
    imgs = [Image.open(p).convert('RGB').resize((256,256)) for p in sample]
    cols = int(sample_count**0.5)
    rows = (len(imgs) + cols - 1) // cols

    montage = Image.new('RGB', (cols*256, rows*256), (255,255,255))

    for idx, img in enumerate(imgs):
        x = (idx % cols) * 256
        y = (idx // cols) * 256
        montage.paste(img, (x,y))

    montage.save(out_file)
    print(f"Saved preview montage to {out_file}")

if __name__ == '__main__':
    create_preview()