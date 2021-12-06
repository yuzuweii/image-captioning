import argparse
import os
from PIL import Image

def resize_single_image(image, size):
    return image.resize(size, Image.ANTIALIAS)

def resize_images(image_dir, output_dir, size):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    images = os.listdir(image_dir)
    for image in images:
        with open(os.path.join(image_dir, image), 'r+b') as f:
            with Image.open(f) as img:
                img = resize_single_image(img, size)
                save_path = os.path.join(output_dir, image)
                img.save(save_path, img.format)

def main(args):
    resize_images(args.image_dir, args.output_dir, [args.image_size, args.image_size])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, default='./data2/train2014/',
                        help='path to training images')
    parser.add_argument('--output_dir', type=str, default='./data2/resized2014/',
                        help='path to save resized images')
    parser.add_argument('--image_size', type=int, default=256,
                        help='size for image after processing')
    args = parser.parse_args()
    main(args)