#!/usr/bin/env python3

# pip install numpy
# pip install Pillow

import numpy as np
import sys, os, getopt, glob
from PIL import Image

def main():
    # Parse command line
    if len(sys.argv) <= 1:
        usage()

    args = sys.argv[1:]
    short_opts = "hi:o:rw:rh:"
    long_opts = ["help", "input=", "output=", "resize_width=", "resize_height="]
    try:
        arguments, values = getopt.getopt(args, short_opts, long_opts)
    except getopt.error as err:
        print(str(err))
        sys.exit(2)


    width = 64
    height = 64
    directory = None
    output = None


    for current_arg, current_val in arguments:
        if current_arg in ("-h", "--help"):
            usage()
        elif current_arg in ("-i", "--input"):
            directory = current_val
        elif current_arg in ("-o", "--output"):
            output = current_val
        elif current_arg in ("-rw", "--resize_width"):
            width = int(current_val)
        elif current_arg in ("-rh", "--resize_height"):
            height = int(current_val)

    if not output or not directory:
        usage()

    print("Proessing images in " + directory)
    image_data_gray = []
    image_data_colour = []
    supported_extensions = [".WEBP", ".PNG", ".JPG", ".JPEG"]
    index = 0
    for file in glob.iglob(os.path.join(directory,"**"), recursive=True):
        _, ext = os.path.splitext(file)
        if ext.upper() in supported_extensions:
            gray, colour = get_image_data(file, resize_width=width, resize_height=height)
            image_data_gray.append(gray)
            image_data_colour.append(colour)

            if index % 1000 == 0:
                print("Processed " + str(index) + " files.")
            index += 1
    print("GrayScale Shape:", np.shape(image_data_gray))
    print("Colour Shape:", np.shape(image_data_colour))
    gray_file = output + "_gray.npy"
    colour_file = output +"_colour.npy"
    np.save(gray_file, image_data_gray)
    print("Wrote File:", gray_file)
    np.save(colour_file, image_data_colour)
    print("Wrote File:", colour_file)


def usage():
    print("Prepares a folder of images for use in the DCGAN by scaling them (default 64 by 64) and saving as npy files.")
    print("usage: prepare_images.py [-h] -i:<directory of images>] -o:<output_filestem> [-rw:rescale_width] [-rh:rescale_height]")
    print("example: ./prepare_images.py -d./data/image_folder -o./data/images")
    print("This example will read image files from ./data/image_folder and create two files:")
    print("./data/images_gray.npy and ./data/images_colour.npy")
    sys.exit(0)


def get_image_data(file, resize_width, resize_height):
    ig = Image.open(file).convert("L")
    ig = ig.resize((resize_width, resize_height))
    ic = Image.open(file).convert("RGB")
    ic = ic.resize((resize_width, resize_height))
    return np.asarray(ig), np.asarray(ic)


if __name__ == "__main__":
    main()
