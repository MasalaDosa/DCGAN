#!/usr/bin/env python3

import sys, getopt
import gan
from keras.models import load_model


def usage():
    print("Loads and runs a saved model to generate some images.")
    print("usage: generate_images.py [-h] -i:<saved_model>] -o:<output_filestem> [-c]")
    print("example: ./generate_images.py -i./data/example_model  -o./foo.png -c")
    print("This example will load a model from ./data/example_model and create one filer:")
    print("./foo.png")
    print("Images will be assumed to be colour.")
    sys.exit(0)


def main():
    # Parse command line
    if len(sys.argv) <= 1:
        usage()

    args = sys.argv[1:]
    short_opts = "hi:o:c"
    long_opts = ["help", "input=", "output=", "colour"]
    try:
        arguments, values = getopt.getopt(args, short_opts, long_opts)
    except getopt.error as err:
        print(str(err))
        sys.exit(2)

    input_file = None
    output = None
    is_colour = False

    for current_arg, current_val in arguments:
        if current_arg in ("-h", "--help"):
            usage()
        elif current_arg in ("-i", "--input"):
            input_file = current_val
        elif current_arg in ("-o", "--output"):
            output = current_val
        elif current_arg in ("-c", "--colour"):
            is_colour = True

    if not input or not output:
        usage()

    print("Source file: " + input_file)
    print("Output file: " + output)
    print("IsColour: " + str(is_colour))

    generator = load_model(input_file)

    gan.checkpoint_generate_images(output, generator, is_colour)


if __name__ == "__main__":
    main()