#!/usr/bin/env python3

#pip install numpy
#pip install tensorflow
#pip install keras
#pip install matplotlib

import os, sys, getopt
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense, Reshape, BatchNormalization, Flatten, UpSampling2D, Conv2D, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential
from keras.optimizers import Adam
from copy import deepcopy

# The size of the noise used as input to the generator
NOISE_SIZE = 100
# The images are 64 by 64 - and either one or three channels
IMAGE_W = 64
IMAGE_H = 64
#IMAGE_C = 3


# How many epochs to train the gan with.
NUMBER_OF_EPOCHS = 100
# The batch size for training.
# For each batch in an epoch a coin is tossed weighted by TOSS_CHANCE_REAL_OR_FAKE and images are then taken from
# either the training data or generated
# These images are then used to train the discriminator.
# Next the same number of images are based through the whole GAN (generator + discriminator)
# in order to train the generator.
# When there are no longer enough real images left in the training data the epoch ends.
BATCH_SIZE = 128
TOSS_CHANCE_REAL_OR_FAKE = 0.8
# When training the generator occasional incorrect labelling can help.
# This determines the chance of a correct label.
TOSS_CHANCE_REAL_LABEL = 0.9
# After this many batchs/epochs a checkpoint is performed
# - some generated images are saved in the output_images dir and the generator model is saved in the output_models dir.
CHECKPOINT = 40

# Loads and scales data
def load_npy(npy_path,amount_of_data=0.25):
    X_train = np.load(npy_path)
    X_train = X_train[:int(amount_of_data*float(len(X_train)))]
    X_train = (np.float32(X_train) - 127.5) / 127.5
    X_train = np.expand_dims(X_train, axis=3)
    return X_train


# A weighted coin toss
def flip_coin(chance=0.5):
    return np.random.binomial(1, chance)


# Generates noise to use as input into the generator/gan
def generate_noise(instances):
    return np.random.normal(0, 1, (instances, NOISE_SIZE))


# Performs a checkpoint
def checkpoint(output_stem, is_colour, generator, label):
    output_image_folder = output_stem + "_generated_images"
    if not os.path.exists(output_image_folder):
        os.mkdir(output_image_folder)
    output_image_file = os.path.join(output_image_folder, str(label) + ".png")
    checkpoint_generate_images(output_image_file, generator, is_colour)

    output_model_folder = output_stem + "_models"
    if not os.path.exists(output_model_folder):
        os.mkdir(output_model_folder)
    output_model_file = os.path.join(output_model_folder, str(label))
    checkpoint_save_model(output_model_file, generator)
    return


# Generates and saves some images to use in a checkpoint
def checkpoint_generate_images(plot_filename, generator, is_colour):
    noise = generate_noise(25)
    images = generator.predict(noise)
    plt.figure(figsize=(20, 20))
    for i in range(images.shape[0]):
        plt.subplot(5, 5, i + 1)
        if is_colour == False:
            image = images[i, :, :]
            image = np.reshape(image, [IMAGE_H, IMAGE_W])
            image = (255 * (image - np.min(image)) / np.ptp(image)).astype(int)
            plt.imshow(image, cmap='gray')
        else:
            image = images[i, :, :, :]
            image = np.reshape(image, [IMAGE_H, IMAGE_W, 3])
            image = (255 * (image - np.min(image)) / np.ptp(image)).astype(int)
            plt.imshow(image)
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(plot_filename)
    plt.close("all")


# Saves the generator model at a checkpoint.
def checkpoint_save_model(model_filename, generator):
    generator.save(filepath=model_filename, overwrite=True, include_optimizer=True)


# Builds the generator
def build_generator(is_colour):
    power = 8

    size = pow(2,power)
    optimizer = Adam(lr=1e-4, beta_1=0.2)
    model = Sequential()

    # from NOISE_SIZE to 16384
    model.add(Dense(size * 8 * 8, input_dim=NOISE_SIZE))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization())

    model.add(Reshape((8, 8, size)))
    model.add(UpSampling2D())
    # Now 16x16x256

    model.add(Conv2D(int(size/2), (5, 5), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization())
    model.add(UpSampling2D())
    # Now 32x32x128

    model.add(Conv2D(int(size/4), (5, 5), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization())
    model.add(UpSampling2D())
    # Now 64x64x64

    model.add(Conv2D(3 if is_colour else 1, (5, 5), padding='same', activation='tanh'))
    # Now 64x64x3 (or 1 depending on channels)
    model.compile(loss='binary_crossentropy', optimizer=optimizer)

    return model


# Builds the discriminator
def build_discriminator(is_colour):
    shape = (IMAGE_W, IMAGE_H, 3 if is_colour else 1)

    optimizer = Adam(lr=1e-4, beta_1=0.2)
    model = Sequential()
    # 64 filters, 5x5 kernel, 2x2 stride.  Output from this section will be 32x32x64
    model.add(Conv2D(64, (5, 5), strides=(2, 2), input_shape=shape, padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3))

    model.add(BatchNormalization())
    # 128 filters, 5x5 kernel, 2x2 stride. Output from this section will be 16x16x128
    model.add(Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())

    # 16x16x128 flattened to 32768 then 1 output
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=optimizer)

    return model


# Combine the generator and discriminator into a GAN
def build_GAN(generator, discriminator):
    optimizer = Adam(lr=0.0002, decay=8e-9)
    # Disable training of discriminator when training the GAN model
    discriminator.trainable = False
    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    model.compile(loss='binary_crossentropy', optimizer=optimizer)
    return model


# Train the GAN
def train(file_path, output_stem, is_colour, fraction_of_input=0.25):
    X_train = load_npy(file_path, fraction_of_input)
    generator = build_generator(is_colour)
    print(generator.summary())
    discriminator = build_discriminator(is_colour)
    print(discriminator.summary())
    gan = build_GAN(generator, discriminator)
    print(gan.summary())

    for epoch in range(NUMBER_OF_EPOCHS):
        batch = 0

        X_train_temp = deepcopy(X_train)
        while len(X_train_temp) > BATCH_SIZE:
            # Firstly train the discriminator
            if flip_coin(TOSS_CHANCE_REAL_OR_FAKE):
                print("Training discriminator on real images.")
                starting_idx = np.random.randint(0, len(X_train_temp) - BATCH_SIZE)
                real_images_raw = X_train_temp[starting_idx: (starting_idx + BATCH_SIZE)]
                x_batch = real_images_raw.reshape(BATCH_SIZE, IMAGE_W, IMAGE_H, 3 if is_colour else 1)
                # These are real, so label accordingly
                y_batch = np.ones([BATCH_SIZE, 1])

                X_train_temp = np.delete(X_train_temp, range(starting_idx, (starting_idx + BATCH_SIZE)), 0)
                print("Real images remaining in this epoch: " + str(len(X_train_temp)))
            else:
                print("Training discriminator on generated images")
                noise = generate_noise(BATCH_SIZE)
                x_batch = generator.predict(noise)
                # These are fake so label them accordingly.
                y_batch = np.zeros([BATCH_SIZE, 1])

            discriminator_loss = discriminator.train_on_batch(x_batch, y_batch)

            # Secondly train the generator
            # Occasionally mis-label when training the generator
            if flip_coin(TOSS_CHANCE_REAL_LABEL):
                y_generated_labels = np.ones([BATCH_SIZE, 1])
            else:
                y_generated_labels = np.zeros([BATCH_SIZE, 1])
            # Generate noise
            noise = generate_noise(BATCH_SIZE)

            # Now train the generator
            generator_loss = gan.train_on_batch(noise, y_generated_labels)

            print("Epoch: " + str(epoch) + " Batch: " + str(batch) +
                  " Discriminator Loss: " + str(discriminator_loss) +
                  " Generator Loss: " + str(generator_loss))

            if batch % CHECKPOINT == 0:
                label = str(epoch) + "_" + str(batch)
                checkpoint(output_stem, is_colour, generator, label)

            batch += 1
        print("Epoch: " + str(epoch) + " completed.")

        if epoch % CHECKPOINT == 0:
            checkpoint(output_stem, is_colour, generator, epoch)


def usage():
    print("Trains a DCGAN on a specified .npy file of data. This can take a long time!")
    print( "usage: gan.py [-h] -i:<input.npy>] -o:<output_filestem> [-c] [-f:fraction_of_input (default = 0.25)]")
    print("example: ./gan.py -i./data/images_rgb.npy  -o./data/out -c -f:0.5")
    print("This example will read image training data from ./data/images_rgb.npy and create two folders:")
    print("./data/out_generated_images and ./data/out_models")
    print("Images will be assumed to be colour.")
    print("0.5 of the input data will be utilised.")
    sys.exit(0)


def main():
    # Parse command line
    if len(sys.argv) <= 1:
        usage()

    args = sys.argv[1:]
    short_opts = "hi:o:cf:"
    long_opts = ["help", "input=", "output=", "colour", "fraction_of_input="]
    try:
        arguments, values = getopt.getopt(args, short_opts, long_opts)
    except getopt.error as err:
        print(str(err))
        sys.exit(2)

    input_file = None
    output = None
    is_colour = False
    fraction_of_input = 0.25

    for current_arg, current_val in arguments:
        if current_arg in ("-h", "--help"):
            usage()
        elif current_arg in ("-i", "--input"):
            input_file = current_val
        elif current_arg in ("-o", "--output"):
            output = current_val
        elif current_arg in ("-c", "--colour"):
            is_colour = True
        elif current_arg in ("-f", "--fraction_of_inout"):
            fraction_of_input = float(current_val)


    if not input or not output:
        usage()

    print("Source file: " + input_file)
    print("Output stem: " + output)
    print("IsColour: " + str(is_colour))

    train(input_file, output, is_colour, fraction_of_input)


if __name__ == "__main__":

    main()