**Simple Deep Convolutional GAN**

An example generative adversarial network working with the MNIST dataset

To train this python3 program ensure you have the following libraries available in your environment:

numpy, tensorflow, keras and matplotlib

They can be installed by running the following commands in your environment:

pip install --upgrade pip

pip install numpy

pip install tensorflow

pip install keras

pip install matplotlib

pip install Pillow

First you will need a folder of training images.
I have used the outdoor churches images from the LSUN dataset and the 
cropped face images from UTKFace, but you can try any dataset you like.
In order to be processed more efficiently the images must be converted to raw data using the
prepare_images script:

./prepare_images.py -i/Path/To/ImageFolder -o /Path/To/OutputFolder/Stem

This program will result in two files:
/Path/To/OutputFolder/Stem_gray.npy and
/Path/To/OutputFolder/Stem_colour.npy

Once you have your training data available you can train the gan as follows:

./gan.py -i/Path/To/NPY/my_colour.npy -o/Path/To/Output/Stem -c -f1.0

(The -c specifies colour - omit it if using the gray npy file)
(The -f:1.0 indicates that 100% of the images will be used in training.  Make this number less to speed things up a little)
This will generate images in /Path/To/Output/Stem_generated_images and models in
/OPath/To/Output/Stem_models.

The gan will train for 100 epochs using a batch size of 128.
Every 40 batches and every 40 epochs example images will be written to the image folder and models to the model folder.
**This will take a long time!**

If you are feeling impatient then you can instead run the generate_images.py script.
This generates example images using a saved model.
I have included models for LSUN churches (church_example) and UKTFace (utkface_example).
Both examples are for colour images.
The script can be run as follows:


./generate_images.py -i./utkface_example -o./utkface_example.png -c

