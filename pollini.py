import tensorflow as tf

from scipy import misc
import glob

for image_path in glob.glob("*.png"):
    image = misc.imread(image_path)
    print image.shape
    print image.dtype