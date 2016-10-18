import tensorflow as tf

from scipy import misc
import glob
import json
 

print("start")
for image_path in glob.glob("*.jpg"):
    dataFile = image_path[:-4]+".txt";
    print (dataFile)
    with open(dataFile) as f:
        data = json.load(f);
        print(len(data["squareYellow"]))
    image = misc.imread(image_path)
    print (image.shape)
    print (image.dtype)
    print data["squareYellow"].count()(image[0,0])
