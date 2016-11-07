import json

import numpy
import struct
from scipy import misc, random
import skimage.color
from scipy import ndimage
import matplotlib.pyplot as plt

typesDict = dict()
typeCount = 0
HALFSIZE = 9
INVALID_RANGE = 4
sampleCount = 0


def _read32(bytestream):
    dt = numpy.dtype(numpy.uint32).newbyteorder('>')
    return numpy.frombuffer(bytestream.read(4), dtype=dt)[0]


def _write32(bytestream, value):
    us32bit = struct.pack("I", value)
    bytestream.write(us32bit)


def extractData(image, x, y, angle):
    shape = image.shape
    left = x - HALFSIZE
    if left < 0:
        left = 0
    right = x + HALFSIZE
    if right > shape[1]:
        right = shape[1]
    top = y - HALFSIZE
    if top < 0:
        top = 0
    bottom = y + HALFSIZE
    if bottom > shape[0]:
        bottom = shape[0]

    sub = image[top:bottom, left:right]
    # plt.imshow(sub)
    if angle > 0:
        sub = ndimage.rotate(sub, angle, reshape=False)
   # sub = skimage.color.rgb2hsv(sub)
    sub = 255 * sub
    return sub.flatten()


def save_data(image, writer, x, y, angle=0, typeId=0):
    global sampleCount
    imgData = extractData(image, x, y, angle)
    writer.write(numpy.append(numpy.array(typeId, dtype=numpy.uint8), numpy.array(imgData, dtype=numpy.uint8)))
    sampleCount += 1
    for index in range (0,10):
        imgData = extractData(image, x, y, angle)
        for index in range(0,20):
            point = random.randint(0, imgData.size)
            minVal = max(imgData[point]-20, 0)
            maxVal = min(imgData[point] + 20, 255)
            imgData[point] = random.randint(minVal,maxVal)
        writer.write(numpy.append(numpy.array(typeId, dtype=numpy.uint8),numpy.array(imgData, dtype=numpy.uint8) ))
        sampleCount += 1



def manageImage(singleType, image, typeId, writer, file_name, points):
    # fig = plt.figure()
    image_count = 1
    for point in singleType:
        #        a = fig.add_subplot(3, 3, image_count)
        image_count += 1
        x = int(singleType[point]['x'])
        y = int(singleType[point]['y'])
        points.append((x, y))
        title = '{} [{},{}]'.format(file_name, x, y)
        #        a.set_title(title)
        save_data(image, writer, x, y, 0, typeId)
  #      save_data(image, writer, x, y, 90, typeId)
  #      save_data(image, writer, x, y, 180, typeId)
  #      save_data(image, writer, x, y, 270, typeId)


# plt.show()



def add_invalid_points(image, points, writer):
    for x, y in points:
        for x_dist in range(3, INVALID_RANGE, 2):
            for y_dist in range(3, INVALID_RANGE, 2):
                save_data(image, writer, x + x_dist, y + y_dist, 0, 0)
                save_data(image, writer, x + x_dist, y - y_dist, 0, 0)
                save_data(image, writer, x - x_dist, y + y_dist, 0, 0)
                save_data(image, writer, x - x_dist, y - y_dist, 0, 0)
    for i in range(1, 100):
        found = True
        pointX = random.randint(HALFSIZE, image.shape[1] - HALFSIZE)
        pointY = random.randint(HALFSIZE, image.shape[0] - HALFSIZE)
        while found:
            found=False
            for x, y in points:
                if pointX >= x - INVALID_RANGE and pointX <= x + INVALID_RANGE and pointY >= y - INVALID_RANGE and pointY <= y + INVALID_RANGE:
                    found = True
            if found:
                pointX = random.randint(HALFSIZE, image.shape[1] - HALFSIZE)
                pointY = random.randint(HALFSIZE, image.shape[0] - HALFSIZE)
        save_data(image, writer, x - x_dist, y - y_dist, 0, 0)


def addSingleTypeToDic(singleType):
    if not singleType in typesDict:
        typesDict[singleType] = len(typesDict) + 1
    return typesDict[singleType];


def manageFile(dataFile, writer):
    imageFileName = dataFile['file']
    types = dataFile['type']
    image = misc.imread(imageFileName)

    for listTypes in types:
        points = []
        for singleType in listTypes:
            typeId = addSingleTypeToDic(singleType)
            manageImage(listTypes[singleType], image, typeId, writer, imageFileName, points)
     #       add_invalid_points(image, points, writer)


print("start")

with open("data.txt") as f:
    data = json.load(f);
with open("samples.img", "wb") as writer:
    _write32(writer, HALFSIZE * 2);
    _write32(writer, HALFSIZE * 2);
    for dataFile in data:
        manageFile(dataFile, writer)
print('written ' + str(sampleCount) + " samples")
