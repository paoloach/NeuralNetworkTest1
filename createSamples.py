import json

import numpy
import struct
from scipy import misc

typesDict = dict()
typeCount = 0
HALFSIZE = 7
sampleCount = 0


def _read32(bytestream):
    dt = numpy.dtype(numpy.uint32).newbyteorder('>')
    return numpy.frombuffer(bytestream.read(4), dtype=dt)[0]


def _write32(bytestream, value):
    us32bit = struct.pack("I", value)
    bytestream.write(us32bit)


def extractData(image, x, y):
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
    return image[top:bottom, left:right].flatten().tostring()


def manageImage(singleType, image, typeId, writer):
    global sampleCount
    for point in singleType:
        x = int(singleType[point]['x'])
        y = int(singleType[point]['y'])
        imgData = extractData(image, x, y)
        data = bytes([typeId]) + imgData
        writer.write(data)
        sampleCount += 1


def addSingleTypeToDic(singleType):
    if not singleType in typesDict:
        typesDict[singleType] = len(typesDict) + 1
    return typesDict[singleType];


def manageFile(dataFile, writer):
    imageFileName = dataFile['file']
    types = dataFile['type']
    image = misc.imread(imageFileName)
    for listTypes in types:
        for singleType in listTypes:
            typeId = addSingleTypeToDic(singleType)
            manageImage(listTypes[singleType], image, typeId, writer)


print("start")

with open("data.txt") as f:
    data = json.load(f);
with open("samples.img", "bw") as writer:
    _write32(writer, HALFSIZE * 2);
    _write32(writer, HALFSIZE * 2);
    for dataFile in data:
        manageFile(dataFile, writer)
print('written ' + str(sampleCount) + " samples")
