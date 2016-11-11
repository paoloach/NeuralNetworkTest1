import json
import struct

import numpy
from scipy import misc, random

typesDict = dict()
typeCount = 0
HALFSIZE = 9
INVALID_RANGE = 9
sample_count = 0


def _read32(bytestream):
    dt = numpy.dtype(numpy.uint32).newbyteorder('>')
    return numpy.frombuffer(bytestream.read(4), dtype=dt)[0]


def _write32(bytestream, value):
    us32bit = struct.pack("I", value)
    bytestream.write(us32bit)


def extract_data(image, x, y):
    shape = image.shape
    left = x - HALFSIZE
    if left < 0:
        left = 0
    right = left + 2*HALFSIZE
    if right > shape[1]:
        right = shape[1]
    top = y - HALFSIZE
    if top < 0:
        top = 0
    bottom = top + 2*HALFSIZE
    if bottom > shape[0]:
        bottom = shape[0]

    sub = image[top:bottom, left:right]
    sub *= 255
    return sub.flatten()


def save_data_single(image, x, y, type_id=0):
    global sample_count
    img_data = extract_data(image, x, y)
    if len(img_data) < 4*HALFSIZE*3:
        print("ERRORE")
    writer.write(img_data)
    label_writer.write(numpy.array(type_id, dtype=numpy.uint8))

    sample_count += 1


def save_data(image, x, y, type_id=0, samples=11):
    global sample_count
    img_data = extract_data(image, x, y)
    writer.write(img_data)
    label_writer.write(numpy.array(type_id, dtype=numpy.uint8))
    sample_count += 1
    for index in range(1, samples):
        img_data = extract_data(image, x, y)
        for index_point in range(0, 20):
            point = random.randint(0, img_data.size)
            min_val = max(img_data[point] - 20, 0)
            max_val = min(img_data[point] + 20, 255)
            img_data[point] = random.randint(min_val, max_val)
        writer.write(img_data)
        label_writer.write(numpy.array(type_id, dtype=numpy.uint8))
        sample_count += 1


def manage_image(single_type, image, type_id, sammpes=11):
    image_count = 1
    for point in single_type:
        image_count += 1
        x = int(single_type[point]['x'])
        y = int(single_type[point]['y'])
        save_data(image, x, y, type_id, sammpes)


def add_invalid_points(image, points, left, top, right, bottom):
    global sample_count
    print("Generate wrong data")
    for y in range(top, bottom, 2):
        for x in range(left, right, 2):
            if not (x, y) in points:
                save_data_single(image, x, y, 0)


def add_single_type_to_dic(single_type):
    if single_type not in typesDict:
        typesDict[single_type] = len(typesDict) + 1
    return typesDict[single_type]


def calc_valid_images(data_file):
    count = 0
    for list_types in data_file['type']:
        for single_type in list_types:
            count += len(list_types[single_type])
    return count


def manage_valid_images(data_file, sample_for_point):
    image_file_name = data_file['file']
    types = data_file['type']
    print("imageFile: %s" % image_file_name)
    image = misc.imread(image_file_name)

    for listTypes in types:
        for singleType in listTypes:
            type_id = add_single_type_to_dic(singleType)
            manage_image(listTypes[singleType], image, type_id, sample_for_point)


def manage_invalid_images(data_file):
    image_file_name = data_file['file']
    left = data_file['left']
    top = data_file['top']
    right = data_file['right']
    bottom = data_file['bottom']
    types = data_file['type']
    print("imageFile: %s" % image_file_name)
    image = misc.imread(image_file_name)

    points = []
    for listTypes in types:
        for singleType in listTypes:
            list_point = listTypes[singleType]
            for point in list_point:
                x = int(list_point[point]['x'])
                y = int(list_point[point]['y'])
                points.append((x, y))
    add_invalid_points(image, points, left, top, right, bottom)


print("start")

with open("data.txt") as f:
    data = json.load(f)
with open("samples.img", "wb") as writer:
    with open("samples.label","wb") as label_writer:
        sample_for_point = 11
        for dataFile in data:
            manage_valid_images(dataFile, sample_for_point)
        for dataFile in data:
            manage_invalid_images(dataFile)
print('written ' + str(sample_count) + " samples")
