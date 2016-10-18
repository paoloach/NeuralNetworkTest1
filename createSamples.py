import tensorflow as tf

from scipy import misc
import glob
import json

 
typesDict = dict()
typeCount=0
HALFSIZE=7
sampleCount=0
 
def extractData(image, x, y):
  shape = image.shape
  left = x - HALFSIZE
  if left < 0:
    left=0
  right = x + HALFSIZE
  if right > shape[1]:
    right = shape[1]
  top = y -HALFSIZE
  if top < 0:
    top=0
  bottom = y + HALFSIZE
  if bottom > shape[0]:
    bottom = shape[0]
    
  return  image[top:bottom,left:right].flatten().tostring();
  
   
def manageImage(singleType, image, typeId,writer):
    global sampleCount
    for point in singleType:
        x = int(singleType[point]['x'])
        y = int(singleType[point]['y'])
        imgData = extractData(image, x, y)
        data = bytes([typeId]) + imgData
        writer.write(data)
        sampleCount+=1
        
def addSingleTypeToDic(singleType):
    if not singleType in typesDict :
        typesDict[singleType] = len(typesDict)+1
    return typesDict[singleType];

def manageFile(dataFile, writer):
    imageFileName = dataFile['file']
    types = dataFile['type']
    image = misc.imread(imageFileName)
    for listTypes in types:
        for singleType in listTypes:
            typeId = addSingleTypeToDic(singleType)
            manageImage(listTypes[singleType], image, typeId,writer)

print("start")
with open("data.txt") as f:
    data = json.load(f);
    writer = tf.python_io.TFRecordWriter('samples.img')
for dataFile in data: 
    manageFile(dataFile,writer)
print ('written ' + str(sampleCount) + " samples")
