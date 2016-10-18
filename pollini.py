import tensorflow as tf

from scipy import misc
import glob
import json
 
typesDict = dict()
typeCount=0
HALFSIZE=7
 
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
  print shape
  result = ''
  for xPos in range(left,right+1):
    for yPos in range(top, bottom+1):
      result += unichr(image[y][x][0]) + unichr(image[y][x][1]) +unichr(image[y][x][2])
  return result
 
def manageImage(singleType, image):
    print(singleType)
    for point in singleType:
        x = int(singleType[point]['x'])
        y = int(singleType[point]['y'])
        extractData(image, x, y);
        
def addSingleTypeToDic(singleType):
    if not singleType in typesDict :
        typesDict[singleType] = len(typesDict)+1

def manageFile(dataFile):
    imageFileName = dataFile['file']
    types = dataFile['type']
    image = misc.imread(imageFileName)
    for singleType in types:
        addSingleTypeToDic(singleType)
        manageImage(types[singleType], image)
    print(typesDict)

print("start")
with open("data.txt") as f:
    data = json.load(f);
for dataFile in data: 
    manageFile(dataFile)
        
