import tensorflow as tf

from scipy import misc
import glob
import json
 
typesDict = dict()
typeCount=0
 
def extractData(image,points):
    print (points)
    for point in points:
        x = points[point]['x']
        y = points[point]['y']
        
 
def manageImage(singleType, image):
    print(singleType)
    for point in singleType:
        
        
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
        
