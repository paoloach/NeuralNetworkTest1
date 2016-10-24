from scipy import misc
import matplotlib.pyplot as plt
from scipy import ndimage

f = misc.imread('01-08-2016.jpg') # uses the Image module (PIL)
point = f[300:330, 139:169];
f2= ndimage.rotate(point, 45, reshape=False) 

fig = plt.figure()
a=fig.add_subplot(1,2,1)
imgplot = plt.imshow(point[8:22,8:22])
a.set_title('original')
a=fig.add_subplot(1,2,2)
imgplot = plt.imshow(f2[8:22,8:22]  )
a.set_title('45 ° rotation')

plt.show()
