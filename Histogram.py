from PIL import Image
import numpy as np  
from matplotlib import pyplot as plt

img = Image.open("/path/to/file/image.tif")

r, g, b = img.split()

hist_r = np.array(r.histogram())
hist_r[:5] = hist_r[10] # :5 for the thredshold can be changed according to the need 
hist_g = np.array(g.histogram())
hist_g[:5] = hist_g[10]
hist_b = np.array(b.histogram())
hist_b[:5] = hist_b[10]
plt.plot((hist_r+hist_g),color = 'y')
plt.plot(hist_r,color = 'r')
plt.plot(hist_g,color = 'g')
plt.plot(hist_b,color = 'b')
plt.xlim([0,256])
plt.show()

Red = np.array(r)>100
plt.imshow(Red)
