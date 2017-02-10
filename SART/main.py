from pySART import pysart
from skimage.transform import radon, iradon
import phantom
import numpy as np

def generate_phantom():
    theta = range(0,180)
    data = np.rot90(radon(phantom.phantom(n=100),theta=theta))
    data1 = np.zeros((data.shape[0],1,data.shape[1]))
    data1[:,0,:] = data
    return theta, data1

# tes = (*generate_phantom())
# print tes
# pysart = pysart(tes)
# pysart.sart()

pysart = pysart(*generate_phantom())
print pysart
pysart.sart()
