#!/usr/bin/env python
# coding: utf-8

# In[20]:


import numpy as np
from mayavi import mlab
from os import listdir
from os.path import isfile, join


# In[16]:
# In[17]:





# In[18]:



onlyfiles = [join('./extracted/', f) for f in listdir('./extracted/') if isfile(join('./extracted/', f))]
for i in onlyfiles:

    c = np.load(i)
    print("file_name: ", i)
    print("number of points: ", c.shape[0])
    x = c[:, 0]  # x position of point
    y = c[:, 1]  # y position of point
    z = c[:, 2]  # z position of point
    r = c[:, 3]  # reflectance value of point
    d = np.sqrt(x ** 2 + y ** 2)  # Map Distance from sensor
     
    vals='height'
    if vals == "height":
        col = z
    else:
        col = d
     
    fig = mlab.figure(bgcolor=(0, 0, 0), size=(640, 500))
    mlab.points3d(x, y, z,
                         col,          # Values used for Color
                         mode="point",
                         colormap='spectral', # 'bone', 'copper', 'gnuplot'
                         # color=(0, 1, 0),   # Used a fixed (r,g,b) instead
                         figure=fig,
                         )
    mlab.show()


