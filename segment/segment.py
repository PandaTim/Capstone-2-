
# coding: utf-8

# In[1]:

from skimage import segmentation, util
import numpy as np
import cv2
from matplotlib import pyplot as plt
get_ipython().magic('matplotlib tk')


# In[41]:

im = cv2.imread('ex.png')
print(im.shape)
#resize image


# In[42]:

#im_new = cv2.GaussianBlur(im, (3,3), 5)
#im_new = cv2.bilateralFilter(im, 3, 50, 50)
im_new = im


# In[43]:

plt.imshow(im_new)


# In[24]:

# newHeight = 256
# newWidth = int(im_new.shape[1]*256/im_new.shape[0])
# im_new = cv2.resize(im_new, (newWidth, newHeight))


# In[62]:

im_mask = segmentation.felzenszwalb(
        util.img_as_float(im_new), scale=200.0, sigma= 0.75,
        min_size=10)


# In[63]:

plt.imshow(im_mask)


# In[26]:

im_mask.shape


# In[27]:

im_mask


# In[28]:

plt.imshow(im_mask)


# In[ ]:
