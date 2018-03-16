import numpy as np
import cv2
import torchvision.transforms as transforms
import torch
from torch.autograd import Variable

"""
The Following example is to show how to transfer the image to the Tensor Format
The image format:  R,G,B for every pixel, and shape is [ height, width, channels ]
transforms.ToTenor will change the shape to [ channels, height, width ]
"""
transform = transforms.ToTensor()

def convert_image_to_tensor(image):
    image = image.astype(np.float)
    return transform(image)

image_path = "../../data/image/test.jpg"

im = cv2.imread(image_path)
print("the im shape is : {}".format(im.shape))
im_nm = np.asarray(im)
print("the im_nm shape is : {}".format(im.shape))
im_tensor = convert_image_to_tensor(im_nm)
print("the im_tensor shape is : {}".format(im_tensor.shape))

"""
Below show when calling the transforms.ToTensor() the data will Normalize to [0,1]
"""
print("im raw data:")
for i in range(5):
    print(im[0][i][0])
print("transfer numpy data:")
for i in range(5):
    print(im_nm[0][i][0])
print("transfer Tensor data:")
for i in range(5):
    print(im_tensor[0][0][i])

"""
below show how to transform batch pics into the Tensors
"""
count = 5
process_im = list()
for i in range(count):
    im = cv2.imread(image_path)
    process_im.append(im)
im_np_array = np.asarray(process_im)
print("the im_array shape is :{}".format(im_np_array.shape))
im_tensor_array = [ convert_image_to_tensor(im_np_array[i,:,:,:]) for i in range(count) ]
print("im_tensor_array type is {}".format(type(im_tensor_array)))
im_tensor_array = torch.stack(im_tensor_array)
print("after call torch.stack im_tensor_array type is {}".format(type(im_tensor_array)))
print("the im_tensor_array shape is: {}".format(im_tensor_array.shape))

"""
change the Tensor to the Variable
"""
im_tensor_array = Variable(im_tensor_array)
print("after call Variable the im_tensor_array type is {}".format(type(im_tensor_array)))
print("the im_tensor_array shape is: {}".format(im_tensor_array.shape))