
# Car Image Classifier

Technologies Used:

- [NumPy](http://www.numpy.org/)
- [matplotlib](http://matplotlib.org/)
- [Pytorch](https://pytorch.org)
- [Google Colaboratory](https://colab.research.google.com)
- [ResNet152](https://pytorch.org/hub/pytorch_vision_resnet/)


# Introduction
Convolutional neural networks achieved impressive results on a wide range of image classification tasks. CNN-based approaches use dominant CNNs as the main framework for the fine-grained classification. what makes it a strong candidate to use in this project where the main goal is to classify cars by picture.

![Semantic description of image](https://miro.medium.com/max/1204/1*TVu5hyywTrUsCAkfJbVHQw.png "Image Title")



## Related work and approaches.

### texture feature-based approaches:
The texture feature-based approaches are designed for traffic monitoring using fixed surveillance cameras. These approaches are limited to the fine-grained classification of frontal car images.

### 3D representation-based approaches:
3D representation-based approaches are able to handle car images with unconstrained poses and multiple viewpoints.

### Convolutional Neural Networks.

Convolutional Neural Networks (CNNs) are at the heart of most CV applications and the approach that will be used in this project. CNNs use the convolution operation to transform input images into outputs. A single step of convolution multiplies and sums the pixel values of an image with the values of a filter. This filter can be of shape 3x3. Next, the filter is shifted to a different position and the convolutional step is repeated until all Pixels were processed at least once. The resulting matrix eventually detects ***edges*** or transitions between dark and light colors and eventually more complex forms. The more filters you apply, the more details the CNN is capable to recognize.

![Semantic description of image](https://miro.medium.com/max/488/1*4h_J0Zpx93_sFHKxWUoHAw.gif "Image Title")

Horizontal edge detection works by creating a horizontal edge in the filter and vice versa for vertical edges. The weights for edge filter detection can be learned through backpropagation instead of manually coding the values because images generally have many complex edges.

Add an additional pixel border around the image to preserve the original image size. This helps to prevent shrinking the input through convolutional filtering. “Valid” padding means that you use zero padding and the size of the image shrinks. “Same” padding adds as much padding as is needed to keep the dimension of the output equal to the input.

![Semantic description of image](https://miro.medium.com/max/790/1*nYf_cUIHFEWU1JXGwnz-Ig.gif "Image Title")

CNN consists of a convolutional layer followed by a pooling layer. At the end, you can use general fully connected layers, which are just flattened pooling layers and eventually generate a result.
The greatest advantages of using convolutions are parameter sharing and sparsity of connections. Parameter sharing is very efficient as it reduces the number of weight parameters in one layer without losing accuracy. Additionally, the convolution operation breaks down the input features into a smaller feature space, so that each output value depends on a small number of inputs and can be quickly adjusted.

![Semantic description of image](https://miro.medium.com/max/1400/1*XbuW8WuRrAY5pC4t-9DZAQ.jpeg "Image Title")

This project mainly aims to classify car pictures using ***CNN-based*** approach and Stanford Car Dataset. The pre-trained ResNet152 as the CNN to classify images.

## ResNet.





### Data

Stanford University cars dataset. The Cars dataset contains 16,185 images of 196 classes of cars. The data is split into 8,144 training images and 8,041 testing images, where each class has been split roughly in a 50-50 split. Classes are typically at the level of Make, Model, Year, e.g. 2012 Tesla Model S or 2012 BMW M3 coupe.
http://ai.stanford.edu/~jkrause/cars/car_dataset.html
