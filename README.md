
# Car Image Classifier



Technologies Used:

- [NumPy](http://www.numpy.org/)
- [matplotlib](http://matplotlib.org/)
- [Pytorch](https://pytorch.org)
- [Google Colaboratory](https://colab.research.google.com)

# Introduction
The project mainly aims to classify car pictures, using ***CNN-based*** approach. 
![Semantic description of image](https://miro.medium.com/max/1204/1*TVu5hyywTrUsCAkfJbVHQw.png "Image Title")

Recently, convolutional neural networks achieve impressive results on a wide range of image classification tasks. CNN-based approaches use dominant CNNs as the main framework for the fine-grained classification.  

Used the pretrained model resnet-152 and applied fine tuning to be able to predict the car model/make/year.



### Data

Stanford University cars dataset. The Cars dataset contains 16,185 images of 196 classes of cars. The data is split into 8,144 training images and 8,041 testing images, where each class has been split roughly in a 50-50 split. Classes are typically at the level of Make, Model, Year, e.g. 2012 Tesla Model S or 2012 BMW M3 coupe.
http://ai.stanford.edu/~jkrause/cars/car_dataset.html
