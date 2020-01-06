# Image Denoising System implementation on PyTorch, based on [Noise2Noise](https://arxiv.org/abs/1803.04189) (Lehtinen et al. 2018)

We attempt to reconstruct a clean image from noisy images using machine learning techniques. It uses Convolutional Neural Networks in particular to achieve this and describes various approaches attempted by the authors in order to do this. It focuses on training a Denoiser with pairs of noisy images to predict an image, which is as good as a clean image. The current problem is that the clean targets which are necessary for training a denoiser, are difficult to obtain. For example, images taken from the Hubble telescope are too costly, as they require shutter to be opened for prolonged periods of time to get clean target images, but long exposures are not cost efficient. So, this paper focuses on corrupt or noisy images as well as noisy targets, and exploits how it can be prolific and cost efficient, with or without clean images, using algorithms like U-Net with skip connections.
In Convolutional Neural Networks, the success rate is more for simpler images, but it doesn’t give good results for complex images like Astrophotography, which is abundant with Noisy H-alpha spectrum. In CNNs the images is converted into vectors and mostly used for classification problems. On the other hand, if a model has a requirement for individual pixel classification of the image, it can be solved with U-Net. It converts image into vector and then convert it back into an image, which preserves the real structure of the image. For example, corrupt Image with randomly positioned and oriented strings just pasted on top of it, is fixed by using U-Net Structure, which learns in a supervised fashion to detect where the strings are and then removes them.




# Dependencies
  - [Python 3.0.x](https://www.python.org/download/releases/3.0/)
  - [PyTorch](https://pytorch.org/)
  - [Torchvision ](https://pytorch.org/docs/stable/torchvision/index.html)
  - [NumPy](https://numpy.org)
  - [Matplotlib](https://matplotlib.org/)
  - [Pillow](https://pillow.readthedocs.io/en/stable/)
  - [OpenCV](https://opencv.org/)
  - [Scipy](https://www.scipy.org/)
  
 A system with GPU is required.
### Technologies
  - [Jupyter notebook](https://jupyter.org/)
  - [Google colab](https://colab.research.google.com/notebooks/welcome.ipynb#recent=true)





### Dataset
As System is trained by noisy images, any image dataset would work, in this case we have a small data set of about 300 training images and about 100 for validation.

### Training
System is trained by adding noisy images specifically:
  - [Gaussian noise](https://en.wikipedia.org/wiki/Gaussian_noise)
  - [Poisson noise](https://en.wikipedia.org/wiki/Shot_noise)
  - [Multiplicative bernoulli noise](https://arxiv.org/abs/1506.03208)
  
ELU activation function is used as well as LeakyReLU.
### Results
In each case where we applied ELU activation function, the training loss and validation loss had better values compared to when we applied LeakyReLU on the model.

