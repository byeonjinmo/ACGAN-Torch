# ACGAN-Torch
Implementing ACGAN with Torch using the cifar10 dataset Labeling car classes 


The learning of ACGAN in the existing model was incomplete, but we attempted to reduce mode decay and generate higher quality images through the following process. 

1. increasing the network complexity of the generator and discriminator
2. change the activation function to LeakyReLU 
3. using filters during the dataset import process 
4. changed the weight initialization to xavier_normal 
5. increased training frequency 

Finally, print the final average loss and accuracy of the model as a benchmark to make the process more numerical.
