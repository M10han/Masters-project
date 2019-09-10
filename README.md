# **Learning Audio Features For Genre Classification with Deep Belief Networks**


Feature extraction is a crucial part of many music information retrieval(MIR) tasks. In this project, I present an end to end deep neural network that extract the features for a given audio sample, performs genre classification. The feature extraction is based on Discrete Fourier Transforms (DFTs) of the audio. The extracted features are used to train over a Deep Belief Network (DBN). A DBN is built out of multiple layers of Randomized Boltzmann Machines (RBMs), which makes the system a fully connected neural network. The same network is used for testing with a softmax layer at the end which serves as the classifier. This entire task of genre classification has been done with the Tzanetakis dataset and yielded a test accuracy of 74.6%.

## Technologies used:
- Python3
- Keras
- TensorFlow
- Theano
- NumPy
- Pickle
- LibROSA
- Google Cloud Platform (GCP)
