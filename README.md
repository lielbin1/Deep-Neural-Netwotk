# README

## MNIST Classification with Deep Neural Network (DNN)

This repository contains an implementation of a Deep Neural Network (DNN) from scratch to classify the MNIST dataset.

### Overview
- **Goal**: Train a DNN to accurately classify handwritten digits from the MNIST dataset.
- **Implementation**: The DNN is implemented from scratch using NumPy, without relying on deep learning frameworks like TensorFlow or PyTorch.

### Features
- **Forward Propagation**: Implemented linear and activation functions (ReLU, Softmax).
- **Backward Propagation**: Computed gradients for all parameters.
- **Parameter Updates**: Updated weights and biases using gradient descent.
- **Batch Normalization**: Optional batch normalization layer.
- **L2 Regularization**: Optional L2 regularization to prevent overfitting.

### Results
- The model was trained with and without batch normalization and L2 regularization to compare their effects on performance and training time.

### Dependencies
- Python 3.9
- NumPy
- TensorFlow (for data loading and preprocessing)
- scikit-learn

### Usage
1. **Clone the repository**:
   ```bash
   git clone https://github.com/lielbin1/Deep-Neural-Netwotk.git
   cd Deep-Neural-Netwotk
   ```

2. **Run the Jupyter notebook**:
   ```bash
   jupyter notebook DNN.ipynb
   ```

4. **Train the Model**:
   Follow the steps in the notebook to train the DNN on the MNIST dataset with and without batch normalization and L2 regularization.

### Author
- **Liel Binyamin**
