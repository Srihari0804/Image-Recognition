# Image Recognition: Handwritten Digits (MNIST)

## 📌 Overview

This repository marks my first complete computer vision pipeline, focusing on the classic problem of recognizing handwritten numbers. Using **TensorFlow**, this project builds, trains, and evaluates a neural network on the famous **MNIST dataset**, which consists of 70,000 grayscale images of digits from 0 to 9.

All implementation details, from data loading to model evaluation, are contained within the `Digit_recognition.ipynb` notebook.

## ✨ Features & Performance

* **Foundational Computer Vision:** Serves as a practical introduction to image processing, tensor manipulation, and deep learning architectures.
* **TensorFlow Integration:** Utilizes `tf.keras` for building and compiling the neural network layers seamlessly.
* **High Accuracy:** The model effectively learned the underlying patterns of the handwritten digits, achieving a remarkable **0.99 (99%) accuracy** on the unseen test dataset.

## 🧮 The Mathematics (Multiclass Classification)

Because this model predicts one of ten possible digits (0-9), the final layer of the network relies on the **Softmax activation function** to output a probability distribution across all 10 classes.

For an output vector $\mathbf{z}$ from the final dense layer, the probability that the image belongs to class $i$ is calculated as:

$$\sigma(\mathbf{z})_i = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}$$

Where $K=10$ (the total number of digit classes). The model then calculates the error using **Categorical Cross-Entropy** and updates its weights via backpropagation to minimize this loss.

## 🚀 Getting Started

### Prerequisites

To run this notebook and train the model locally, you will need a standard Python data science environment. For optimal performance with TensorFlow, a GPU-accelerated environment using Conda and WSL2 is highly recommended.

* Python 3.8+
* TensorFlow 2.x
* NumPy
* Matplotlib (for image visualization)
* Jupyter Notebook or JupyterLab

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Srihari0804/Image-Recognition.git
cd Image-Recognition

```


2. Activate your development environment and install dependencies:
```bash
conda activate your-tf-env
pip install tensorflow numpy matplotlib jupyter

```



## 💻 Usage

To view the code, model architecture, and training process, launch Jupyter Notebook:

```bash
jupyter notebook Digit_recognition.ipynb

```

Run the cells sequentially to watch the model download the MNIST dataset, train across epochs, and ultimately output its 0.99 accuracy score.

## 📂 Project Structure

* `Digit_recognition.ipynb` - The primary notebook containing the entire end-to-end workflow for the MNIST digit recognition model.
