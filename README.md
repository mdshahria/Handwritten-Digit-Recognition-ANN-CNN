# Handwritten Digit Recognition using ANN & CNN (MNIST)

This repository contains the implementation of **Artificial Neural Network (ANN)** and **Convolutional Neural Network (CNN)** models for handwritten digit recognition using the **MNIST dataset**.

The project was developed as part of **EEE385L – Machine Learning Laboratory (Summer 2025)** at **BRAC University**.

---

## 📌 Project Overview

Handwritten digit recognition is a fundamental computer vision task widely used in:
- Banking systems
- Postal services
- Document automation
- Digital security

This project compares ANN and CNN architectures to analyze performance differences in accuracy, loss, and generalization.

---

## 📂 Dataset

- **Dataset:** MNIST
- **Images:** 70,000 grayscale images (28×28 pixels)
- **Classes:** Digits 0–9
- **Source:** Built-in Keras dataset

---

## 🧠 Model Architectures

### Artificial Neural Network (ANN)
- Input: 784 neurons
- Hidden Layers: 512, 256 (ReLU)
- Output: 10 neurons (Softmax)
- Optimizer: Adam
- Epochs: 10
- Accuracy: ~97–98%

### Convolutional Neural Network (CNN)
- Conv2D (32 filters, 3×3) + MaxPooling
- Conv2D (64 filters, 3×3) + MaxPooling
- Dense: 128 neurons (ReLU)
- Output: 10 neurons (Softmax)
- Optimizer: Adam
- Epochs: 10
- Accuracy: ~98–99%

---

## 📊 Results

| Model | Accuracy | Baseline Error |
|------|---------|----------------|
| ANN  | ~97–98% | ~2.1% |
| CNN  | ~98–99% | ~0.97% |

CNN outperforms ANN due to better spatial feature extraction.

---

## ▶️ How to Run

### Install dependencies
```bash
pip install -r requirements.txt
