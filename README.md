# Handwritten Digit Classifier

A neural network-based classifier for recognizing handwritten digits using PyTorch and the MNIST dataset.

## Project Overview

This project implements a feedforward neural network that learns to classify handwritten digits (0-9) with high accuracy. The model is trained on the classic MNIST dataset and can also recognize custom handwritten digit images.

## Screenshots

<div align="center">

### Training Progress

<img width="364" height="232" alt="image" src="https://github.com/user-attachments/assets/aceca47e-767a-4e73-a0b7-32f228d0fa8d" />

*Model training with loss decreasing over epochs*

### Custom Image Testing

<img width="478" height="403" alt="image" src="https://github.com/user-attachments/assets/0e0aefd9-799f-4079-b37b-de8f0a37c768" />

*Testing on user-provided handwritten digits*

### Model Architecture

<img width="908" height="498" alt="image" src="https://github.com/user-attachments/assets/2af9010e-9fc3-41ad-b8e4-1288b5224eb7" />

*Neural network architecture diagram*

</div>

---

## Features

- **Deep Learning Architecture**: 3-hidden-layer neural network with ReLU activations
- **High Accuracy**: Achieves ~95%+ accuracy on MNIST test set
- **Custom Image Support**: Preprocesses and classifies user-provided handwritten digits
- **Batch Processing**: Efficiently processes multiple test images simultaneously
- **Detailed Metrics**: Tracks training loss per epoch and provides accuracy reports

## Tech Stack

- **PyTorch**: Deep learning framework
- **torchvision**: Dataset handling and transformations
- **NumPy**: Numerical operations
- **Pillow (PIL)**: Image processing
- **Matplotlib**: Visualization (optional)

## Model Architecture

```
Input Layer:    784 neurons (28x28 flattened image)
Hidden Layer 1: 64 neurons + ReLU
Hidden Layer 2: 64 neurons + ReLU
Hidden Layer 3: 64 neurons + ReLU
Output Layer:   10 neurons (digits 0-9) + Log Softmax
```

**Loss Function**: Negative Log-Likelihood Loss (NLL)  
**Optimizer**: Adam (learning rate: 0.005)  
**Training**: 10 epochs with batch size of 10

## Results

- **MNIST Test Accuracy**: ~95%+
- **Training Time**: ~10 epochs
- **Model Parameters**: ~52,000 trainable parameters
- **Custom Image Testing**: Validates on user-provided test images

## Getting Started

### Prerequisites

```
pip install torch torchvision pillow numpy matplotlib
```

### Training the Model

```
python num_detector.py
```

The script will:
1. Download the MNIST dataset (if not already present)
2. Train the neural network for 10 epochs
3. Display loss per epoch
4. Evaluate on the test set
5. Test on custom images in the `tests/` folder

### Testing Custom Images

1. Create a `tests/` folder in the project directory
2. Add PNG images named in the format: `testX_description.png` (e.g., `test3_one.png` for digit 3)
3. Run the script - it will automatically process and classify your images

**Image Requirements**:
- Format: PNG
- Naming: `testX_xxx*.png` where X is the actual digit and xxx is (anyof "one" "two" ...)
- Content: White digit on black background (or will be inverted)
- Any size (will be resized to 28x28)

## Project Structure

```
.
├── num_detector.py          # Main training and inference script
├── tests/                   # Folder for custom test images
│   ├── test0_zero.png
│   ├── test1_one.png
│   └── ...
├── mnist_model.pth          # Saved model weights (optional)
└── README.md
```

## How It Works

### 1. Data Preprocessing
- MNIST images are normalized using mean=0.1307, std=0.3081
- Custom images are converted to grayscale, resized to 28x28, and inverted if needed

### 2. Training Process
- Forward pass through the network
- Calculate loss using NLL
- Backward propagation to compute gradients
- Update weights using Adam optimizer

### 3. Inference
- Process images through the trained network
- Apply softmax to get probability distribution
- Select the digit with highest probability

## Sample Output

```
Epoch 1, Loss: 0.4523
Epoch 2, Loss: 0.2156
...
Epoch 10, Loss: 0.0891
Accuracy: 95.43%

==================================================
Test Results on Custom Images
==================================================
Correct: 8/10
Accuracy: 80.00%

Incorrect Predictions:
  test7_seven.png: expected 7, predicted 1
  test9_nine.png: expected 9, predicted 4
```

## Key Concepts Demonstrated

- Neural network architecture design
- Training loop implementation
- Loss calculation and backpropagation
- Model evaluation and testing
- Image preprocessing pipeline
- Batch processing
- Custom dataset handling

## Future Enhancements

- [ ] Add model checkpointing to save best weights
- [ ] Implement data augmentation for better generalization
- [ ] Create confusion matrix visualization
- [ ] Add learning rate scheduling
- [ ] Build web interface for live digit recognition
- [ ] Experiment with deeper architectures (CNN)
- [ ] Add early stopping based on validation loss

## Notes

- The model uses a simple feedforward architecture - CNNs would likely perform better
- MNIST normalization values (0.1307, 0.3081) are dataset-specific
- For best results with custom images, use clear, centered digits with good contrast

## License

This project is open source and available for educational purposes.

## Contributing

Feel free to fork this project and submit pull requests for any improvements!

---

**Author**: Ishvir Singh Chopra  
**Date**: 27 Oct 2025  
**Contact**: ishvir.chopra@gmail.com  
**Video**: Video coming out soon!
