import torch
import os
from torchvision import datasets, transforms
from PIL import Image
import numpy as np
import PIL.ImageOps
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

class Network(nn.Module):
    
    def __init__(self):
        super().__init__()
        # image -> 28x28
        self.input_layer = nn.Linear(784, 64)
        self.hidden1 = nn.Linear(64, 64)
        self.hidden2 = nn.Linear(64, 64)
        self.output = nn.Linear(64, 10)
        
    def forward(self, data):
        data = F.relu(self.input_layer(data))
        data = F.relu(self.hidden1(data))
        data = F.relu(self.hidden2(data))
        data = self.output(data)
        
        return F.log_softmax(data, dim=1)


training = datasets.MNIST("", train=True, download=True,
                          transform = transforms.Compose([transforms.ToTensor()]))

testing = datasets.MNIST("", train=False, download=True,
                          transform = transforms.Compose([transforms.ToTensor()]))


train_set = torch.utils.data.DataLoader(training, batch_size=10, shuffle=True)

test_set = torch.utils.data.DataLoader(testing, batch_size=10, shuffle=True)


network = Network()

# Load network
#network.load_state_dict(torch.load("mnist_model.pth"))
#network.eval()  # important for testing/inference

learn_rate = optim.Adam(network.parameters(), lr=0.005)
epochs = 10

for i in range(epochs):
    total_loss = 0
    for data in train_set:
        image, output = data
        network.zero_grad()
        result = network(image.view(-1,784))
        loss = F.nll_loss(result,output)
        loss.backward()
        learn_rate.step()
        total_loss += loss.item()
    print(f"Epoch {i+1}, Loss: {total_loss / len(train_set)}")

# Test network
network.eval()

correct = 0
total = 0

with torch.no_grad():
    for data in test_set:
        image, output = data
        result = network(image.view(-1,784))
        for index, tensor_value in enumerate(result):
            total += 1
            if torch.argmax(tensor_value) == output[index]:
                correct += 1

# Save network
# torch.save(network.state_dict(), "mnist_model.pth")

accuracy = correct / total
print(f"Accuracy: {accuracy}")

# Image processing

def preprocess_image(path): 
    img = Image.open(path).convert("L") 
    img = img.resize((28, 28)) 
    img = PIL.ImageOps.invert(img) 
    img = np.array(img, dtype=np.float32) / 255.0 
    img = (img - 0.1307) / 0.3081 # MNIST normalization 
    return torch.from_numpy(img).view(1, 784) # shape [1,784]


folder = "tests"
images = []
filenames = []

for file in os.listdir(folder):
    if file.endswith(".png"):
        path = os.path.join(folder, file)
        images.append(preprocess_image(path))
        filenames.append(file)

# Combine all into one tensor [num_images, 784]
batch = torch.cat(images, dim=0)

network.eval()
with torch.no_grad():
    results = network(batch)
    preds = torch.argmax(results, dim=1)

# Calculate accuracy on test images
correct_predictions = 0
total_predictions = 0
incorrect_files = []

for i, file in enumerate(filenames):
    # Extract true label from filename (e.g., "test3_one.png" -> 3)
    true_label = int(file.split('_')[0].replace('test', ''))
    predicted_label = preds[i].item()
    
    total_predictions += 1
    if predicted_label == true_label:
        correct_predictions += 1
    else:
        incorrect_files.append((file, true_label, predicted_label))

# Print results
print(f"\n{'='*50}")
print(f"Test Results on Custom Images")
print(f"{'='*50}")
print(f"Correct: {correct_predictions}/{total_predictions}")
print(f"Accuracy: {correct_predictions/total_predictions*100:.2f}%")

if incorrect_files:
    print(f"\nIncorrect Predictions:")
    for file, true, pred in incorrect_files:
        print(f"  {file}: expected {true}, predicted {pred}")
else:
    print(f"\nAll predictions correct")

# Show a single image and test result
'''
img = Image.open("tests/test0_one.png")
img = img.resize((28,28))
img = img.convert("L")
img = PIL.ImageOps.invert(img)

plt.imshow(img)

img = np.array(img)
img = img / 255
img = (img - 0.1307) / 0.3081

image = torch.from_numpy(img)
image = image.float()

result = network.forward(image.view(-1,784))
print(torch.argmax(result))
'''

