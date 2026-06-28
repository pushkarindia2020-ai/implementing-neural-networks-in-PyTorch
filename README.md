# PyTorch Neural Network — MNIST Digit Classifier
### Complete line-by-line explanation for beginners

---

## Table of Contents
1. [Import Libraries](#step-1-import-libraries)
2. [Hyperparameters & Transforms](#step-2-hyperparameters--transforms)
3. [Load Dataset](#step-3-load-dataset)
4. [Define the Neural Network](#step-4-define-the-neural-network)
5. [Loss, Optimizer & Device](#step-5-loss-optimizer--device)
6. [Training & Test Loops](#step-6-training--test-loops)
7. [Run Training & Visualize](#step-7-run-training--visualize)
8. [Save & Load Model Weights](#save--load-model-weights)
9. [Sample Check with One Image](#sample-check-with-one-image)

---

## Full Code

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

batch_size = 64
num_epochs = 10
learning_rate = 0.01

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='.', train=True, download=True, transform=transform)
test_dataset  = datasets.MNIST(root='.', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader  = torch.utils.data.DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Net().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, 1)
    return torch.sum(preds == labels).item() / len(labels)

def train(model, device, train_loader, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.0
    running_acc  = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        running_acc  += accuracy(outputs, labels)
        if (i + 1) % 200 == 0:
            print(f'Epoch {epoch}, Batch {i+1}, Loss: {running_loss/200:.4f}, Accuracy: {running_acc/200:.4f}')
            running_loss = 0.0
            running_acc  = 0.0

def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0.0
    test_acc  = 0.0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            test_acc  += accuracy(outputs, labels)
    print(f'Test Loss: {test_loss/len(test_loader):.4f}, Test Accuracy: {test_acc/len(test_loader):.4f}')

for epoch in range(1, num_epochs + 1):
    train(model, device, train_loader, criterion, optimizer, epoch)
    test(model, device, test_loader, criterion)

torch.save(model.state_dict(), 'mnist_model.pth')
```

---

## Step 1: Import Libraries

```python
import torch
```
**The heart of everything.**
`torch` is PyTorch itself. It gives you tensors (multi-dimensional number arrays), automatic differentiation, and GPU support. Think of it as NumPy but built for deep learning.

---

```python
import torch.nn as nn
```
**Neural network building blocks.**
`nn` stands for "neural network." It contains pre-built layers (`Linear`, `Conv2d`), loss functions (`CrossEntropyLoss`), and the base `Module` class that all models must inherit from.

---

```python
import torch.nn.functional as F
```
**Activation functions and more.**
`F` gives you functions like `F.relu()`, `F.sigmoid()`, `F.softmax()`. Unlike `nn` layers, these are stateless — they take an input, transform it, and return output with no stored weights of their own.

---

```python
import torch.optim as optim
```
**Optimizers — the "learning" mechanism.**
`optim` contains algorithms like SGD, Adam, RMSProp that decide how to update model weights after each backward pass. Without this, the model can see its errors but cannot fix them.

---

```python
from torchvision import datasets, transforms
```
**Image dataset helpers.**
`torchvision` is a separate package (install with `pip install torchvision`).
- `datasets` gives you MNIST, CIFAR10 etc. already packaged and ready.
- `transforms` gives you image preprocessing tools like `ToTensor()` and `Normalize()`.

---

```python
import matplotlib.pyplot as plt
```
**For drawing images and graphs.**
`plt.imshow()` shows an image. `plt.plot()` draws a line chart. Used at the very end to visually display sample predictions. Not needed during training itself.

---

```python
import numpy as np
```
**Numerical arrays in Python.**
`matplotlib` works with NumPy arrays, not PyTorch tensors. So you convert tensors → NumPy arrays before plotting. `np` is used minimally here — just for that conversion step.

---

## Step 2: Hyperparameters & Transforms

> Hyperparameters are knobs you set manually before training starts. The model does NOT learn these — you decide them.

```python
batch_size = 64
```
**How many images to process together.**
Instead of feeding the model 1 image or all 60,000 at once, you feed 64 at a time. Larger batches = more stable gradients but more memory. 64 is a safe, standard starting value.

---

```python
num_epochs = 10
```
**How many times to go through the entire dataset.**
1 epoch = the model sees all 60,000 training images once. 10 epochs means it sees them 10 times total, getting a bit better each pass. Too many epochs → overfitting (memorizing training data).

---

```python
learning_rate = 0.01
```
**Size of each weight correction step.**
After backprop tells us how wrong each weight is, `learning_rate` controls how much we adjust it. 0.01 means "fix 1% of the error each step." Too high = overshooting. Too low = very slow learning.

---

```python
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
```
**Chain multiple transforms into one pipeline.**
`Compose` takes a list of transforms and applies them in order — like a conveyor belt of image processing. Every image passes through each transform before being fed to the model.

- **`transforms.ToTensor()`** — Converts a PIL image (pixels 0–255) into a PyTorch float tensor (values 0.0–1.0). Also changes shape from `(H, W)` → `(C, H, W)`, adding the channel dimension.

- **`transforms.Normalize((0.1307,), (0.3081,))`** — Shifts pixel values so the dataset has mean ≈ 0 and std ≈ 1. Formula: `(pixel - 0.1307) / 0.3081`. These exact numbers are the pre-calculated mean and std of the entire MNIST dataset. Normalizing helps gradients flow smoothly during training.

---

## Step 3: Load Dataset

> MNIST has 70,000 handwritten digit images (0–9). 60,000 for training, 10,000 for testing. PyTorch downloads and manages them automatically.

```python
train_dataset = datasets.MNIST(
    root='.',
    train=True,
    download=True,
    transform=transform
)
```

- **`root='.'`** — Where to save the downloaded files. `'.'` = current folder. PyTorch creates an `MNIST/` subfolder. If files already exist, download is skipped.
- **`train=True`** — Gets the training split (60,000 images). Use `train=False` for the test split (10,000 images).
- **`download=True`** — Downloads from the internet if not already on disk. Safe to always leave as `True`.
- **`transform=transform`** — Every time an image is loaded, it automatically runs through `ToTensor()` then `Normalize()`.

---

```python
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)
```
**A conveyor belt that feeds batches to the model.**
`DataLoader` sits on top of the Dataset and handles batching, shuffling, and parallel loading automatically.

- **`batch_size=batch_size`** — Each iteration gives a tensor of shape `[64, 1, 28, 28]` — 64 grayscale 28×28 images.
- **`shuffle=True`** — Randomizes image order every epoch. Without this, the model always sees digits in the same order and might learn to predict based on position rather than image features.
- For `test_loader`, use `shuffle=False` — order doesn't matter for evaluation.

---

## Step 4: Define the Neural Network

> Data flow through the model:
> `Image [64,1,28,28]` → `Flatten [64,784]` → `fc1 + ReLU [64,512]` → `fc2 [64,10]` → `Prediction`

```python
class Net(nn.Module):
```
**Create a class that IS a neural network.**
By inheriting from `nn.Module`, our `Net` class automatically gets PyTorch's weight tracking, GPU support, `.train()`/`.eval()` mode switching, and `.parameters()` for the optimizer. Every PyTorch model must inherit from `nn.Module`.

---

```python
def __init__(self):
    super(Net, self).__init__()
```
**Constructor — runs once when you write `Net()`.**
`super(Net, self).__init__()` calls the parent class (`nn.Module`) constructor. This sets up internal PyTorch machinery for weight tracking. **If you forget this line, nothing will work** — you'll get a confusing error about missing attributes.

---

```python
self.fc1 = nn.Linear(28 * 28, 512)
```
**First fully connected layer: 784 inputs → 512 outputs.**
`nn.Linear` creates a weight matrix of shape `(784, 512)` = 401,408 weights, plus 512 bias values. "fc" = "fully connected" = every input connects to every output. 784 because 28×28 = 784 pixels flattened. 512 is our design choice for the hidden layer size.

---

```python
self.fc2 = nn.Linear(512, 10)
```
**Second fully connected layer: 512 inputs → 10 outputs.**
Takes the 512 features from fc1 and maps them to 10 scores — one per digit class (0–9). The class with the highest score is the model's prediction. 10 outputs is fixed by the number of classes in MNIST.

---

```python
def forward(self, x):
```
**Defines how data flows through the model.**
PyTorch calls this automatically when you write `model(input)`. The `x` parameter is a batch of input images. You define the order in which layers are applied and what comes out at the end.

---

```python
x = x.view(-1, 28 * 28)
```
**Flatten 2D image → 1D vector.**
Images come in as shape `[64, 1, 28, 28]`. `.view()` reshapes without copying memory. `-1` means "figure out this dimension automatically" (= 64, the batch size). Result: `[64, 784]`. Linear layers need 1D input per sample, not 2D images.

---

```python
x = F.relu(self.fc1(x))
```
**Pass through fc1, then apply ReLU activation.**
`self.fc1(x)` does matrix multiplication: `[64,784] × [784,512]` = `[64,512]`. Then `F.relu()` sets every negative value to 0, keeping positives unchanged. Without ReLU, stacking two linear layers is equivalent to just one — they'd mathematically collapse into a single transformation.

---

```python
x = self.fc2(x)
```
**Pass through fc2 — get raw class scores (logits).**
`[64,512] × [512,10]` = `[64,10]`. These 10 numbers per image are called "logits" — raw unnormalized scores. No activation here because `CrossEntropyLoss` applies Softmax internally. Adding an extra Softmax here would cause incorrect gradient calculation.

---

```python
return x
```
**Send the `[64, 10]` tensor back to the caller.**
The caller is usually the training loop, which passes this to the loss function. Each row = one image's 10 class scores. The model's prediction for image `i` = the column index with the max value in row `i`.

---

## Step 5: Loss, Optimizer & Device

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```
**Auto-detect GPU vs CPU.**
CUDA is NVIDIA's GPU computing framework. If a GPU is available, `torch.cuda.is_available()` returns `True` and training runs ~10–100× faster. If no GPU exists, it falls back to CPU. All model weights and data tensors must be on the same device or PyTorch throws an error.

---

```python
model = Net().to(device)
```
**Create the model and move all its weights to GPU/CPU.**
`Net()` calls the constructor, building fc1 and fc2 with random initial weights. `.to(device)` moves all those weight tensors to the chosen device. This must be done before training.

---

```python
criterion = nn.CrossEntropyLoss()
```
**The loss function — measures how wrong the model is.**
Takes raw logits `[64,10]` and true labels `[64]`, and computes a single number showing how far predictions are from correct. Internally applies log-softmax + negative log-likelihood. Lower loss = better predictions. A random-guessing model on 10 classes has loss ≈ 2.3.

---

```python
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
```
**The update algorithm — decides how to fix weights.**
SGD (Stochastic Gradient Descent) is the simplest optimizer. `model.parameters()` hands it all the weights (~403,000 numbers). After backprop, SGD does: `weight = weight - learning_rate × gradient` for every single weight. That's the entire learning process.

---

## Step 6: Training & Test Loops

> **The 5-step training heartbeat — memorize this:**
> `zero_grad()` → `forward pass` → `compute loss` → `backward()` → `step()`
> This repeats for every single batch.

```python
model.train()
```
**Put model in training mode.**
This matters for layers like `Dropout` and `BatchNorm` which behave differently during training vs evaluation. Dropout randomly zeroes activations during training (prevents overfitting) but passes everything through during eval. Always call this before your training loop.

---

```python
optimizer.zero_grad()
```
**CRITICAL: Clear gradients from the previous batch.**
PyTorch ACCUMULATES gradients by default — each `.backward()` call adds to existing gradients rather than replacing them. If you forget `zero_grad()`, gradients stack up across batches and training behaves incorrectly. This must be called before every forward pass. This is one of the most common beginner mistakes.

---

```python
outputs = model(inputs)
```
**Forward pass — run inputs through the network.**
This calls our `forward()` method. `inputs` shape: `[64,1,28,28]`. `outputs` shape: `[64,10]`. PyTorch secretly records every mathematical operation here so it can compute gradients later during `backward()`.

---

```python
loss = criterion(outputs, labels)
```
**Compute how wrong the predictions are.**
`criterion` is our `CrossEntropyLoss`. It compares the 10 scores per image against the true label and returns a single scalar number (the loss). A perfect prediction gives loss ≈ 0. Random guessing gives loss ≈ 2.3.

---

```python
loss.backward()
```
**Backpropagation — compute how much each weight contributed to the error.**
PyTorch traces backwards through every operation recorded during `forward()`, using the chain rule to compute a gradient for every single weight. After this, each weight tensor has a `.grad` attribute filled with its gradient value. This is the "learning signal."

---

```python
optimizer.step()
```
**Update all weights using their gradients.**
The SGD optimizer looks at every weight's `.grad` and does: `weight = weight - 0.01 × gradient`. This nudges each weight slightly in the direction that reduces the loss. After ~9,360 of these steps (10 epochs × 936 batches), the model becomes accurate.

---

```python
model.eval()
```
**Switch to evaluation mode — no learning happens here.**
Disables Dropout (all neurons active), uses running stats in BatchNorm. This gives deterministic, stable predictions. Always call before running on test or validation data.

---

```python
with torch.no_grad():
```
**Disable gradient computation entirely.**
During testing we never call `backward()` — so there's no need to record operations. `torch.no_grad()` skips that recording, saving memory and making inference ~2× faster. Every line of code inside this block runs without gradient tracking.

---

```python
_, preds = torch.max(outputs, 1)
```
**Find which class scored highest for each image.**
`outputs` has shape `[64,10]`. `torch.max(outputs, 1)` finds the maximum value along dimension 1 (across the 10 classes). Returns two tensors: the max values (ignored with `_`) and the indices of those max values = predicted class numbers 0–9.

---

```python
return torch.sum(preds == labels).item() / len(labels)
```
**Count correct predictions, divide by batch size.**
`preds == labels` gives a boolean tensor `[True, False, True...]`. `torch.sum()` counts the Trues. `.item()` converts the 1-element tensor to a plain Python number. Dividing by `len(labels)` gives accuracy as a fraction between 0 and 1.

---

## Step 7: Run Training & Visualize

```python
for epoch in range(1, num_epochs + 1):
    train(model, device, train_loader, criterion, optimizer, epoch)
    test(model, device, test_loader, criterion)
```
**Outer loop: train 10 epochs, test after each one.**
`range(1, 11)` gives `[1,2,...,10]`. After each epoch of training, we test on unseen data to track whether the model is genuinely learning or just memorizing.

---

```python
samples, labels = next(iter(test_loader))
```
**Grab one batch of 64 test images.**
`iter()` creates an iterator from the DataLoader. `next()` takes the first batch it yields — 64 real handwritten digit images with their true labels.

---

```python
input_tensor = single_image.unsqueeze(0).to(device)
```
**Add batch dimension before passing to model.**
Your image is shape `[1, 28, 28]` but the model always expects a batch — shape `[batch, channel, H, W]`. `.unsqueeze(0)` artificially adds a batch dimension of 1, making it `[1, 1, 28, 28]`. This is one of the most common beginner mistakes when doing single-image inference.

---

```python
samples = samples.cpu().numpy()
```
**Move tensor to CPU and convert to NumPy.**
If training ran on GPU, samples is still on the GPU. `.cpu()` moves it to RAM. `.numpy()` converts the PyTorch tensor into a NumPy array. `matplotlib`'s `imshow()` needs a NumPy array — it cannot work with PyTorch tensors.

---

```python
ax.imshow(samples[i].squeeze(), cmap='gray')
```
**Display image as grayscale.**
`samples[i]` has shape `[1,28,28]`. `.squeeze()` removes the channel dimension → `[28,28]`, which `imshow` needs. `cmap='gray'` renders it as grayscale, not a color heatmap.

---

## Save & Load Model Weights

> By default, weights are stored only in RAM — they vanish the moment your script ends. Save them to disk so you never have to retrain from scratch.

### Save after training

```python
torch.save(model.state_dict(), 'mnist_model.pth')
print("Model saved!")
```

`state_dict()` is a Python dictionary of all layer names → their weight tensors. That's literally all a "saved model" is — just a dictionary of numbers written to a file.

### Load next time

```python
model = Net().to(device)                   # create empty architecture first
model.load_state_dict(torch.load('mnist_model.pth', map_location=device))
model.eval()
print("Model loaded!")
```

**You must define the `Net` class again** — PyTorch saves weights only, not the architecture. `map_location=device` handles the case where you saved on GPU but are loading on CPU (or vice versa).

### Smart version — train once, reuse forever

```python
import os

if os.path.exists('mnist_model.pth'):
    model = Net().to(device)
    model.load_state_dict(torch.load('mnist_model.pth', map_location=device))
    model.eval()
    print("Loaded saved weights. Skipping training.")
else:
    for epoch in range(1, num_epochs + 1):
        train(model, device, train_loader, criterion, optimizer, epoch)
        test(model, device, test_loader, criterion)
    torch.save(model.state_dict(), 'mnist_model.pth')
    print("Training done. Weights saved.")
```

| Action | Code |
|--------|------|
| Save   | `torch.save(model.state_dict(), 'file.pth')` |
| Load   | `model.load_state_dict(torch.load('file.pth', map_location=device))` |
| After loading | Always call `model.eval()` |

---

## Sample Check with One Image

### Option 1: Use an image from the test set

```python
import matplotlib.pyplot as plt

index = 42   # any number from 0 to 9999
single_image, true_label = test_dataset[index]

# add batch dimension: [1,28,28] → [1,1,28,28]
input_tensor = single_image.unsqueeze(0).to(device)

model.eval()
with torch.no_grad():
    output = model(input_tensor)           # shape: [1, 10]
    _, predicted = torch.max(output, 1)    # get highest scoring class

print(f"True label  : {true_label}")
print(f"Predicted   : {predicted.item()}")

plt.imshow(single_image.squeeze(), cmap='gray')
plt.title(f"True: {true_label}  |  Predicted: {predicted.item()}")
plt.axis('off')
plt.show()
```

### Option 2: Use your own image file

```python
from PIL import Image

my_transform = transforms.Compose([
    transforms.Grayscale(),            # convert RGB → grayscale if needed
    transforms.Resize((28, 28)),       # resize to 28x28
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

img = Image.open("my_digit.png")      # your image path here
input_tensor = my_transform(img).unsqueeze(0).to(device)

model.eval()
with torch.no_grad():
    output = model(input_tensor)
    probabilities = torch.softmax(output, dim=1)
    confidence, predicted = torch.max(probabilities, 1)

print(f"Predicted digit : {predicted.item()}")
print(f"Confidence      : {confidence.item() * 100:.1f}%")

plt.imshow(img, cmap='gray')
plt.title(f"Predicted: {predicted.item()}  ({confidence.item()*100:.1f}% confident)")
plt.axis('off')
plt.show()
```

### Flow for single image inference

```
your image → same transforms as training → .unsqueeze(0) → model → softmax → prediction
```

### Tips for your own image

| Tip | Why |
|-----|-----|
| White digit on black background | MNIST is formatted this way — inverted images confuse the model |
| Clean, centered digit | Model was trained on centered digits |
| If prediction is wrong | Try `transforms.functional.invert(img)` before processing |

---

## Key Concepts Summary

| Concept | What it does |
|---------|-------------|
| `nn.Module` | Base class all PyTorch models inherit from |
| `nn.Linear(in, out)` | Fully connected layer — matrix multiply + bias |
| `F.relu()` | Activation: kills negatives, keeps positives |
| `CrossEntropyLoss` | Loss function for multi-class classification |
| `SGD` | Optimizer: updates weights using gradients |
| `loss.backward()` | Computes gradients for all weights via backprop |
| `optimizer.step()` | Applies gradients to update weights |
| `optimizer.zero_grad()` | Clears gradients before next batch (required!) |
| `model.train()` | Enables training behaviour (dropout etc.) |
| `model.eval()` | Enables evaluation behaviour |
| `torch.no_grad()` | Disables gradient tracking during inference |
| `.unsqueeze(0)` | Adds batch dimension for single-image inference |
| `state_dict()` | Dictionary of all model weights |

---

*Built with PyTorch. MNIST dataset: 60,000 training images, 10,000 test images, digits 0–9. Final accuracy: ~96.6% after 10 epochs.*
