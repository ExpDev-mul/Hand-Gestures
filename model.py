# This is the neural-network model that classifies the image with it's respective label

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
import torchvision.models as models
from PIL import Image

IMG_SIZE = 130
TRAIN = False

dataset = np.load('dataset.npz')

TOTAL_CLASSES = len(dataset)

class NN(nn.Module):
    def __init__(self, num_classes=10):
        super(NN, self).__init__()
        self.vgg16 = models.vgg16(pretrained=True)

        for param in self.vgg16.parameters():
            param.requires_grad = False

        num_features = self.vgg16.classifier[-1].in_features
        self.vgg16.classifier[-1] = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.vgg16(x)

class CustomImageDataset(Dataset):
    def __init__(self, data_dict, transform=None):
        self.data_dict = data_dict
        self.transform = transform
        self.image_data = []
        self.labels = []

        for label, images in self.data_dict.items():
            for img_array in images:
                self.image_data.append(img_array)
                self.labels.append(label)

    def __len__(self):
        return len(self.image_data)

    def __getitem__(self, idx):
        image_array = self.image_data[idx]
        label = self.labels[idx]

        image = Image.fromarray(image_array.astype('uint8'), mode='L' if image_array.ndim == 2 else 'RGB')

        if self.transform:
            image = self.transform(image)

        return image, label

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])


def load_model():
    model = torch.load('model.pth', map_location=torch.device('cpu'))
    model.eval()
    return model

def predict(img, model):
    img = Image.fromarray(img.astype('uint8'))
    img = transform(img)
    img = img.unsqueeze(0)
    with torch.no_grad():
        output = model(img)
    _, predicted = torch.max(output, 1)
    return [predicted.item(), output]

print(dataset.keys())

if TRAIN:
    model = NN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1E-4)

    custom_dataset = CustomImageDataset(data_dict=dataset, transform=transform)
    data_loader = DataLoader(custom_dataset, batch_size=64, shuffle=True)

    epochs = 20
    label_encoder = LabelEncoder()
    label_encoder.fit(custom_dataset.labels)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for images, labels in data_loader:
            labels = label_encoder.transform(labels) # Encode string labels into corresponding integers
            labels = torch.tensor(labels).long() # Convert into a tensor
            optimizer.zero_grad() # Zero out all the parameter gradients
            outputs = model(images) # Forward propogation
            loss = criterion(outputs, labels) # Compute our loss
            loss.backward() # Backward propogation to trian the model
            optimizer.step() # Optimize our architecture
            running_loss += loss.item() # Sum up our running loss

        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(data_loader):.4f}')

    torch.save(model, 'model.pth')
    print("Succesfully saved model at model.pth")