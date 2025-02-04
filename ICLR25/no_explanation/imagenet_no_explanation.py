import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import torchvision.models as models


TRAIN_TEST_SPLIT = 0.9
INPUT_SIZE = 224*224
INPUT_SINGLE_DIMENSION = 28
NUM_WORKERS = 9
USE_HALF_VAL = True

class ImageNetResNet(nn.Module):
    def __init__(self):
        super(ImageNetResNet, self).__init__()
        
        self.model_resnet = models.resnet50(pretrained=True)

        num_ftrs = self.model_resnet.fc.in_features
        self.model_resnet.fc = nn.Identity()
        self.fc1 = nn.Linear(num_ftrs, 1000)
        self.fc2 = nn.Linear(num_ftrs, 224*224)

        self.threshold = None
        self.forward_status = 0
        self.reconstruction = None
        self.put_random_noise = True
        self.epsilon = None
        self.complementry = None

    def forward(self, x):
        x = self.model_resnet(x)
        out1 = self.fc1(x)
        return out1

def train_model(learning_rate, gamma_value, epsilon_pgd, step_size_pgd, number_of_steps_pgd, threshold, batch_size, num_of_epochs, complementry, evaluate_adversarial, use_explainability):
    transform=transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),  # reverse 50% of images
            transforms.ToTensor()
    ])

    val_transform=transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
    ])

    multiple_gpus = False

    train_dataset = datasets.ImageFolder(root='./sufficient_xai/data/imagenet/imagenet/ILSVRC/Data/CLS-LOC/train', transform=transform)
    if USE_HALF_VAL:
        val_dataset = datasets.ImageFolder(root='./sufficient_xai/data/imagenet/imagenet/ILSVRC/Data/CLS-LOC/val', transform=val_transform)
    else:
        val_dataset = datasets.ImageFolder(root='./sufficient_xai/data/imagenet/imagenet/ILSVRC/Data/CLS-LOC/train', transform=val_transform)
    
    if USE_HALF_VAL:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
        indices = list(range(len(val_dataset)))
        # Take the second half of ImageNet's validation set to be the validation set (the first half will be used as test):
        valid_idx = indices[(len(indices) + 1) // 2:]
        valid_sampler = SubsetRandomSampler(valid_idx)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler = valid_sampler, num_workers=NUM_WORKERS, pin_memory=True)
    else:
        valid_size = 0.1
        num_train = len(train_dataset)
        indices = list(range(num_train))
        split = int(np.floor(valid_size * num_train))

        np.random.shuffle(indices)

        train_idx, valid_idx = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, pin_memory=True, sampler = train_sampler, num_workers=NUM_WORKERS)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, pin_memory=True, sampler = valid_sampler, num_workers=NUM_WORKERS)

    # Initialize the network, loss function, and optimizer:
    net = ImageNetResNet()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('gpu' if torch.cuda.is_available() else 'cpu', flush=True)
    print("the device", flush=True)
    print(device, flush=True)
    net = net.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    if multiple_gpus:
        net = nn.DataParallel(net)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    net.threshold = threshold
    net.epsilon = epsilon_pgd

    # Training loop
    num_epochs = num_of_epochs
    min_val_loss = float('inf')
    count_val_loss = 0

    for epoch in range(num_epochs):
        net.train()
        train_loss = 0.0
        correct = 0
        total = 0
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            if batch_idx % 100 == 0:
                print(str(batch_idx) +" out of " + str(len(train_dataset)/batch_size), flush=True)

            digit_output = net(inputs)
            
            loss = criterion(digit_output, labels)
            if multiple_gpus:
                loss.sum().backward()
            else:
                loss.backward()
            optimizer.step()

            if multiple_gpus:
                train_loss += loss.sum().item()
            else:
                train_loss += loss.item()

            _, predicted = digit_output.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_accuracy = 100.0 * correct / total
        net.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        #random noise
        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(val_loader):

                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                if batch_idx % 100 == 0:
                    print(str(batch_idx) +" out of " + str(len(train_dataset)), flush=True)

                digit_output = net(inputs)
                loss = criterion(digit_output, labels)

                if multiple_gpus:
                    val_loss += loss.sum().item()
                else:
                    val_loss += loss.item()
                _, predicted = digit_output.max(1)

                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        optimizer.zero_grad()
        val_loss = val_loss/total
        val_accuracy = 100.0 * correct / total

        print(f"Epoch [{epoch+1}/{num_epochs}] - "
              f"Train Loss: {train_loss/total:.4f}, Train Acc: {train_accuracy:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%", flush=True)

        if val_loss <= min_val_loss:
            min_val_loss = val_loss
            torch.save(net.state_dict(), f'./sufficient_xai/methodological/trained_models/imagenet/imagenet_noe_lr_{learning_rate}')
            count_val_loss = 0
        else:
            count_val_loss += 1
        if count_val_loss == 5:
            break


def run_main_code_imagenet_no_explanation(learning_rate, gamma_value, epsilon_pgd, step_size_pgd, number_of_steps_pgd, threshold, batch_size, num_of_epochs, complementry, evaluate_adversarial, use_explainability):
    train_model(learning_rate, gamma_value, epsilon_pgd, step_size_pgd, number_of_steps_pgd, threshold, batch_size, num_of_epochs, complementry, evaluate_adversarial, use_explainability)
