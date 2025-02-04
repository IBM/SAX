import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F
import os
import torchvision.models as models
import math
from datetime import datetime
import numpy as np
from no_explanation.imagenet_no_explanation import ImageNetResNet as imagenet_net_noe
from attack_files.pgd_file import PGD


print("start", flush=True)
INPUT_SIZE = 224*224
NUM_WORKERS = 9
USE_HALF_VAL = True
START_FROM_OUR_NOE_MODEL = False


transform=transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),  # reverse 50% of images
        transforms.ToTensor()
])

val_transform=transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
])

class ImageNetResNet(nn.Module):
    def __init__(self):
        super(ImageNetResNet, self).__init__()

        if START_FROM_OUR_NOE_MODEL:
            model_filepath = './sufficient_xai/methodological/trained_models/imagenet/imagenet_noe_lr_1e-06'
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model_state = torch.load(model_filepath, map_location=device)
            self.model_resnet = imagenet_net_noe().to(device)
            try:
                self.model_resnet.load_state_dict(model_state)
            except Exception as e:
                print(f'Model - weights mismatch:\n{e}', flush=True)
            num_ftrs = 1000
        else:
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
        if self.forward_status == 0:
            x = self.model_resnet(x)
            if START_FROM_OUR_NOE_MODEL:
                out1 = x
            else:
                out1 = self.fc1(x)
            out2 = torch.sigmoid(self.fc2(x))
            return out1, out2

        if self.forward_status == 1:
            x_orig = x
            mask = self.reconstruction.ge(self.threshold)
            mask2 = self.reconstruction.lt(self.threshold)
            reshaped_mask = mask.view(x_orig.shape[0], 224, 224).unsqueeze(1)
            reshaped_mask2 = mask2.view(x_orig.shape[0], 224, 224).unsqueeze(1)

            # Duplicate it 3 times along the third dimension
            final_mask = reshaped_mask.repeat(1, 3, 1, 1)
            final_mask2 = reshaped_mask2.repeat(1, 3, 1, 1)

            if self.complementry == 'noise':
                new_tensor_noise = (torch.rand(x_orig.shape[0], 3, 224, 224).to(torch.device("cuda" if torch.cuda.is_available() else "cpu")) * self.epsilon) + x_orig
                new_tensor_noise[new_tensor_noise > 1] = 1
                new_tensor_noise[new_tensor_noise < 0] = 0
                new_tensor = (x_orig * final_mask) + (new_tensor_noise * final_mask2)
            elif self.complementry == 'attack':
                new_tensor = (x_orig * final_mask) + (self.new_tensor_attack * final_mask2)
            else:
                new_tensor = (x_orig * final_mask)

            new_tensor = self.model_resnet(new_tensor)
            if START_FROM_OUR_NOE_MODEL:
                y2 = new_tensor
            else:
                y2 = self.fc1(new_tensor)
            return y2


def train_model(learning_rate, gamma_value, epsilon_pgd, step_size_pgd, number_of_steps_pgd, threshold, batch_size, num_of_epochs, complementry, evaluate_adversarial, use_explainability, beta_value):
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
    
    class_names = sorted(os.listdir('./sufficient_xai/data/imagenet/imagenet/ILSVRC/Data/CLS-LOC/train'))
    num_classes = len(class_names)
    model = ImageNetResNet()
    model.threshold = threshold
    model.complementry = complementry
    model.epsilon = epsilon_pgd

    # Move model to GPU if available:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f'imagenet.py::train_model: Using {"GPU" if torch.cuda.is_available() else "CPU"} | device={device}', flush=True)
    model = model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    if multiple_gpus:
        model = nn.DataParallel(model)

    # Initialize optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    criterion = nn.CrossEntropyLoss()
    min_val_loss = float('inf')
    count_val_loss = 0

    # Training loop
    for epoch in range(num_of_epochs):
        print(f'imagenet.py::train_model: Training epoch {epoch}', flush=True)
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        for index, (images, labels) in enumerate(train_loader):
            if (index % 100 == 0) and (index > 0):
                print(f'imagenet.py::train_model: Training on batch {index} out of {math.ceil(len(train_dataset)/batch_size)}', flush=True)

            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            model.forward_status = 0
            outputs, explanation = model(images)
            if complementry == 'attack':
                attack = PGD(model, eps=epsilon_pgd, alpha=step_size_pgd, steps=number_of_steps_pgd,
                                          random_start=True)
                adv_images = attack(images, labels, reconstruction=explanation, threshold=threshold)
                model.new_tensor_attack = adv_images
            model.forward_status = 1
            model.reconstruction = explanation
            outputs2 = model(images)
            cardinality_loss = gamma_value * torch.norm(explanation, p=1) / (images.shape[0] * INPUT_SIZE)
            if use_explainability:
                loss = criterion(outputs, labels) + criterion(outputs2, torch.argmax(outputs, axis=1)) + cardinality_loss
            else:
                loss = criterion(outputs, labels)
            if multiple_gpus:
                loss.sum().backward()
            else:
                loss.backward()
            optimizer.step()
            if multiple_gpus:
                train_loss += loss.sum().item()
            else:
                train_loss += loss.item()
            _, predicted = outputs.max(1)
            total_train += labels.size(0)
            correct_train += predicted.eq(labels).sum().item()

        train_accuracy = 100. * correct_train / total_train
        avg_train_loss = train_loss / len(train_loader)

        model.eval()

        val_loss = 0.0
        correct_val = 0
        total_val = 0
        counter = 0
        total_regularization = 0
        correct_explanation = 0
        optimizer.zero_grad()

        for index, (images, labels) in enumerate(val_loader):
            if (index % 100 == 0) and (index > 0):
                print(f'imagenet.py::train_model: Validating on batch {index} out of {math.ceil(len(val_dataset)/batch_size)}', flush=True)

            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            model.forward_status = 0
            outputs, explanation = model(images)
            if complementry == 'attack':
                attack = PGD(model, eps=epsilon_pgd, alpha=step_size_pgd, steps=number_of_steps_pgd,
                                          random_start=True)
                
                adv_images = attack(images, labels, reconstruction=explanation, threshold=threshold)
                model.new_tensor_attack = adv_images
            model.forward_status = 1
            model.reconstruction = explanation
            outputs2 = model(images)

            num_positive = (explanation >= threshold).sum().item()
            total_positive = num_positive / outputs2.numel()
            regularization_term = torch.tensor(total_positive)
            total_regularization += regularization_term

            cardinality_loss = gamma_value * torch.norm(explanation, p=1) / (images.shape[0] * INPUT_SIZE)
            if use_explainability:
                loss = criterion(outputs, labels) + beta_value * criterion(outputs2, torch.argmax(outputs, axis=1)) + cardinality_loss
            else:
                loss = criterion(outputs, labels)

            if multiple_gpus:
                val_loss += loss.sum().item()
            else:
                val_loss += loss.item()
            _, predicted = outputs.max(1)
            _, predicted_explanation = outputs2.max(1)
            correct_explanation += predicted_explanation.eq(predicted).sum().item()

            total_val += labels.size(0)
            correct_val += predicted.eq(labels).sum().item()
            counter+=1

        optimizer.zero_grad()

        val_accuracy = 100 * correct_val / total_val
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy_explanation = 100.0 * correct_explanation / total_val
        regularization_total = 100.0 * total_regularization / counter

        dt_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        print(f"imagenet.py::train_model: {dt_string} - "
            f"Epoch [{epoch + 1}/{num_of_epochs}] - "
            f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f} - "
            f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f} -"
            f"Val Acc explanation: {val_accuracy_explanation:.4f} - "
            f"Percentage sufficiency: {regularization_total:.4f}", flush=True)

        if avg_val_loss <= min_val_loss:
            min_val_loss = avg_val_loss
            torch.save(model.state_dict(), f'./sufficient_xai/methodological/trained_models/imagenet/imagenet_{complementry}_lr_{learning_rate}_g_{gamma_value}_e_{epsilon_pgd}')
            count_val_loss = 0
        else:
            count_val_loss += 1
        if count_val_loss == 5:
            break


def run_main_code_imagenet(learning_rate, gamma_value, epsilon_pgd, step_size_pgd, number_of_steps_pgd, threshold, batch_size, num_of_epochs, complementry, evaluate_adversarial, use_explainability, beta_value):
    train_model(learning_rate, gamma_value, epsilon_pgd, step_size_pgd, number_of_steps_pgd, threshold, batch_size, num_of_epochs, complementry, evaluate_adversarial, use_explainability, beta_value)
