import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from attack_files.pgd_file import PGD
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import torchvision.models as models

TRAIN_TEST_SPLIT = 0.9
INPUT_SIZE = 1024

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()


        self.model_resnet = models.resnet18(pretrained=False)
        num_ftrs = self.model_resnet.fc.in_features
        self.model_resnet.fc = nn.Identity()
        self.fc1 = nn.Linear(512, 10)
        self.fc2 = nn.Linear(512, 1024)
        self.threshold = None
        self.forward_status = 0
        self.reconstruction = None
        self.complementry = None
        self.epsilon = None


    def forward(self, x):
        if self.forward_status == 0:
            x = self.model_resnet(x)
            out1 = self.fc1(x)
            out2 = torch.sigmoid(self.fc2(x))
            return out1, out2

        if self.forward_status == 1:
            x_orig = x
            mask = self.reconstruction.ge(self.threshold)
            mask2 = self.reconstruction.lt(self.threshold)
            reshaped_mask = mask.view(x_orig.shape[0], 32, 32).unsqueeze(1)
            reshaped_mask2 = mask2.view(x_orig.shape[0], 32, 32).unsqueeze(1)

            # Duplicate it 3 times along the third dimension
            final_mask = reshaped_mask.repeat(1, 3, 1, 1)
            final_mask2 = reshaped_mask2.repeat(1, 3, 1, 1)

            if self.complementry == 'noise':
                new_tensor_noise = (torch.rand(x_orig.shape[0], 3, 32, 32).to(torch.device("cuda" if torch.cuda.is_available() else "cpu")) * self.epsilon) + x_orig
                new_tensor_noise[new_tensor_noise > 1] = 1
                new_tensor_noise[new_tensor_noise < 0] = 0
                new_tensor = (x_orig * final_mask) + (new_tensor_noise * final_mask2)
            elif self.complementry == 'fixed':
                new_tensor = (x_orig * final_mask) + (torch.zeros(final_mask2.shape).to(torch.device("cuda" if torch.cuda.is_available() else "cpu")) * final_mask2)
            else:
                new_tensor = (x_orig * final_mask) + (self.new_tensor_attack * final_mask2)

            new_tensor = self.model_resnet(new_tensor)
            y2 = self.fc1(new_tensor)
            return y2


def train_model(learning_rate, gamma_value, epsilon_pgd, step_size_pgd, number_of_steps_pgd, threshold, batch_size, num_of_epochs, complementry, evaluate_adversarial, use_explainability):

    # Data preprocessing and augmentation
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor()
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    valid_size = 0.1

    # Load CIFAR-10 dataset:
    train_dataset = torchvision.datasets.CIFAR10(root='./sufficient_xai/data', train=True, transform=transform, download=True)
    val_dataset = torchvision.datasets.CIFAR10(root='./sufficient_xai/data', train=True, transform=val_transform, download=True)
    test_dataset = torchvision.datasets.CIFAR10(root='./sufficient_xai/data', train=False, transform=val_transform, download=True)

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, sampler = train_sampler, num_workers=4, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, sampler = valid_sampler, num_workers=4, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, num_workers=4, pin_memory=True)

    # Initialize the model, loss function, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net().to(device)
    model.threshold = threshold
    model.complementry = complementry
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.epsilon = epsilon_pgd
    min_val_loss = float('inf')
    count_val_loss = 0

    # Training loop
    for epoch in range(num_of_epochs):
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        correct_explanation = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            model.forward_status = 0
            outputs, explanation = model(images)
            if complementry=='attack':
                attack = PGD(model, eps=epsilon_pgd, alpha=step_size_pgd, steps=number_of_steps_pgd, random_start=True)
                adv_images = attack(images, labels, reconstruction = explanation, threshold = threshold)
                model.new_tensor_attack = adv_images
            model.forward_status = 1
            model.reconstruction = explanation
            outputs2 = model(images)

            cardinality_loss =  gamma_value * torch.norm(explanation, p=1) / (images.shape[0]*INPUT_SIZE)

            if use_explainability:
                loss = criterion(outputs, labels) + criterion(outputs2,torch.argmax(outputs, axis=1)) + cardinality_loss
            else:
                loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total_train += labels.size(0)
            _, predicted_explanation = outputs2.max(1)
            correct_explanation += predicted_explanation.eq(predicted).sum().item()
            correct_train += predicted.eq(labels).sum().item()

        train_accuracy = correct_train / total_train
        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        correct_val = 0
        correct_explanation_2 = 0
        total_val = 0
        total_regularization = 0
        counter = 0
        optimizer.zero_grad()

        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            model.forward_status = 0
            outputs, explanation = model(images)
            if complementry == 'attack':
                attack = PGD(model, eps=epsilon_pgd, alpha=step_size_pgd, steps=number_of_steps_pgd,
                                          random_start=True)
                adv_images = attack(images, labels, reconstruction = explanation, threshold = threshold)
                model.new_tensor_attack = adv_images

            model.forward_status = 1
            model.reconstruction = explanation
            outputs2 = model(images)

            num_positive = (explanation >= threshold).sum().item()
            total_positive = num_positive / explanation.numel()
            regularization_term = torch.tensor(total_positive)
            total_regularization += regularization_term

            cardinality_loss = gamma_value * torch.norm(explanation, p=1) / (images.shape[0] * INPUT_SIZE)

            if use_explainability:
                loss = criterion(outputs, labels) + criterion(outputs2,
                                                              torch.argmax(outputs, axis=1)) + cardinality_loss
            else:
                loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            _, predicted_explanation = outputs2.max(1)
            correct_explanation_2 += predicted_explanation.eq(predicted).sum().item()

            total_val += labels.size(0)
            correct_val += predicted.eq(labels).sum().item()
            counter += 1

        optimizer.zero_grad()

        val_accuracy = 100.0 * correct_val / total_val
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy_explanation = 100.0 * correct_explanation / total_train
        regularization_total = 100.0 * total_regularization / counter
        val_accuracy_explanation_2 = 100.0 * correct_explanation_2 / total_val

        print(f"Epoch [{epoch+1}/{num_of_epochs}] - "
              f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f} - "
              f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}"
              f" Val Acc explanation train: {val_accuracy_explanation:.4f}"
              f" Val Acc explanation validation: {val_accuracy_explanation_2:.4f}"
              f" Percentage sufficiency: {regularization_total:.4f}", flush=True)

        if val_loss <= min_val_loss:
            min_val_loss = val_loss
            torch.save(model.state_dict(), f'./sufficient_xai/methodological/trained_models/cifar10/cifar10_{complementry}_lr_{learning_rate}_g_{gamma_value}_e_{epsilon_pgd}')
            count_val_loss = 0
        else:
            count_val_loss += 1
        if count_val_loss == 300:
            break


    val_loss = 0.0
    correct_val = 0
    correct_explanation_2 = 0
    total_val = 0
    total_regularization = 0
    counter = 0
    optimizer.zero_grad()

    # with torch.no_grad():
    for images, labels in test_loader:
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
        total_positive = num_positive / explanation.numel()
        regularization_term = torch.tensor(total_positive)
        total_regularization += regularization_term

        cardinality_loss = gamma_value * torch.norm(explanation, p=1) / (images.shape[0] * INPUT_SIZE)

        if use_explainability:
            loss = criterion(outputs, labels) + criterion(outputs2,
                                                          torch.argmax(outputs, axis=1)) + cardinality_loss
        else:
            loss = criterion(outputs, labels)
        val_loss += loss.item()
        _, predicted = outputs.max(1)
        _, predicted_explanation = outputs2.max(1)
        correct_explanation_2 += predicted_explanation.eq(predicted).sum().item()

        total_val += labels.size(0)
        correct_val += predicted.eq(labels).sum().item()
        counter += 1

    optimizer.zero_grad()

    val_accuracy = correct_val / total_val
    regularization_total = 100.0 * total_regularization / counter
    val_accuracy_explanation = 100.0 * correct_explanation_2 / total_val

    print(f"Test Acc: {val_accuracy:.4f}"
          f" Test Acc explanation: {val_accuracy_explanation:.4f}"
          f" Percentage sufficiency: {regularization_total:.4f}", flush=True)


def run_main_code_cifar10(learning_rate, gamma_value, epsilon_pgd, step_size_pgd, number_of_steps_pgd, threshold, batch_size, num_of_epochs, complementry, evaluate_adversarial, use_explainability):
    train_model(learning_rate, gamma_value, epsilon_pgd, step_size_pgd, number_of_steps_pgd, threshold, batch_size, num_of_epochs, complementry, evaluate_adversarial, use_explainability)
