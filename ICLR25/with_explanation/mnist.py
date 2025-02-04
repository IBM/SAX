import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import numpy as np
from attack_files.pgd_file import PGD


TRAIN_TEST_SPLIT = 0.9
INPUT_SIZE = 784
INPUT_DIMENSION = 255
INPUT_SINGLE_DIMENSION = 28

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.x_adversarial= None
        self.orig_reconstruction = None
        self.forward_status = 0
        self.new_tensor_attack = None
        self.put_random_noise = True
        self.reconstruction = None
        self.threshold = None
        self.epsilon = None
        self.complementry = None

        self.fc1 = nn.Linear(784, 128)
        self.fco = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 10)
        self.fc3 = nn.Linear(64, 784)

    def forward(self, x_orig):

        if self.forward_status == 0:
            x_orig = x_orig.view(-1, INPUT_SIZE)
            x = torch.relu(self.fc1(x_orig))
            x = torch.relu(self.fco(x))
            digit_output = self.fc2(x)
            reconstruction = torch.sigmoid(self.fc3(x))
            return reconstruction, digit_output

        if self.forward_status == 1:
            x_orig = x_orig.view(-1, INPUT_SIZE)
            mask = self.reconstruction.ge(self.threshold)
            mask2 = self.reconstruction.lt(self.threshold)

            if self.complementry == 'noise':
                new_tensor_noise = torch.rand(INPUT_SIZE) * self.epsilon + x_orig
                new_tensor_noise[new_tensor_noise > 1] = 1
                new_tensor_noise[new_tensor_noise < 0] = 0
                new_x = (x_orig * mask) + (new_tensor_noise * mask2)
            elif self.complementry == 'attack':
                new_x = (x_orig * mask) + (self.new_tensor_attack.view(-1, INPUT_SIZE) * mask2)
            else:
                new_x = (x_orig * mask)

            new_x = torch.relu(self.fc1(new_x))
            new_x = torch.relu(self.fco(new_x))
            digit_output_b = self.fc2(new_x)
            reconstruction_b = torch.sigmoid(self.fc3(new_x))
            return reconstruction_b, digit_output_b

def train_model(learning_rate, gamma_value, epsilon_pgd, step_size_pgd, number_of_steps_pgd, threshold, batch_size, num_of_epochs, complementry, evaluate_adversarial, use_explainability):

    # Load and preprocess MNIST dataset:
    transform = transforms.Compose([transforms.ToTensor()])
    mnist_dataset = torchvision.datasets.MNIST(root='./sufficient_xai/data', train=True, transform=transform, download=True)

    # Split dataset into training, validation, and test sets:
    train_size = int(TRAIN_TEST_SPLIT * len(mnist_dataset))
    val_size = len(mnist_dataset) - train_size
    train_dataset, val_dataset = random_split(mnist_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) # was 64 beore
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) # also

    # Initialize the network, loss function, and optimizer:
    net = SimpleNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    net.threshold = threshold
    net.epsilon = epsilon_pgd
    net.complementry = complementry

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

            optimizer.zero_grad()
            net.forward_status = 0
            reconstruction, digit_output = net(inputs)

            if complementry=='attack':
                attack = PGD(net, eps=epsilon_pgd, alpha=step_size_pgd, steps=number_of_steps_pgd, random_start=True)
                adv_images = attack(inputs, labels, reconstruction = reconstruction, threshold= threshold)
                net.new_tensor_attack = adv_images
            
            net.forward_status = 1
            net.reconstruction = reconstruction
            _, digit_output_b = net(inputs)

            cardinality_loss =  gamma_value * torch.norm(reconstruction, p=1) / (inputs.shape[0]*INPUT_SIZE)

            if use_explainability:
                loss = criterion(digit_output, labels) + criterion(digit_output_b, torch.argmax(digit_output, axis=1)) + cardinality_loss
            else:
                loss = criterion(digit_output, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = digit_output.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_accuracy = 100.0 * correct / total
        train_loss = train_loss / total
        net.eval()
        val_loss = 0.0
        correct_val = 0
        correct_explanation = 0
        total_regularization = 0
        total_val = 0
        counter = 0

        optimizer.zero_grad()
        for batch_idx, (inputs, labels) in enumerate(val_loader):

            optimizer.zero_grad()

            net.forward_status = 0
            reconstruction, digit_output = net(inputs)

            if complementry == 'attack':
                attack = PGD(net, eps=epsilon_pgd, alpha=step_size_pgd, steps=number_of_steps_pgd,
                                          random_start=True)
                adv_images = attack(inputs, labels, reconstruction=reconstruction,
                    threshold=threshold)
                net.new_tensor_attack = adv_images
            net.forward_status = 1
            net.reconstruction = reconstruction
            _, digit_output_b = net(inputs)

            cardinality_loss = gamma_value * torch.norm(reconstruction, p=1) / (inputs.shape[0] * INPUT_SIZE)
            if use_explainability:
                loss = criterion(digit_output, labels) + criterion(digit_output_b, torch.argmax(digit_output,
                                                                                            axis=1)) + cardinality_loss
            else:
                loss = criterion(digit_output, labels)

            val_loss += loss.item()
            _, predicted = digit_output.max(1)
            _, predicted_explanation = digit_output_b.max(1)
            correct_explanation += predicted_explanation.eq(predicted).sum().item()
            num_positive = (reconstruction >= threshold).sum().item()
            total_positive = num_positive / reconstruction.numel()
            regularization_term = torch.tensor(total_positive)
            total_regularization += regularization_term

            total_val += labels.size(0)
            correct_val += predicted.eq(labels).sum().item()
            counter+=1

        optimizer.zero_grad()

        val_accuracy = 100.0 * correct_val / total_val
        val_loss = val_loss / total_val
        val_accuracy_explanation = 100.0 * correct_explanation / total_val
        regularization_total = 100.0 * total_regularization / counter

        print(f"Epoch [{epoch+1}/{num_epochs}]"
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%"
              f", Val Acc explanation: {val_accuracy_explanation:.2f}%"
              f", Percentage sufficient: {regularization_total:.5f}%",flush=True)

        if val_loss <= min_val_loss:
            min_val_loss = val_loss
            torch.save(net.state_dict(), f'./sufficient_xai/methodological/trained_models/mnist/mnist_{complementry}_lr_{learning_rate}_g_{gamma_value}_e_{epsilon_pgd}')
            count_val_loss = 0
        else:
            count_val_loss += 1
        if count_val_loss == 300:
            break

    # Calculate test accuracy:
    test_loader = DataLoader(torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True),
                             batch_size=batch_size, shuffle=False)

    net.eval()
    correct = 0
    correct_explanation = 0

    total_size_of_sufficient_set = 0
    total = 0

    optimizer.zero_grad()
    for inputs, labels in test_loader:

        optimizer.zero_grad()

        net.forward_status = 0
        reconstruction, digit_output = net(inputs)

        if complementry == 'attack':
            attack = PGD(net, eps=epsilon_pgd, alpha=step_size_pgd, steps=number_of_steps_pgd,
                                      random_start=True)
            adv_images = attack(inputs, labels, reconstruction=reconstruction, threshold=threshold)
            net.new_tensor_attack = adv_images
        
        net.forward_status = 1
        net.reconstruction = reconstruction
        reconstruction_b, digit_output_b = net(inputs)

        input_array = np.array(reconstruction.detach().numpy())
        size_of_sufficient_set = np.sum(input_array > threshold)
        total_size_of_sufficient_set += size_of_sufficient_set/INPUT_SIZE

        _, predicted = digit_output.max(1)
        _, predicted_explanation = digit_output_b.max(1)

        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        correct_explanation += predicted_explanation.eq(predicted).sum().item()

    test_accuracy = 100.0 * correct / total
    print(f"Test Accuracy: {test_accuracy:.2f}%",flush=True)

    test_accuracy_explanation = 100.0 * correct_explanation / total
    print(f"Test Accuracy explanation: {test_accuracy_explanation:.2f}%",flush=True)

    average_size_of_sufficient_set = 100.0 * total_size_of_sufficient_set/total
    print(f"Average size of sufficient set: {average_size_of_sufficient_set:.2f}%",flush=True)


def run_main_code_mnist(learning_rate, gamma_value, epsilon_pgd, step_size_pgd, number_of_steps_pgd, threshold, batch_size, num_of_epochs, complementry, evaluate_adversarial, use_explainability):
    train_model(learning_rate, gamma_value, epsilon_pgd, step_size_pgd, number_of_steps_pgd, threshold, batch_size, num_of_epochs, complementry, evaluate_adversarial, use_explainability)
