import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torch.optim import Adam
import time
from torch.utils.tensorboard import SummaryWriter


def load_data(data_path, download, num_workers, shuffle, batch_size):
    transform = transforms.Compose(
        [transforms.ToTensor(), 
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    trainset = torchvision.datasets.CIFAR10(root=data_path, 
                                            train=True, 
                                            download=download, 
                                            transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, 
                                              batch_size=batch_size, 
                                              shuffle=shuffle, 
                                              num_workers=num_workers)
    testset = torchvision.datasets.CIFAR10(root=data_path, 
                                           train=False, 
                                           download=download, 
                                           transform=transform)
    testloader = torch.utils.data.DataLoader(testset, 
                                             batch_size=batch_size, 
                                             shuffle=shuffle, 
                                             num_workers=num_workers)
    return trainloader, testloader

class ConvNet(nn.Module):
    def __init__(self, num_classes):
        super(ConvNet, self).__init__()
        
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(), 
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(), 
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.block3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(), 
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.block4 = nn.Sequential(
            nn.Linear(256 * 4 * 4, 256),
            nn.ReLU(), 
            nn.Linear(256, 32),
            nn.ReLU(), 
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out = out.reshape(out.size(0), -1)
        out = self.block4(out)
        return out

    
def train(model, criterion, optimizer, train_loader, test_loader, max_epochs, log_path):
    tb = SummaryWriter(log_path)
    for epoch in range(max_epochs):
        start_time = time.time()
        train_correct_preds = 0
        train_loss = 0
        for batch_num, (x_batch, y_batch) in enumerate(train_loader):
            x_batch = x_batch.cuda(non_blocking=True).float()
            y_batch = y_batch.cuda(non_blocking=True)
            output = model(x_batch)
            optimizer.zero_grad()
            loss = criterion(output, y_batch)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            _, pred = torch.max(output.data, 1)
            train_correct_preds += torch.sum(pred == y_batch.data).item()
        train_accuracy = float(train_correct_preds / len(train_loader.dataset))
        train_loss = train_loss / len(train_loader.dataset)
        model.eval()
        test_correct_preds = 0
        test_loss = 0
        for batch_num, (x_batch, y_batch) in enumerate(test_loader):
            x_batch = x_batch.cuda(non_blocking=True).float()
            y_batch = y_batch.cuda(non_blocking=True)
            output = model(x_batch)
            loss = criterion(output, y_batch)
            test_loss += loss.item()
            _, pred = torch.max(output.data, 1)
            test_correct_preds += torch.sum(pred == y_batch.data).item()
        test_accuracy = float(test_correct_preds / len(test_loader.dataset))
        test_loss = test_loss / len(test_loader.dataset)
        end_time = time.time()
        print(f"epoch : {epoch + 1} - train loss : {train_loss:.3f} - train accuracy : {(train_accuracy*100):.3f} - test loss : {test_loss:.3f} - test accuracy : {(test_accuracy*100):.3f} - time : {(end_time - start_time):.3f} s")
        tb.add_scalar("Train loss", train_loss, epoch + 1)
        tb.add_scalar("Train accuracy", train_accuracy, epoch + 1)
        tb.add_scalar("Test loss", test_loss, epoch + 1)
        tb.add_scalar("Test accuracy", test_accuracy, epoch + 1)

        
if __name__ == '__main__':
    data_path = "/home/rostami/DMLS/data/"
    download = False
    num_workers = 2
    shuffle = False
    batch_size = 256
    num_classes = 10
    gpu_number = 0
    max_epochs = 20
    log_path = "/home/rostami/DMLS/main/sample"
    optimizer_lr = 5e-4
    optimizer_weight_decay = 5e-4
    print("GPU : ", torch.cuda.get_device_name(gpu_number))
    model = ConvNet(num_classes)
    model.to(gpu_number)
    print(model)
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda(gpu_number)
    optimizer = Adam(model.parameters(), lr=optimizer_lr, weight_decay=optimizer_weight_decay)
    train_loader, test_loader = load_data(data_path, download, num_workers, shuffle, batch_size)
    start_time = time.time()
    train(model, criterion, optimizer, train_loader, test_loader, max_epochs, log_path)
    end_time = time.time()
    print(f"total time : {(end_time - start_time):.3f}")
    
    
