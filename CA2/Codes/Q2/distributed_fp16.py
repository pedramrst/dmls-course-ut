import os
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torch.optim import Adam
import time
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
import torch.multiprocessing as mp
import datetime
from torch.nn.parallel import DistributedDataParallel as DDP


def load_data(data_path, download, shuffle, batch_size, world_size, rank):
    transform = transforms.Compose(
        [transforms.ToTensor(), 
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    trainset = torchvision.datasets.CIFAR10(root=data_path, 
                                            train=True, 
                                            download=download, 
                                            transform=transform)
    train_sampler = torch.utils.data.distributed.DistributedSampler(trainset, 
                                                    num_replicas=world_size
                                                    , rank=rank)
    trainloader = torch.utils.data.DataLoader(dataset=trainset,
                                              sampler=train_sampler, 
                                              batch_size=batch_size, 
                                              shuffle=shuffle, 
                                              num_workers=0, 
                                              pin_memory=True)
    
    testset = torchvision.datasets.CIFAR10(root=data_path, 
                                           train=False, 
                                           download=download, 
                                           transform=transform)
    test_sampler = torch.utils.data.distributed.DistributedSampler(testset, 
                                                    num_replicas=world_size
                                                    , rank=rank)
    testloader = torch.utils.data.DataLoader(dataset=testset, 
                                             sampler=test_sampler, 
                                             batch_size=batch_size, 
                                             shuffle=shuffle, 
                                             num_workers=0, 
                                             pin_memory=True)
    return trainloader, testloader


def setup(rank, world_size, master_port, backend, timeout):
    os.environ["MASTER_ADDR"] = 'localhost'
    os.environ["MASTER_PORT"] = master_port
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size, timeout=timeout)

def cleanup():
    dist.destroy_process_group()
    

def find_free_port():
    """ https://stackoverflow.com/questions/1365265/on-localhost-how-do-i-pick-a-free-port-number """
    import socket
    from contextlib import closing

    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return str(s.getsockname()[1])


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

    
def train(rank, data_path, download, shuffle, batch_size, max_epochs, log_path, world_size, master_port, backend, timeout, num_classes, optimizer_lr, optimizer_weight_decay):
    setup(rank, world_size, master_port, backend, timeout)
    train_loader, test_loader = load_data(data_path, download, shuffle, batch_size, world_size, rank)
    torch.cuda.set_device(rank)
    model = ConvNet(num_classes).to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda(rank)
    optimizer = Adam(ddp_model.parameters(), lr=optimizer_lr, weight_decay=optimizer_weight_decay)
    tb = SummaryWriter(f"{log_path}/{rank}")
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(max_epochs):
        start_time = time.time()
        train_correct_preds = 0
        train_loss = 0
        number_of_batches = 0
        if epoch == 5:
            print("Memory usage: ", torch.cuda.memory_allocated())
        for batch_num, (x_batch, y_batch) in enumerate(train_loader):
            x_batch = x_batch.cuda(non_blocking=True).float()
            y_batch = y_batch.cuda(non_blocking=True)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                output = ddp_model(x_batch)
                loss = criterion(output, y_batch)
            train_loss += loss.item()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            _, pred = torch.max(output.data, 1)
            train_correct_preds += torch.sum(pred == y_batch.data).item()
            number_of_batches += 1
        train_accuracy = float(train_correct_preds / (number_of_batches * batch_size))
        train_loss = train_loss / (number_of_batches * batch_size)
        ddp_model.eval()
        test_correct_preds = 0
        test_loss = 0
        number_of_batches = 0
        for batch_num, (x_batch, y_batch) in enumerate(test_loader):
            x_batch = x_batch.cuda(non_blocking=True).float()
            y_batch = y_batch.cuda(non_blocking=True)
            output = ddp_model(x_batch)
            loss = criterion(output, y_batch)
            test_loss += loss.item()
            _, pred = torch.max(output.data, 1)
            test_correct_preds += torch.sum(pred == y_batch.data).item()
            number_of_batches += 1
        test_accuracy = float(test_correct_preds / (number_of_batches * batch_size))
        test_loss = test_loss / (number_of_batches * batch_size)
        end_time = time.time()
        print(f"rank : {rank} - epoch : {epoch + 1} - train loss : {train_loss:.3f} - train accuracy : {(train_accuracy*100):.3f} - test loss : {test_loss:.3f} - test accuracy : {(test_accuracy*100):.3f} - time : {(end_time - start_time):.3f} s")
        tb.add_scalar("Train loss", train_loss, epoch + 1)
        tb.add_scalar("Train accuracy", train_accuracy*100, epoch + 1)
        tb.add_scalar("Test loss", test_loss, epoch + 1)
        tb.add_scalar("Test accuracy", test_accuracy*100, epoch + 1)

        
if __name__ == '__main__':
    data_path = "/home/rostami/DMLS/data/"
    download = False
    shuffle = False
    batch_size = 256
    num_classes = 10
    gpu_number = 0
    max_epochs = 20
    log_path = "/home/rostami/DMLS/main/distributed_fp16"
    optimizer_lr = 5e-4
    optimizer_weight_decay = 5e-4
    backend = 'gloo'
    timeout = datetime.timedelta(seconds=10)
    world_size = torch.cuda.device_count()
    master_port = find_free_port()
    
    
    start_time = time.time()
    mp.spawn(train, nprocs=world_size, args=(data_path, download, shuffle, batch_size, max_epochs, log_path, world_size, master_port, backend, timeout, num_classes, optimizer_lr, optimizer_weight_decay))
    end_time = time.time()
    print(f"total time : {(end_time - start_time):.3f}")
    
    
