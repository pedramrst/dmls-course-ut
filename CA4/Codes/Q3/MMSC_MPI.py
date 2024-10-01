import torchvision
import torchvision.transforms as transforms
import torch
import argparse
import torch.distributed as dist
import torch.nn as nn
import time
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam
import os

def get_data_loader(rank, args):
    train_dataset = torchvision.datasets.MNIST(root=args.data_root,
                                                train=True,
                                                transform=transforms.ToTensor(),
                                                download=False)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,
                                                                   num_replicas=args.world_size,
                                                                   rank=rank)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                batch_size=args.train_batch_size,
                                                shuffle=args.shuffle,
                                                num_workers=args.num_workers,
                                                pin_memory=True,
                                                sampler=train_sampler)
    
    test_dataset = torchvision.datasets.MNIST(root=args.data_root,
                                                train=False,
                                                transform=transforms.ToTensor(),
                                                download=False)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset,
                                                                   num_replicas=args.world_size,
                                                                   rank=rank)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                batch_size=args.test_batch_size,
                                                shuffle=args.shuffle,
                                                num_workers=args.num_workers,
                                                pin_memory=True,
                                                sampler=test_sampler)
    
    return train_loader, test_loader

class ConvNet(nn.Module):
    def __init__(self, num_classes, args):
        super(ConvNet, self).__init__()
        self.quantized = args.quantized
        if self.quantized:
            self.quant = torch.quantization.QuantStub()
            self.dequant = torch.quantization.DeQuantStub()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.max1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.max2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.linear = nn.Linear(8 * 7 * 7, num_classes)

    def forward(self, x):
        if self.quantized:
            x = x.contiguous(memory_format=torch.channels_last)
            x = self.quant(x)
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.max1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.max2(out)
        out = out.reshape(out.size(0), -1)
        out = self.linear(out)
        if self.quantized:
            out = self.dequant(out)
        return out

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train(ddp_model, train_loader, test_loader, optimizer, criterion, rank, args):
    train_accuracies = []
    train_losses = []
    test_accuracies = []
    test_losses = []
    for epoch in range(args.epochs):
        start_time = time.time()
        ## training phase
        train_correct_preds, train_loss, n_batches = 0, 0, 0
        for batch_num, (x_batch, y_batch) in enumerate(train_loader):
            optimizer.zero_grad()
            x_batch = x_batch.float()
            output = ddp_model(x_batch)
            loss = criterion(output, y_batch)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            _, preds = torch.max(output.data, 1)
            train_correct_preds += torch.sum(preds == y_batch.data).item()
            n_batches += 1
            if (batch_num + 1) % 100 == 0:
                train_accuracies.append(float(train_correct_preds / (100 * args.train_batch_size)))
                train_losses.append(float(train_loss / (100 * args.train_batch_size)))     
                train_correct_preds, train_loss = 0, 0
                print(f"{batch_num + 1} / {len(train_loader)} - time: {(time.time() - start_time):.3f}")
        ## eval phase
        ddp_model.eval()
        if args.quantized:
            eval_model = torch.quantization.convert(ddp_model)
        else:
            eval_model = ddp_model
        test_correct_preds, test_loss, n_batches = 0, 0, 0
        for batch_num, (x_batch, y_batch) in enumerate(test_loader):
            x_batch = x_batch.float()
            output = eval_model(x_batch)
            loss = criterion(output, y_batch)
            test_loss += loss.item()
            _, preds = torch.max(output.data, 1)
            test_correct_preds += torch.sum(preds == y_batch.data).item()
            n_batches += 1
        test_accuracies.append(float(test_correct_preds / (n_batches * args.test_batch_size)))
        test_losses.append(float(test_loss / (n_batches * args.test_batch_size)))
            ## logging phase
        end_time = time.time()
        print(f"rank: {rank} - epoch: {epoch + 1} - time: {(end_time - start_time):.3f}")
    with open(f"{args.log_path}/{rank}/train.acc", 'w') as f:
        f.write('\n'.join(map(str, train_accuracies)))
    with open(f"{args.log_path}/{rank}/train.loss", 'w') as f:
        f.write('\n'.join(map(str, train_losses)))
    with open(f"{args.log_path}/{rank}/test.acc", 'w') as f:
        f.write('\n'.join(map(str, test_accuracies)))
    with open(f"{args.log_path}/{rank}/test.loss", 'w') as f:
        f.write('\n'.join(map(str, test_losses)))

def init_model(args):
    torch.backends.quantized.engine = None
    torch.manual_seed(args.seed)
    model = ConvNet(args.num_classes, args)
    num_params = count_parameters(model)
    if args.quantized:
        torch.backends.quantized.engine = 'qnnpack'
        model.eval()
        model.qconfig = torch.quantization.get_default_qat_qconfig('qnnpack')
        modules_to_fuse = [['conv1', 'relu1'], ['conv2', 'relu2']]
        model = torch.quantization.fuse_modules(model, modules_to_fuse)
        model = torch.quantization.prepare_qat(model.train())
    ddp_model = DDP(model)
    return model, ddp_model, num_params

def init_criterion():
    criterion = nn.CrossEntropyLoss()
    return criterion

def init_optimizer(ddp_model, args):
    optimizer = Adam(ddp_model.parameters(), lr=args.optimizer_lr, 
                     weight_decay=args.optimizer_lr_decay)
    return optimizer

def init_data_loaders(rank, args):
    train_loader, test_loader = get_data_loader(rank, args)
    return train_loader, test_loader

def init_log_dir(rank, args):
    os.system(f"mkdir -p {args.log_path}/{rank}")


def main():
    ## parsing args phase
    parser = argparse.ArgumentParser(description='Distributed MNIST Classification')
    parser.add_argument('--train-batch-size', type=int, default=32, metavar='N', 
                        help='data batch size for training - default: 32')
    parser.add_argument('--test-batch-size', type=int, default=128, metavar='N', 
                        help='data batch size for testing - default: 128')
    parser.add_argument('--data-root', type=str, default='/home/rostami/pytorch_ddp/dataset/data', metavar='/path/to/mnist', 
                        help='MINST dataset path - default: /home/rostami/pytorch_ddp/dataset/data')
    parser.add_argument('--shuffle', default=False, action='store_true',
                        help='flag to shuffle - default: not flag')
    parser.add_argument('--num-workers', type=int, default=0, metavar='N',
                        help='data loader number of workers - default: 0')
    parser.add_argument('--world-size', type=int, default=1, metavar='N',
                        help='number of processes - default: 1')
    parser.add_argument('--epochs', type=int, default=3, metavar='N',
                        help='number of epochs - default: 3')
    parser.add_argument('--optimizer-lr', type=float, default=5e-4, metavar='a.aaaa',
                        help='Adam init learning rate - default: 5e-4')
    parser.add_argument('--optimizer-lr-decay', type=float, default=5e-4, metavar='a.aaaa',
                        help='Adam learning rate decay - default: 5e-4')
    parser.add_argument('--num-classes', type=int, default=10, metavar='N',
                        help='number of MNIST classes - default: 10')
    parser.add_argument('--log-path', type=str, default='/home/rostami/pytorch_ddp/tb_logs', metavar='/path/to/log/path', 
                        help='Tensorboard log path - default: /home/rostami/pytorch_ddp/tb_logs')
    parser.add_argument('--seed', type=int, default=123, metavar='N',
                        help='torch seed for instantiating model weights - default: 123')
    parser.add_argument('--quantized', default=False, action='store_true', 
                        help='flag to train quantized model - default: not flag')
    parser.add_argument('--backend', type=str, default='gloo', metavar='gloo/mpi', 
                        help='distributed communication backend - default: gloo')
    args = parser.parse_args()
    
    ## init phase
    dist.init_process_group(backend=args.backend)
    rank = dist.get_rank()
    model, ddp_model, model_params_count = init_model(args)
    print(f'Rank {rank} started')
    if rank == 0:
        print(f"model params: {model_params_count}")
    criterion = init_criterion()
    optimizer = init_optimizer(ddp_model, args)
    train_loader, test_loader = init_data_loaders(rank, args)
    init_log_dir(rank, args)
    ## training phase
    train(ddp_model, train_loader, test_loader, optimizer, criterion, rank, args)
    
if __name__ == '__main__':
    main()