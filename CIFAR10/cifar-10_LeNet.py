import os
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch import optim as optim
from torch import nn
import math
import torch.nn.functional as F
class LENet(torch.nn.Module):
    def __init__(self):
        super(LENet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5) 
        self.conv2 = nn.Conv2d(6, 16, 5)  
        self.fc1   = nn.Linear(16*5*5, 120)  
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)
        self.pool1 = torch.nn.MaxPool2d(2,stride=2)
        self.pool2 = torch.nn.MaxPool2d(2,stride=2)

    def forward(self, x): 
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x) 
        x = x.view(x.size()[0], -1) 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)        
        return x
def train(args, model, device, train_loader, optimizer, epoch):
	criterion = torch.nn.CrossEntropyLoss()#loss funciton
	model.train()
	for batch_id,(data,label) in enumerate(train_loader):
		data,label = data.to(device),label.to(device)
		optimizer.zero_grad()
		output = model(data)
		loss = criterion(output,label)
		loss.backward()
		optimizer.step()
		if batch_id % args.log_interval == 0:
			print("Train epoch: %d, loss: %f"%(epoch,loss.item()))
def test(args, model, device, test_loader):
	criterion = torch.nn.CrossEntropyLoss()
	model.eval()
	test_loss = 0.0
	correct = 0
	with torch.no_grad():
		for data,label in test_loader:
			data,label = data.to(device),label.to(device)
			output = model(data)
			test_loss += criterion(output,label).item()
			pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
			correct += pred.eq(label.view_as(pred)).sum().item()
	print("Test set, total loss: %f,correct %d / %d"%(test_loss,correct,len(test_loader.dataset)))

def weight_init(m):
    if isinstance(m, torch.nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
    elif isinstance(m, torch.nn.BatchNorm2d):
        m.weigth.data.fill_(1)
        m.bias.data.zero_()
def main():
	parser = argparse.ArgumentParser(description="Cifar-10")

	parser.add_argument("--batch_size",type=int,default=100,metavar="N",
		help="input batch_size for training")
	parser.add_argument("--test_batch_size",type=int,default=1000,metavar="N",
		help="input batch_size for testing")
	parser.add_argument("--epochs",type=int,default=50,metavar="N",
		help="number for epochs in the train")
	parser.add_argument("--lr",type=float,default=0.01,metavar="LR",
		help='learning rate (default: 0.01)')
	parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
		help='SGD momentum (default: 0.5)')
	parser.add_argument('--no_cuda', action='store_true', default=False,
		help='disables CUDA training')
	parser.add_argument('--seed', type=int, default=1, metavar='S',
		help='random seed (default: 1)')
	parser.add_argument('--log_interval', type=int, default=10, metavar='N',
		help='how many batches to wait before logging training status')
	args = parser.parse_args()
	use_cuda = not args.no_cuda and torch.cuda.is_available()
	torch.manual_seed(args.seed)
	device = torch.device("cuda" if use_cuda else "cpu")
	kwargs = {"num_workers":1,"pin_memory":True} if use_cuda else {}
	train_loader = DataLoader(
        datasets.CIFAR10(root='../data/CIFAR-10', train=True, download=True,transform=
                         transforms.Compose([
                                            transforms.RandomCrop(32, padding=4),  #先四周填充0，在吧图像随机裁剪成32*32
                                            transforms.RandomHorizontalFlip(),  #图像一半的概率翻转，一半的概率不翻转
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])),
                                            batch_size=args.batch_size,
                                            shuffle=True
                                            )

	test_loader = DataLoader(
        datasets.CIFAR10(root='../data/CIFAR-10', train=False,
                         transform=transforms.Compose([transforms.ToTensor(),
                         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])),
        batch_size=args.test_batch_size
    )
	model = LENet().to(device)
	model.apply(weight_init)
	optimizer = torch.optim.SGD(model.parameters(),lr=args.lr,momentum=0.9)
	for epoch in range(1,args.epochs+1):
		train(args,model,device,train_loader,optimizer,epoch)
		test(args,model,device,test_loader)

if __name__ == "__main__":
	main()