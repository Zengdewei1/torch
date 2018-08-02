import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch import optim as optim
class BasicBlock(torch.nn.Module):
	def __init__(self,in_channel,out_channels,stride=1):
		super(BasicBlock,self).__init__()
		self.mainLayer = torch.nn.Sequential(
			torch.nn.Conv2d(in_channel,out_channels,kernel_size=3,stride=stride,padding=1,bias=False),
			torch.nn.BatchNorm2d(out_channels),
			torch.nn.ReLU(inplace=True),
			torch.nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding=1,bias=False),
			torch.nn.BatchNorm2d(out_channels))
		self.shortCut = torch.nn.Sequential()
		if stride != 1 or in_channel != out_channels:
			self.shortCut = torch.nn.Sequential(
				torch.nn.Conv2d(in_channel,out_channels,kernel_size=1,stride=stride,bias=False),
				torch.nn.BatchNorm2d(out_channels))
	def forward(self,input):
		out = self.mainLayer(input)
		out += self.shortCut(input)
		out = torch.nn.functional.relu(out)
		return out
class RESNet(torch.nn.Module):
	def __init__(self,BasicBlock):
		super(RESNet,self).__init__()
		self.in_channel = 64
		self.conv1 = torch.nn.Sequential(
			torch.nn.Conv2d(3,64,kernel_size=3,stride=1,padding=1,bias=False),
			torch.nn.BatchNorm2d(64),
			torch.nn.ReLU())
		self.layer1 = self.make_layer(BasicBlock,64,2,stride=1)
		self.layer2 = self.make_layer(BasicBlock,128,2,stride=2)
		self.layer3 = self.make_layer(BasicBlock,256,2,stride=2)
		self.layer4 = self.make_layer(BasicBlock,512,2,stride=2)
		self.fc = torch.nn.Linear(512,10)
	def make_layer(self,block,channels,num_blocks,stride):
		strides = [stride] + [1]*(num_blocks-1)
		layers = []
		for stride in strides:
			layers.append(block(self.in_channel,channels,stride))
			self.in_channel = channels
		return torch.nn.Sequential(*layers)
	def forward(self,input):
		output = self.conv1(input)
		output = self.layer1(output)
		output = self.layer2(output)
		output = self.layer3(output)
		output = self.layer4(output)
		output = torch.nn.functional.avg_pool2d(output,4)
		output = output.view(output.size(0),-1)
		output = self.fc(output)
		return output
def RESNet18():
	return RESNet(BasicBlock)
	
def train(model, data, target, loss_func, optimizer):

    model.train()
    # initial optimizer
    optimizer.zero_grad()

    # net work will do forward computation defined in net's [forward] function
    output = model(data)

    # get predictions from outputs, the highest score's index in a vector is predict class
    predictions = output.max(1, keepdim=True)[1]

    # cal correct predictions num
    correct = predictions.eq(target.view_as(predictions)).sum().item()

    # cal accuracy
    acc = correct / len(target)

    # use loss func to cal loss
    loss = loss_func(output, target)

    # backward will back propagate loss
    loss.backward()

    # this will update all weights use the loss we just back propagate
    optimizer.step()

    return acc, loss


def test(model, test_loader, loss_func, use_cuda):

    model.eval()
    acc_all = 0
    loss_all = 0
    step = 0
    with torch.no_grad():
        for data, target in test_loader:
            step += 1
            if use_cuda:
                data = data.cuda()
                target = target.cuda()
            output = model(data)
            predictions = output.max(1, keepdim=True)[1]
            correct = predictions.eq(target.view_as(predictions)).sum().item()
            acc = correct / len(target)
            loss = loss_func(output, target)
            acc_all += acc
            loss_all += loss
    return acc_all / step, loss_all / step


def main():
    """
    main function
    """

    # define some hyper parameters
    num_classes = 10
    eval_step = 1000
    num_epochs = 100
    batch_size = 64
    model_name = 'resnet' # resnet, vgg

    # first check directories, if not exist, create
    dir_list = ('../data', '../data/MNIST', '../data/CIFAR-10')
    for directory in dir_list:
        if not os.path.exists(directory):
            os.mkdir(directory)

    # if cuda available -> use cuda
    use_cuda = torch.cuda.is_available()
    # this step will create train_loader and  test_loader

    train_loader = DataLoader(
        datasets.CIFAR10(root='../data/CIFAR-10', train=True, download=True,transform=
                         transforms.Compose([
                                            transforms.RandomCrop(32, padding=4),  #先四周填充0，在吧图像随机裁剪成32*32
                                            transforms.RandomHorizontalFlip(),  #图像一半的概率翻转，一半的概率不翻转
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])),
                                            batch_size=batch_size,
                                            shuffle=True
                                            )

    test_loader = DataLoader(
        datasets.CIFAR10(root='../data/CIFAR-10', train=False,
                         transform=transforms.Compose([transforms.ToTensor(),
                         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])),
        batch_size=batch_size
    )

    # define network
    model = RESNet18()
    if use_cuda:
        model = model.cuda()
    # define loss function
    ce_loss = torch.nn.CrossEntropyLoss()

    # define optimizer
    optimizer = optim.SGD(model.parameters(), lr=1e-2,momentum=0.9, weight_decay=5e-4)

    # start train
    train_step = 0
    for _ in range(num_epochs):
        for data, target in train_loader:
            train_step += 1
            if use_cuda:
                data = data.cuda()
                target = target.cuda()
            acc, loss = train(model, data, target, ce_loss, optimizer)
            if train_step % 100 == 0:
                print('Train set: Step: {}, Loss: {:.4f}, Accuracy: {:.2f}'.format(train_step, loss, acc))
            if train_step % eval_step == 0:
                acc, loss = test(model, test_loader, ce_loss, use_cuda)
                print('\nTest set: Step: {}, Loss: {:.4f}, Accuracy: {:.2f}\n'.format(train_step, loss, acc))


if __name__ == '__main__':
    main()
