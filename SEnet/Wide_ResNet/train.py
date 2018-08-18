import torch
import config
import torchvision
import torch.backends.cudnn as cudnn
import wideResNet

import os
import sys
import argparse

parser = argparse.ArgumentParser(description='Pytorch CIFAR training on WideResNet')
parser.add_argument("--lr",default=0.1,type=float,help='basic learning rate')
parser.add_argument("--depth",default=28,type=int,help='depth of model')
parser.add_argument("--widen_factor",default=10,type=int,help="width of model")
parser.add_argument("--dropout_rate",default=0.3,type=float,help='dropout_rate')
parser.add_argument("--resume","-r",default=True,help="Resume from checkPoint")
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
best_acc = 0.0
start_epoch,num_epochs,batch_size,optim = config.start_epoch,config.num_epochs,config.batch_size,config.optim
print(args.resume)
#Load Data
print("Data Preparation......")
transform_train = torchvision.transforms.Compose([
	torchvision.transforms.RandomCrop(32,padding=4),
	torchvision.transforms.RandomHorizontalFlip(),
	torchvision.transforms.ToTensor(),
	torchvision.transforms.Normalize(config.mean_cifar100,config.std_cifar100)]
	)
transform_test = torchvision.transforms.Compose([
	torchvision.transforms.ToTensor(),
	torchvision.transforms.Normalize(config.mean_cifar100,config.std_cifar100)]
	)
train_data = torchvision.datasets.CIFAR100(root='/home/lianfei/data/CIFAR-100',train=True,download=True,transform=transform_train)
test_data = torchvision.datasets.CIFAR100(root='/home/lianfei/data/CIFAR-100',train=False,download=True,transform=transform_test)
train_loader = torch.utils.data.DataLoader(train_data,batch_size=batch_size,shuffle=True,num_workers=1)
test_loader = torch.utils.data.DataLoader(test_data,batch_size=32,shuffle=False,num_workers=1)

# Model
if args.resume:
	print("Resuming from checkPoint....")
	assert os.path.isdir("checkpoint"),"Error: No checkPoint directory found!"
	checkPoint = torch.load('./checkpoint/'+'Wide_ResNet.t7')
	net = checkPoint['net']
	best_acc = checkPoint['acc']
	start_epoch = checkPoint['epoch']
else:
	print("Building net...")
	net = wideResNet.Wide_ResNet(args.depth,args.widen_factor,args.dropout_rate,100)

if use_cuda:
	net.cuda()
	net = torch.nn.DataParallel(net,device_ids=range(torch.cuda.device_count()))
	cudnn.benchmark = True
criterion = torch.nn.CrossEntropyLoss()

#Training
def train(epoch):
	net.train()
	train_loss = 0
	correct = 0
	total = 0
	optimizer = torch.optim.SGD(net.parameters(),lr=config.adjust_learning_rate(args.lr,epoch),momentum=0.9,weight_decay=5e-4)
	for batch_id,(inputs,targets) in enumerate(train_loader):
		if use_cuda:
			inputs,targets = inputs.cuda(),targets.cuda()
		optimizer.zero_grad()
		inputs,targets = torch.autograd.Variable(inputs),torch.autograd.Variable(targets)
		outputs = net(inputs)
		loss = criterion(outputs,targets)
		loss.backward()
		optimizer.step()

		train_loss += loss.data[0]
		_,predicted = torch.max(outputs.data,1)
		total += targets.size(0)
		correct += predicted.eq(targets.data).cpu().sum()
		print('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f Acc@1: %.3f%%'
                %(epoch, num_epochs, batch_id+1,
                    (len(train_data)//batch_size)+1, loss.data[0], 100.*correct/total))
def test(epoch):
	global best_acc
	net.eval()
	test_loss = 0
	correct = 0
	total = 0
	for batch_idx,(inputs,targets) in enumerate(test_loader):
		if use_cuda:
			inputs,targets = inputs.cuda(),targets.cuda()
		inputs,targets = torch.autograd.Variable(inputs),torch.autograd.Variable(targets)
		outputs = net(inputs)
		loss = criterion(outputs,targets)

		test_loss += loss.data[0]
		_,predicted = torch.max(outputs.data,1)
		total += targets.size(0)
		correct += predicted.eq(targets.data).cpu().sum()
		acc = 100.*correct/total
	print("\n| Validation Epoch #%d\t\t\tLoss: %.4f Acc@1: %.2f%%" %(epoch, loss.data[0], acc))
	if acc > best_acc:
		print("saving model....")
		state = {
			'net':net.module if use_cuda else net,
			'acc':acc,
			'epoch':epoch
			}
		if not os.path.isdir('checkpoint'):
			os.mkdir("checkpoint")
		save_point = './checkpoint/Wide_ResNet.t7'
		torch.save(state,save_point)				
		best_acc = acc
print("Training model...")
print("Training Epochs = " + str(num_epochs))
print("Initial Learning Rate = " + str(args.lr))
print("Optimizer = " + str(optim))

for epoch in range(start_epoch,start_epoch+num_epochs):
	train(epoch)
	test(epoch)
print("Training Modle...")
print("Test Results: ACC = %.2f"%(best_acc))