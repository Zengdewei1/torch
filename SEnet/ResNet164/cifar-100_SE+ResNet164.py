import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch import optim as optim
import visdom
import numpy as np
class BasicBlock(torch.nn.Module):
    expansion = 4
    def __init__(self,in_channels,channels,stride=1):
        super(BasicBlock,self).__init__()
        self.residual = torch.nn.Sequential(
            torch.nn.BatchNorm2d(in_channels),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels,channels,kernel_size=1,stride=1,padding=0,bias=False),
            torch.nn.Dropout(p=0.3),
            torch.nn.BatchNorm2d(channels),
            torch.nn.ReLU(),
            torch.nn.Conv2d(channels,channels,kernel_size=3,stride=stride,padding=1,bias=False),
            torch.nn.Dropout(p=0.3),
            torch.nn.BatchNorm2d(channels),
            torch.nn.ReLU(),
            torch.nn.Conv2d(channels,channels*self.expansion,kernel_size=1,stride=1,bias=False)
            )
        if in_channels != channels*self.expansion or stride != 1:
            self.shortCut = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels,channels*self.expansion,kernel_size=1,stride=stride,bias=False)
                )
        else:
            self.shortCut = torch.nn.Sequential()
        #SE layers
        self.fc1 = torch.nn.Conv2d(channels*self.expansion,channels*self.expansion//16,kernel_size=1)
        self.dropout = torch.nn.Dropout(p=0.3)
        self.fc2 = torch.nn.Conv2d(channels*self.expansion//16,channels*self.expansion,kernel_size=1)
    def forward(self,input):
        short_cut = self.shortCut(input)
        output = self.residual(input)
        #Squeeze
        w = torch.nn.functional.avg_pool2d(output,output.size(2))
        w = self.dropout(torch.nn.functional.relu(self.fc1(w)))
        w = torch.nn.functional.sigmoid(self.fc2(w))
        #Excitation
        output = output*w
        output += short_cut
        return output
class SE_ResNet(torch.nn.Module):
    def __init__(self,block,num_blocks,filter,num_classes = 100):
        super(SE_ResNet,self).__init__()
        self.in_channels = 16
        self.conv1 = torch.nn.Conv2d(3,16,kernel_size=3,stride=1,padding=1,bias=False)
        self.layer1 = self.make_layer(BasicBlock,num_blocks[0],filter[0],stride=1)
        self.layer2 = self.make_layer(BasicBlock,num_blocks[1],filter[1],stride=2)
        self.layer3 = self.make_layer(BasicBlock,num_blocks[2],filter[2],stride=2)
        self.bn = torch.nn.BatchNorm2d(filter[2]*block.expansion)
        self.fc = torch.nn.Linear(filter[2]*block.expansion,num_classes,bias=True)
    def make_layer(self,block,num_blocks,channels,stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels,channels,stride))
            self.in_channels = channels*block.expansion
        return torch.nn.Sequential(*layers)
    def forward(self,input):
        output = self.conv1(input)
        output = self.layer1(output)
        output = self.layer2(output)
        output = self.layer3(output)
        output = torch.nn.functional.relu(self.bn(output))
        output = torch.nn.functional.avg_pool2d(output,8)
        output = output.view(output.size(0),-1)
        output = self.fc(output)
        return output

def ResNet164():
    return SE_ResNet(BasicBlock,[18,18,18],[16,32,64],100)



	
def train(model, data, target, loss_func, optimizer):

    model.train()
    optimizer.zero_grad()
    output = model(data)
    predictions = output.max(1, keepdim=True)[1]
    correct = predictions.eq(target.view_as(predictions)).sum().item()
    acc = correct / len(target)
    loss = loss_func(output, target)
    loss.backward()
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
def adjust_learning_rate(optimizer,epoch,lr):
	learning_rate = lr*((0.1**int(epoch>=40))*(0.1**int(epoch>=60))*(0.5**int(epoch>=80)))
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr

def main():
    base_lr = 0.01
    num_classes = 100
    eval_step = 1000
    num_epochs = 160
    batch_size = 32
    resume = False #resume from check point
    ckpt_path = '/home/lianfei/model_ResNet164/best_ckpt'
    log_dir = '/home/lianfei/model_ResNet164'
    # model
    if resume:
    	print("==>Resuming from checkpoint..")
    	assert os.path.isdir(ckpt_path),'Error:checkpoint directory not exists!'
    	checkpoint = torch.load(os.path.join(ckpt_path,'ckpt.t7'))
    	model = checkpoint['model']
    	best_acc = checkpoint['best_acc']
    	start_epoch = checkpoint['epoch']
    else:
    	print("==>Building model..")
    	model = ResNet164()
    	start_epoch = 0
    use_cuda = torch.cuda.is_available()
    if use_cuda:
    	model = model.cuda()
    #Load file
    train_loader = DataLoader(
        datasets.CIFAR100(root='/home/lianfei/data/CIFAR-100', train=True, download=True,transform=
                         transforms.Compose([
                                            transforms.RandomCrop(32, padding=4),  #先四周填充0，在吧图像随机裁剪成32*32
                                            transforms.RandomHorizontalFlip(),  #图像一半的概率翻转，一半的概率不翻转
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])),
                                            batch_size=batch_size,
                                            shuffle=True
                                            )

    test_loader = DataLoader(
        datasets.CIFAR100(root='/home/lianfei/data/CIFAR-100', train=False,
                         transform=transforms.Compose([transforms.ToTensor(),
                         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])),
        batch_size=batch_size
    )

    # define loss function
    ce_loss = torch.nn.CrossEntropyLoss()

    # define optimizer
    optimizer = optim.SGD(model.parameters(), lr=base_lr,momentum=0.9, weight_decay=0.0002)

    # start train
    viz = visdom.Visdom()
    x,train_acc,test_acc = 0,0,0
    win = viz.line(
        X = np.array([x]),
        Y = np.column_stack((np.array([train_acc]),np.array([test_acc]))),
        opts = dict(
            title = "train ACC and test ACC",
            legend =["train_acc","test_acc"]
            )
        )
    train_step = 0
    best_acc = 0
    for epoch in range(num_epochs):
        print(epoch)
        adjust_learning_rate(optimizer,epoch+1,base_lr)
        for data, target in train_loader:
            train_step += 1
            if use_cuda:
                data = data.cuda()
                target = target.cuda()
            acc, loss = train(model, data, target, ce_loss, optimizer)
            train_acc = acc 
            if train_step % 100 == 0:
                print('Train set: Step: {}, Loss: {:.4f}, Accuracy: {:.2f}'.format(train_step, loss, acc))
            if train_step % eval_step == 0:
                acc, loss = test(model, test_loader, ce_loss, use_cuda)
                test_acc = acc
                print('\nTest set: Step: {}, Loss: {:.4f}, Accuracy: {:.2f}\n'.format(train_step, loss, acc))
                state = {
                	'model':model,
                	'best_acc':best_acc,
                	'epoch':epoch,
                }
                # save model
                if not os.path.isdir(os.path.join(log_dir,'last_ckpt')):
                	os.mkdir(os.path.join(log_dir,'last_ckpt'))
                torch.save(state,os.path.join(log_dir,'last_ckpt','ckpt.t7'))
                if acc > best_acc:
                	best_acc = acc
                	if not os.path.isdir(os.path.join(log_dir,'best_ckpt')):
                		os.mkdir(os.path.join(log_dir,'best_ckpt'))
                	torch.save(state,os.path.join(log_dir,'best_ckpt','ckpt.t7'))
            if train_step % 100 == 0:
                viz.line(
                    X = np.array([train_step]),
                    Y = np.column_stack(
                        (np.array([train_acc]),np.array([test_acc])
                            )
                        ),
                    win = win,
                    update = "append"
                    )

if __name__ == '__main__':
    main()
