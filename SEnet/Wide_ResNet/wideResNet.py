import torch
import sys
import numpy

def conv3x3(in_planes,out_planes,stride=1):
	return torch.nn.Conv2d(in_planes,out_planes,kernel_size=3,stride=stride,padding=1,bias=True)

class wide_basic(torch.nn.Module):
	def __init__(self,in_planes,planes,dropout_rate,stride=1):
		super(wide_basic,self).__init__()
		self.bn1 = torch.nn.BatchNorm2d(in_planes)
		self.conv1 = torch.nn.Conv2d(in_planes,planes,kernel_size=3,padding=1,bias=True)
		self.dropout = torch.nn.Dropout(p=dropout_rate)
		self.bn2 = torch.nn.BatchNorm2d(planes)
		self.conv2d = torch.nn.Conv2d(planes,planes,kernel_size=3,stride=stride,padding=1,bias=True)

		self.shortCut = torch.nn.Sequential()
		if stride != 1 or in_planes != planes:
			self.shortCut = torch.nn.Sequential(
				torch.nn.Conv2d(in_planes,planes,kernel_size=1,stride=stride,bias=True)
				)
	def forward(self,x):
		output = self.dropout(self.conv1(torch.nn.functional.relu(self.bn1(x))))
		output = self.conv2(torch.nn.functional.relu(self.bn2(output)))
		out += self.shortCut(x)

class Wide_ResNet(torch.nn.Module):
	def __init__(self,depth,widen_factor,dropout_rate,num_classes=100):
		super(Wide_ResNet,self).__init__()
		self.in_planes = 16

		assert ((depth-4)%6 == 0),'Wide-ResNet depth should be 6n+4'
		n = (depth-4)/6
		k = widen_factor

		print("Wide_ResNet %dx%d"%(depth,k))
		n_Stages = [16,16*k,32*k,64*k]

		self.conv1 = conv3x3(3,n_Stages[0])
		self.layer1 = self.make_layer(wide_basic,n_Stages[1],n,dropout_rate,stride=1)
		self.layer2 = self.make_layer(wide_basic,n_Stages[2],n,dropout_rate,stride=2)
		self.layer3 = self.make_layer(wide_basic,n_Stages[3],n,dropout_rate,stride=2)
		self.bn1 = torch.nn.BatchNorm2d(n_Stages[3],momentum=0.9)
		self.fc = torch.nn.Linear(n_Stages[3],num_classes)

	def make_layer(self,block,planes,num_blocks,dropout_rate,stride):
		strides = [stride] + [1]*(num_blocks-1)
		layers = []

		for stride in strides:
			layers.append(block(self.in_planes,planes,dropout_rate,stride))
			self.in_planes = planes
		return torch.nn.Sequential(*layers)
	def forward(self,x):
		output = self.conv1(x)
		output = self.layer1(output)
		output = self.layer2(output)
		output = self.layer3(output)
		output = torch.nn.functional.relu(self.bn(out))
		output = torch.nn.avg_pool2d(out,8)
		output = output.view(out.size(0),-1)
		output = self.fc(output)

		return output

def Wide_ResNet28x10():
	return Wide_ResNet(28,10,0.3,10)


