import torch
class BasicBlock(torch.nn.Module):
	expansion = 4
	def __init__(self,in_channels,channels,stride=1):
		super(BasicBlock,self).__init__()
		self.residual = torch.nn.Sequential(
			torch.nn.BatchNorm2d(in_channels),
			torch.nn.ReLU(),
			torch.nn.Conv2d(in_channels,channels,kernel_size=1,stride=1,padding=0,bias=False),
			torch.nn.BatchNorm2d(channels),
			torch.nn.ReLU(),
			torch.nn.Conv2d(channels,channels,kernel_size=3,stride=stride,padding=1,bias=False),
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
	def forward(self,input):
		short_cut = self.shortCut(input)
		output = self.residual(input)
		output += short_cut
		return output
class ResNet(torch.nn.Module):
	def __init__(self,block,num_blocks,filter,num_classes = 100):
		super(ResNet,self).__init__()
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
	return ResNet(BasicBlock,[18,18,18],[16,32,64],100)



