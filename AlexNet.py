import torch

class AlexNet(torch.nn.Module):
	def __init__(self):
		super(AlexNet,self).__init__()
		self.conv1a = torch.nn.Sequential(
			torch.nn.Conv2d(3,128,kernel_size=5,stride=1,padding=0),
			torch.nn.BatchNorm2d(128),
			torch.nn.ReLU(),                                           #input:3x32x32 output:128x16x16
			torch.nn.MaxPool2d(kernel_size=3,stride=2,padding=0),
			torch.nn.LocalResponseNorm(2),
			)
		self.conv1b = torch.nn.Sequential(
			torch.nn.Conv2d(3,128,kernel_size=5,stride=1,padding=0),
			torch.nn.BatchNorm2d(128),
			torch.nn.ReLU(),
			torch.nn.MaxPool2d(kernel_size=3,stride=2,padding=0)
			)
		self.conv2a = torch.nn.Sequential(
			torch.nn.Conv2d(128,192,kernel_size=3,stride=1,padding=0),#input:128x16x16 output:192x16x16
			torch.nn.BatchNorm2d(192),
			torch.nn.ReLU()
			)
		self.conv2b = torch.nn.Sequential(
			torch.nn.Conv2d(128,192,kernel_size=3,stride=1,padding=0),
			torch.nn.BatchNorm2d(192),
			torch.nn.ReLU()
			)
		self.conv3a = torch.nn.Sequential(
			torch.nn.Conv2d(192,192,kernel_size=3,stride=1,padding=0),#input:192x16x16 output:192x16x16
			torch.nn.BatchNorm2d(192),
			torch.nn.ReLU()
			)
		self.conv3b = torch.nn.Sequential(
			torch.nn.Conv2d(192,192,kernel_size=3,stride=1,padding=0),
			torch.nn.BatchNorm2d(192),
			torch.nn.ReLU(),
			)
		self.conv4a = torch.nn.Sequential(
			torch.nn.Conv2d(192,128,kernel_size=3,stride=1,padding=0),#input:192x16x16 output:128x8x8
			torch.nn.BatchNorm2d(128),
			torch.nn.ReLU(),
			torch.nn.MaxPool2d(kernel_size=3,stride=2,padding=0),
			)
		self.conv4b = torch.nn.Sequential(
			torch.nn.Conv2d(192,128,kernel_size=3,stride=1,padding=0),
			torch.nn.BatchNorm2d(128),
			torch.nn.ReLU(),
			torch.nn.MaxPool2d(kernel_size=3,stride=2,padding=0)
			)
		self.fc1a = torch.nn.Linear(1152,2048)
		self.fc1b = torch.nn.Linear(1152,2048)
		self.dropout1a = torch.nn.Dropout(p=0.5)
		self.dropout1b = torch.nn.Dropout(p=0.5)
		self.fc2a = torch.nn.Linear(2048,2048)
		self.fc2b = torch.nn.Linear(2048,2048)
		self.dropout2a = torch.nn.Dropout(p=0.5)
		self.dropout2b = torch.nn.Dropout(p=0.5)
		self.fc3 = torch.nn.Linear(4096,10)
	def forward(self,input):
		outputa = self.conv1a(input)
		outputa = self.conv2a(outputa)
		outputa = self.conv3a(outputa)
		outputa = self.conv4a(outputa)
		outputb = self.conv1b(input)
		outputb = self.conv2b(outputb)
		outputb = self.conv3b(outputb)
		outputb = self.conv4b(outputb)
		outputa = outputa.view(outputa.size(0),-1)
		outputb = outputb.view(outputb.size(0),-1)
		outputa = self.fc1a(outputa)
		outputa = self.fc2a(outputa)
		outputb = self.fc1b(outputb)
		outputb = self.fc2b(outputb)
		output = torch.cat((outputa,outputb),1)
		output = self.fc3(output)
		return output