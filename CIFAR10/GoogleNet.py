import torch
import torchvision

class Inception(torch.nn.Module):
	def __init__(self,in_channels,out_channels_1x1,out_channels_r3x3,out_channels_3x3,out_channels_r5x5,out_channels_5x5,out_channels_pool):
		super(Inception,self).__init__()
		self.conv1x1 = torch.nn.Sequential(
			torch.nn.Conv2d(in_channels,out_channels_1x1,kernel_size=1,stride=1),
			torch.nn.BatchNorm2d(out_channels_1x1),
			torch.nn.ReLU(True))
		self.conv3x3 = torch.nn.Sequential(
			torch.nn.Conv2d(in_channels,out_channels_r3x3,kernel_size=1,stride=1),
			torch.nn.BatchNorm2d(out_channels_r3x3),
			torch.nn.ReLU(),
			torch.nn.Conv2d(out_channels_r3x3,out_channels_3x3,kernel_size=3,stride=1,padding=1),
			torch.nn.BatchNorm2d(out_channels_3x3),
			torch.nn.ReLU())
		self.conv5x5 = torch.nn.Sequential(
			torch.nn.Conv2d(in_channels,out_channels_r5x5,kernel_size=1,stride=1),
			torch.nn.BatchNorm2d(out_channels_r5x5),
			torch.nn.ReLU(),
			torch.nn.Conv2d(out_channels_r5x5,out_channels_5x5,kernel_size=3,stride=1,padding=1),
			torch.nn.BatchNorm2d(out_channels_5x5),
			torch.nn.ReLU())
		self.convmc = torch.nn.Sequential(
			torch.nn.MaxPool2d(kernel_size=3,stride=1,padding=1),
			torch.nn.Conv2d(in_channels,out_channels_pool,kernel_size=1,stride=1),
			torch.nn.BatchNorm2d(out_channels_pool),
			torch.nn.ReLU())
	def forward(self,input):
		output1 = self.conv1x1(input)
		output2 = self.conv3x3(input)
		output3 = self.conv5x5(input)
		output4 = self.convmc(input)
		output = torch.cat((output1,output2,output3,output4),1)
		return output
class GoogleNet(torch.nn.Module):
	def __init__(self):
		super(GoogleNet,self).__init__()
		#input:batchSizex3x32x32,output:batchSizex192x16x16
		self.basicLayer = torch.nn.Sequential(
			torch.nn.Conv2d(3,192,kernel_size=3,stride=1,padding=1),
			torch.nn.ReLU(),
			#torch.nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
			)
		self.Inception3a = self.makeLayer(192,64,96,128,16,32,32)
		self.Inception3b = self.makeLayer(256,128,128,192,32,96,64)
		self.maxpool1 = torch.nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
		self.Inception4a = self.makeLayer(480,192,96,208,16,48,64)
		self.Inception4b = self.makeLayer(512,160,112,224,24,64,64)
		self.Inception4c = self.makeLayer(512,128,128,256,24,64,64)
		self.Inception4d = self.makeLayer(512,112,144,288,32,64,64)
		self.Inception4e = self.makeLayer(528,256,160,320,32,128,128)
		self.maxpool2 = torch.nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
		self.Inception5a = self.makeLayer(832,256,160,320,32,128,128)
		self.Inception5b = self.makeLayer(832,384,192,384,48,128,128)
		self.avgpool = torch.nn.AvgPool2d(kernel_size=8,stride=1)
		self.droupout = torch.nn.Dropout(p=0.4)
		self.fc = torch.nn.Linear(1024,10)
		#self.softmax = torch.nn.Softmax()

	def makeLayer(self,in_channels,out_channels_1x1,out_channels_r3x3,out_channels_3x3,out_channels_r5x5,out_channels_5x5,out_channels_pool):
		return Inception(in_channels,out_channels_1x1,out_channels_r3x3,out_channels_3x3,out_channels_r5x5,out_channels_5x5,out_channels_pool)
	def forward(self,input):
		output = self.basicLayer(input)
		output = self.Inception3a(output)
		output = self.Inception3b(output)
		output = self.maxpool1(output)
		output = self.Inception4a(output)
		output = self.Inception4b(output)
		output = self.Inception4c(output)
		output = self.Inception4d(output)
		output = self.Inception4e(output)
		output = self.maxpool2(output)
		output = self.Inception5a(output)
		output = self.Inception5b(output)
		output = self.avgpool(output)
		output = self.droupout(output)
		output = output.view(output.size(0),-1)
		output = self.fc(output)
		return output
