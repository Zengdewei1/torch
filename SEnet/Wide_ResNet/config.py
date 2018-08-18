import math

start_epoch = 1
num_epochs = 200
batch_size = 32
optim = 'SGD'

mean_cifar100 = (0.5071,0.4867,0.4408)

std_cifar100 = (0.2675,0.2565,0.2761)

def adjust_learning_rate(lr,epoch):
	optim_factor = 0
	if(epoch > 160):
		optim_factor = 3
	elif(epoch > 120):
		optim_factor =2
	elif(epoch > 60):
		optim_factor=1
	return lr*math.pow(0.2,optim_factor)
	