import torch
import torch.nn as nn
import torch.nn.functional as F


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class Inception(nn.Module):
	def __init__(
		self,
		in_channels,
		ch1x1,
		ch3x3red,
		ch3x3,
		ch5x5red,
		ch5x5,
		pooling
		):
		super(Inception, self).__init__()

		# 1x1 conv branch
		self.branch1 = nn.Sequential(
			nn.Conv3d(in_channels, ch1x1, kernel_size=(1, 1, 1), bias=False),
			nn.BatchNorm3d(ch1x1),
			nn.ELU()
			)

		# 1x1 conv + 3x3 conv branch
		self.branch2 = nn.Sequential(
			nn.Conv3d(in_channels, ch3x3red, kernel_size=(1, 1, 1), bias=False),
			nn.BatchNorm3d(ch3x3red),
			nn.ELU(),
			nn.Conv3d(ch3x3red, ch3x3, kernel_size=(3, 3, 3), padding=(1, 1, 1), bias=False), 
			nn.BatchNorm3d(ch3x3),
			nn.ELU()
			)

		# 1x1 conv + 5x5 conv branch
		self.branch3 = nn.Sequential(
			nn.Conv3d(in_channels, ch5x5red, kernel_size=(1, 1, 1), bias=False),
		 	nn.BatchNorm3d(ch5x5red),
		 	nn.ELU(),
		 	nn.Conv3d(ch5x5red, ch5x5, kernel_size=(5, 5, 5), padding=(2, 2, 2), bias=False), 
		 	nn.BatchNorm3d(ch5x5),
		 	nn.ELU()
		 	)

		# 3x3 pool + 1x1 conv branch
		self.branch4 = nn.Sequential(
			nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), ceil_mode=True),
		  	nn.Conv3d(in_channels, pooling, kernel_size=(1, 1, 1), bias=False),
	        nn.BatchNorm3d(pooling),
	        nn.ELU()
		  	)

	def forward(self, x):
		branch1 = self.branch1(x)
		#print(branch1.shape)
		branch2 = self.branch2(x)
		#print(branch2.shape)
		branch3 = self.branch3(x)
		#print(branch3.shape)
		branch4 = self.branch4(x)
		#print(branch4.shape)
		return torch.cat([branch1, branch2, branch3, branch4], 1)

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d((1,1,1))
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)

class MainNet(nn.Module):
    def __init__(self, num_classes):
        super(MainNet, self).__init__()

		#conv layers before inception
        self.pre_inception = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=(7, 7, 7), stride=(1, 2, 2), padding=(3, 3, 3)),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(2, 2, 2), ceil_mode=True),
            nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size=(1, 1, 1)),
            nn.ReLU(),
            nn.Conv3d(64, 192, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), ceil_mode=True),
            nn.ReLU()
            )

        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)


        self.conv_1 = conv1x1x1(192, 256, stride=1)
        self.conv_2 = conv1x1x1(256, 256, stride=1)

        self.sigmoid = nn.Sigmoid()
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
		# self.dropout = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pre_inception(x)
        res = self.conv_1(x)
        out = self.inception3a(x)
        out += res
        out = self.conv_2(out)
        out = self.avgpool(out)
        out = out.view(x.size(0), -1)

        out = self.fc3(out)
        out = self.sigmoid(out)

        return out

def generate_model(classes, **kwargs):
    model = MainNet(num_classes=classes, **kwargs)
    return model