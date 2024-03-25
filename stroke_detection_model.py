from torch import nn

class StrokeClassifierCNN(nn.Module):
   def __init__(self):
       super(StrokeClassifierCNN, self).__init__()
       self.conv_block_1 = nn.Sequential(
           nn.Conv2d(1, 32, kernel_size=3, padding=1),
           nn.BatchNorm2d(32),
           nn.ReLU(inplace=True),
           nn.Conv2d(32, 64, kernel_size=3, padding=1),
           nn.BatchNorm2d(64),
           nn.ReLU(inplace=True),
           nn.MaxPool2d(kernel_size=2, stride=2),
       )

       self.conv_block_2 = nn.Sequential(
           nn.Conv2d(64, 64, kernel_size=3, padding=1),
           nn.BatchNorm2d(64),
           nn.ReLU(inplace=True),
           nn.Conv2d(64, 128, kernel_size=3, padding=1),
           nn.BatchNorm2d(128),
           nn.ReLU(inplace=True),
           nn.MaxPool2d(kernel_size=2, stride=2),
       )

       self.classifier = nn.Sequential(
           nn.Flatten(),
           nn.Linear(128*56*56, 128),
           nn.ReLU(inplace=True),
           nn.Dropout(p=0.2),
           nn.Linear(128, 1)
       )

   def forward(self, x):
       # x = self.conv_block_1(x)
       # print(x.shape)
       # x = self.conv_block_2(x)
       # print(x.shape)
       # x = self.classifier(x)
       # print(x.shape)
       # return x
       return self.classifier(self.conv_block_2(self.conv_block_1(x)))