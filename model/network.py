import torch.nn as nn
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, ksize=3,padding_mode='zeros'):
        super(ResBlock, self).__init__()
        self.actfun1 = nn.LeakyReLU(0.2, True)
        self.actfun2 = nn.LeakyReLU(0.2, True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=ksize, padding=(ksize // 2),padding_mode=padding_mode)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=ksize, padding=(ksize // 2),padding_mode=padding_mode)

    def forward(self, input):
        x = self.conv1(input)
        x = self.actfun1(x)

        x = self.conv2(x)       
        x = self.actfun2(x)

        output = input+x
        return output

class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, ksize=3,padding_mode='zeros'):
        super(Conv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=ksize, padding=(ksize // 2),padding_mode=padding_mode)
        self.actfun1 = nn.LeakyReLU(0.2, True)
    def forward(self, x):
        x = self.conv1(x)
        x = self.actfun1(x)
        return x 

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)

class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Conv2d(n_feat, 2*n_feat, 2, 2)

    def forward(self, x):
        return self.body(x)