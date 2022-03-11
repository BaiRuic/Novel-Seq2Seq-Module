# BasicBlock Encoder,即TCN的残差块
# 输入: x shape:(batch_size, features, timesteps)
# 输出 y [1, batch_size, features] 做为上下文向量
# 参数建议：num_inputs = 3, num_channels = [4, 6, 8, 12], kernel_size=4, dropout=0.5
# 一维卷积是在 (batch_size, features, time_step) 的 time_step 维度上进行卷积的

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        '''裁剪多出来的时间步
        param:
            x shape: [batch_size, features, seq_len]
        return :
            shape: [batch_size, features, seq_len-self.chomp_size]
        '''

        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        '''
        note: 一个标准的残差块，每个残差块中只有两层卷积，且每层卷积的膨胀因子是相同的，
              因此，如果想要构建膨胀因子呈指数增长的编码器，就需要多个残差块串联起来
        :param n_inputs: 一维卷积输入维度，就是每个卷积核的通道数
        :param n_outputs: 一维卷积输出维度，就是卷积核的个数
        :param kernel_size: 卷积核大小
        :param stride: 卷积步长 为 1
        :param dilation: 膨胀因子,在每个残差块中的两层卷积中，膨胀因子是相同的
        :param padding:为了保证残差块输入输出相同，padding = dilation * (kernel_size-1)
        :param dropout
        '''
        super(TemporalBlock, self).__init__()
        # first part
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        # second part
        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        # 1*1 的卷积
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TcnEncoder(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=4, dropout=0.5):
        '''
        note:编码器，每个编码器里面有len(num_channels)个残差块，每个残差块里面有两层相同的卷积+dropout结构
            len(num_channels)个残差块中卷积核的dilation_size 分别为[1,2,4,8,... ]
        param:
            num_inputs: [int] 输入时间序列的特征维度
            num_channels: [List] 编码器里面的残差块中的卷积层的特征数
            kernel_size: [int] 卷积核的大小
        '''

        super(TcnEncoder, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(n_inputs=in_channels,
                                     n_outputs=out_channels,
                                     kernel_size=kernel_size,
                                     stride=1,
                                     dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size,
                                     dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        '''
        param:
            x: the shape of x [batch, input_channel, seq_len]
        return:
            shape:[1, batch, output_channel], as a context vector into decoder
        '''
        # output shape:[batch, output_channel, seq_len]
        output = self.network(x)
        return output[:, :, -1].unsqueeze(dim=0).contiguous()


if __name__ == "__main__":

    batch_size = 128
    time_step = 24
    features = 3
    # x = torch.rand(size =(batch_size, features, time_step) )
    # TCN的输入维度为(batch_size, features, timesteps)
    x = torch.rand(size=(batch_size, time_step, features))
    x = x.permute(0, 2, 1)
    print(f"x.shape:{x.shape}")

    # 测试 Chomp1d
    model = Chomp1d(3)
    y = model(x)
    print(f"the shape is {y.shape} after chomp1d")

    # 测试TemporalBlock
    model = TemporalBlock(n_inputs=features, n_outputs=3, kernel_size=3, stride=1, dilation=1, padding=2, dropout=0.5)
    y = model(x)
    print(f"the shape is {y.shape} after TemporalBlock")

    # 测试 TcnBlock

    model11 = TcnEncoder(num_inputs=features,
                       num_channels=[4, 6, 8, 12],
                       kernel_size=3,
                       dropout=0.5)

    y = model11(x)
    print((y.shape))




