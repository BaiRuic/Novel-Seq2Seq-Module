# BasicBlock Encoder
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
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        '''
        param:
            num_inputs: [int] 输入时间序列的特征维度
            num_channels: [List] 编码器里面的卷积层的特征数
            kernel_size: [int] 卷积核的大小
        '''
        super(TcnEncoder, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        '''
        param:
            x: the shape of x [batch, input_channel, seq_len]
        return:
            shape:[1, batch, output_channel]
        '''
        # output shape:[batch, output_channel, seq_len]
        output = self.network(x)
        return output[:, :, -1].unsqueeze(dim=0).contiguous()


if __name__ == "__main__":

    batch_size = 128
    time_step = 14
    features = 2
    # x = torch.rand(size =(batch_size, features, time_step) )
    x = torch.rand(size=(batch_size, time_step, features))
    x = x.permute(0, 2, 1)

    model = TcnEncoder(num_inputs=features,
                       num_channels=[4, 6, 5],
                       kernel_size=3,
                       dropout=0.5)

    y = model(x)
    print((y.shape))




