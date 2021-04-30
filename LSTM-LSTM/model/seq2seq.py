import torch
import torch.nn as nn
from typing import List

Vector = List[int]
DEVICE = torch.device('cuda' if torch.cuda.is_available() else "cpu")


class LstmEncoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        '''
        input_size:    the number of features in the input X
        hidden_size:   the number of features in the hidden state h
        '''
        super(LstmEncoder, self).__init__()

        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=1,
                            bias=True,
                            batch_first=True)

    def forward(self, inputs):
        '''
        param:
            inputs [batch_size, time_steps, features]
                    function： 作为编码器的输入
        return:
            hidden [1, batch_size, hidden_size]
                    function: 作为编码器的输出，只包含LSTM的最后状态
            cell: [1, batch_size, hidden_size]
        '''
        output, (hidden, cell) = self.lstm(inputs)
        return hidden, cell


class LstmDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1):
        super(LstmDecoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lstm = nn.LSTM(input_size=self.input_size,
                            hidden_size=self.hidden_size,
                            num_layers=1,
                            bias=True,
                            batch_first=True)
        self.fc = nn.Linear(in_features=self.hidden_size, out_features=self.output_size, bias=True)

    def forward(self, inputs, prev_hidden, prev_cell):
        '''
        params:
            inputs: [batch_size, seq_len=1, features=1]
            prev_hidden [1, batch_size, hidden_size]
            prev_cell [1, batch_size, hidden_size]
        returns:
            prediction: [batc_size, seq_len=1, feature=1]
            hidden: [1, batch_size, hidden_size]
            cell: [1, batch_size, hidden_size]
        '''
        # output [batch_size, seq_len=1, features=hidden_size]
        # hidden [1, batch_size, hidden_size]
        # cell [1, batch_size, hidden_size]
        output, (hidden, cell) = self.lstm(inputs, (prev_hidden, prev_cell))
        # prediction [batch_size, seq_len=1, output_size=1]
        prediction = self.fc(output.squeeze(dim=1)).unsqueeze(dim=1)
        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, input_size, hidden_size, input_seqlen, forecast_seqlen):
        '''
        input_size: 输出序列数据的样本特征
        hidden_size: 隐藏层样本特征
        predice_seqlen: 预测的时间步
        estimate_seqlen: 估计的时间步，即输入样本的时间步
        '''
        super(Seq2Seq, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.input_seqlen = input_seqlen
        self.forecast_seqlen = forecast_seqlen

        self.encoder = LstmEncoder(input_size=self.input_size, hidden_size=self.hidden_size)
        self.decoder = LstmDecoder(input_size=1, hidden_size=self.hidden_size, output_size=1)

    def forward(self, inputs):
        batch_size = inputs.shape[0]
        # 对输入进行编码 得到编码状态
        hidden, cell = self.encoder(inputs)

        assert (hidden.shape == (1, batch_size, self.hidden_size))
        assert (cell.shape == (1, batch_size, self.hidden_size))

        # 解码器输入 (最后一个时间步 第0维特征)
        decoder_input = inputs[:, -1, 0].reshape(batch_size, 1, 1)  # for forecast decoder
        assert (decoder_input.shape == (batch_size, 1, 1))

        # 存放 解码器的输出列表
        forecast_outputs = []

        # 做预测
        for _ in range(self.forecast_seqlen):
            out, hidden, cell = self.decoder(decoder_input, hidden, cell)
            forecast_outputs.append(out)
            decoder_input = out

        # 将列表forecast_outputs 转换为tensor
        forecast_outputs = torch.cat(forecast_outputs, dim=1)  # [batch_size, seq_len=estimate_seqlen, 1]

        # 提取输入数据中的 需要预测变量数据 如负载预测中，提取负载数据。剔除温度数据
        input_main = inputs[:, :, 0].unsqueeze(dim=2)  # [batch_size, seq_len, features=1]
        return forecast_outputs


if __name__ == '__main__':
    batch_size = 128
    input_seqlen = 24
    forecast_seqlen = 5
    features = 3
    x = torch.rand(size=(batch_size, input_seqlen, features)).to(DEVICE)

    model = Seq2Seq(input_size=features, hidden_size=6, input_seqlen=input_seqlen,
                   forecast_seqlen=forecast_seqlen).to(DEVICE)
    forecast = model(x)
    print(forecast.shape)