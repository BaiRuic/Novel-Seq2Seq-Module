# Enhenced Seq2Seq Basic block
# input:x: shape [batch_size, time_steps, features] 在Basic Block的forward 已经进行了 x = x.permute(0, 2,1)处理了
# output: predict: [batch_size, pred_seqlen, 1]
#         estimate: [batch_size, input_seqlen, 1]

import torch
import torch.nn as nn
from .encoder import TcnEncoder
from .decoder import GruDecoder


class Seq2Seq_TCN(nn.Module):
    def __init__(self, input_size, encoder_num_channels, forecast_seqlen, kernel_size=4):
        """
        input_size: 输出序列数据的样本特征
        encoder_num_channels: [List[int]]编码器的每个残差块的卷积维度
        predice_seqlen: 预测的时间步
        kernel_size: 卷积核的大大小
        """
        super(Seq2Seq_TCN, self).__init__()
        self.input_size = input_size
        self.encoder_num_channels = encoder_num_channels
        self.hidden_size = encoder_num_channels[-1]
        self.kernel_size = kernel_size

        self.forecast_seqlen = forecast_seqlen

        self.encoder = TcnEncoder(num_inputs=self.input_size, num_channels=self.encoder_num_channels,
                                  kernel_size=self.kernel_size)
        self.f_decoder = GruDecoder(input_size=1, hidden_size=self.hidden_size, output_size=1)

    def forward(self, inputs):
        """
        inputs: [batch_size, seq_len, features]
        """
        # from [batch_size, seq_len, features]  to   [batch_size, features, seq_len]
        inputs = inputs.permute(0, 2, 1)
        batch_size = inputs.shape[0]

        # 对输入进行编码 得到编码状态
        hidden = self.encoder(inputs)
        hidden.contiguous()
        assert (hidden.shape == (1, batch_size, self.hidden_size))

        # 解码器输入
        f_decoder_input = inputs[:, 0, -1].reshape(batch_size, 1, 1)  # for forecast decoder
        assert (f_decoder_input.shape == (batch_size, 1, 1))

        # 存放解码器的输出列表
        forecast_outputs = []

        # 初始化 解码器状态
        f_hidden = hidden  # for forecast decoder

        # 做预测
        for _ in range(self.forecast_seqlen):
            out, f_hidden = self.f_decoder(f_decoder_input, f_hidden)
            forecast_outputs.append(out)
            f_decoder_input = out

        # 将列表  forecast_outputs 转换为tensor
        forecast_outputs = torch.cat(forecast_outputs, dim=2)  # [batch_size, 1, seq_len=estimate_seqlen]


        # from [batch_size, features=1, seq_len]  to  [batch_size, seq_len, features=1]
        forecast_outputs = forecast_outputs.permute(0, 2, 1)

        return forecast_outputs


if __name__ == "__main__":

    batch_size = 128
    time_step = 24
    features = 3
    num_channels = [4, 6, 8, 12]
    # 测试输入 (batch_size, time_step, features) 是否可行
    x = torch.rand(size=(batch_size, time_step, features))
    model = Seq2Seq_TCN(input_size=features, encoder_num_channels=num_channels, forecast_seqlen=6, estimate_seqlen=24)

    p = model(x)

    print(p.shape)
