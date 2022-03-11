# Enhenced Seq2Seq Basic block
# input:x: shape [batch_size, time_steps, features] 在Basic Block的forward 已经进行了 x = x.permute(0, 2,1)处理了
# output: predict: [batch_size, pred_seqlen, 1]
#         estimate: [batch_size, input_seqlen, 1]

import torch
import torch.nn as nn
from .encoder  import TcnEncoder
from .decoder import GruDecoder


class BasicBlock(nn.Module):
    def __init__(self, input_size, encoder_num_channels, forecast_seqlen, estimate_seqlen, kernel_size=5):
        """
        input_size: 输出序列数据的样本特征
        encoder_num_channels: [List[int]]编码器的每个残差块的卷积维度
        predice_seqlen: 预测的时间步
        estimate_seqlen: 估计的时间步，即输入样本的时间步
        """
        super(BasicBlock, self).__init__()
        self.input_size = input_size
        self.encoder_num_channels = encoder_num_channels
        self.hidden_size = encoder_num_channels[-1]
        self.kernel_size = kernel_size

        self.forecast_seqlen = forecast_seqlen
        self.estimate_seqlen = estimate_seqlen

        self.encoder = TcnEncoder(num_inputs=self.input_size, num_channels=self.encoder_num_channels,
                                  kernel_size=self.kernel_size)
        self.f_decoder = GruDecoder(input_size=1, hidden_size=self.hidden_size, output_size=1)
        self.e_decoder = GruDecoder(input_size=1, hidden_size=self.hidden_size, output_size=1)

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
        e_decoder_input = inputs[:, 0, 0].reshape(batch_size, 1, 1)  # for estimate decoder   这里感觉时间步变成第一步比较好

        assert (f_decoder_input.shape == (batch_size, 1, 1))
        assert (e_decoder_input.shape == (batch_size, 1, 1))

        # 存放 两个解码器的输出列表
        forecast_outputs = []
        estimate_outputs = []

        # 初始化 解码器状态
        f_hidden = hidden  # for forecast decoder
        e_hidden = hidden  # for estimate decoder

        # 做预测
        for _ in range(self.forecast_seqlen):
            out, f_hidden = self.f_decoder(f_decoder_input, f_hidden)
            forecast_outputs.append(out)
            f_decoder_input = out

        # 做估计
        for _ in range(self.estimate_seqlen):
            out, e_hidden = self.e_decoder(e_decoder_input, e_hidden)
            estimate_outputs.append(out)
            e_decoder_input = out

        # 将列表 estimate_outputs 和 forecast_outputs 转换为tensor
        estimate_outputs = torch.cat(estimate_outputs, dim=2)  # [batch_size, 1, seq_len=estimate_seqlen]
        forecast_outputs = torch.cat(forecast_outputs, dim=2)  # [batch_size, 1, seq_len=estimate_seqlen]

        # 提取输入数据中的 需要预测变量数据 如负载预测中，提取负载数据。剔除温度数据
        input_main = inputs[:, 0, :].unsqueeze(dim=1)  # [batch_size, features=1, seq_len]
        assert (input_main.shape == estimate_outputs.shape)

        # from [batch_size, features=1, seq_len]  to  [batch_size, seq_len, features=1]
        forecast_outputs = forecast_outputs.permute(0, 2, 1)
        input_main = input_main.permute(0, 2, 1)
        estimate_outputs = estimate_outputs.permute(0, 2, 1)
        return forecast_outputs, input_main - estimate_outputs


if __name__ == "__main__":

    batch_size = 128
    time_step = 24
    features = 3
    num_channels = [4, 6, 8, 12]
    # 测试输入 (batch_size, time_step, features) 是否可行
    x = torch.rand(size=(batch_size, time_step, features))
    model = BasicBlock(input_size=features, encoder_num_channels=num_channels, forecast_seqlen=6, estimate_seqlen=24)
    model_ = BasicBlock(input_size=1, encoder_num_channels=num_channels, forecast_seqlen=6, estimate_seqlen=24)
    p, e = model(x)
    p_, e_ = model_(e)
    print(p.shape, e.shape)
    print(p_.shape, e_.shape)