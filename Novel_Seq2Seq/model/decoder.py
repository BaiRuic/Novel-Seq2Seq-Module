import torch
import torch.nn as nn


# 解码器
class GruDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1):
        """
        解码器
        :param input_size: 在解码器部分输入序列的维度 为 1
        :param hidden_size: 需同上下文向量特征维度相同 ，即编码器参数 num_channels[-1]
        :param output_size: 结果线性层的输出维度  为1
        """
        super(GruDecoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.gru = nn.GRU(input_size=self.input_size,
                          hidden_size=self.hidden_size,
                          num_layers=1,
                          bias=True,
                          batch_first=True)
        self.fc = nn.Linear(in_features=self.hidden_size, out_features=self.output_size, bias=True)

    def forward(self, inputs, prev_hidden):
        """
        params:
            inputs: [batch_size, features=1, seq_len=1]
            prev_hidden [1, batch_size, hidden_size]
        returns:
            prediction: [batch_size, feature=1， seq_len=1]
            hidden: [1, batch_size, hidden_size]
        """
        # output [batch_size, features=hidden_size, seq_len=1]
        # hidden [1, batch_size, hidden_size]
        output, hidden = self.gru(inputs, prev_hidden)
        # prediction [batch_size, output_size=1, seq_len=1]
        prediction = self.fc(output.squeeze(dim=1)).unsqueeze(dim=1)
        return prediction, hidden


if __name__ == "__main__":
    from encoder import TcnEncoder
    batch_size = 128
    time_step = 24
    features = 6
    # 测试输入 (batch_size, time_step, features) 是否可行
    x = torch.rand(size=(batch_size, time_step, features))
    x = x.permute(0, 2, 1)
    encoder = TcnEncoder(num_inputs=features,
                        num_channels=[4,6,8,12],
                        kernel_size=4,
                        dropout=0.5)
    prev_hidden = encoder(x)  # shape(1, batch_size, 12)

    decoder = GruDecoder(input_size=1, hidden_size=12, output_size=1)

    decoder_input = x[:, 0, -1].reshape(batch_size, 1, 1)  # shape(batch_size, 1, 1)
    pred, hidden = decoder(decoder_input, prev_hidden)

    print(pred.shape,  hidden.shape)

