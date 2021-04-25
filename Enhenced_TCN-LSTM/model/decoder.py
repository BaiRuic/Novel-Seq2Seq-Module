import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from encoder import TcnEncoder

# 解码器
class GruDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1):
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
        '''
        params:
            inputs: [batch_size, features=1, seq_len=1]
            prev_hidden [1, batch_size, hidden_size]
        returns:
            prediction: [batch_size, feature=1， seq_len=1]
            hidden: [1, batch_size, hidden_size]
        '''
        # output [batch_size, features=hidden_size, seq_len=1]
        # hidden [1, batch_size, hidden_size]
        output, hidden = self.gru(inputs, prev_hidden)
        # prediction [batch_size, output_size=1, seq_len=1]
        prediction = self.fc(output.squeeze(dim=1)).unsqueeze(dim=1)
        return prediction, hidden


if __name__ == "__main__":

    batch_size = 128
    time_step = 14
    features = 2
    # 测试输入 (batch_size, time_step, features) 是否可行
    x = torch.rand(size=(batch_size, time_step, features))
    x = x.permute(0, 2, 1)
    encoder = TcnEncoder(num_inputs=features,
                            num_channels=[4,6,5],
                            kernel_size=3,
                            dropout=0.5)
    prev_hidden = encoder(x)
    print(prev_hidden.shape)


    decoder = GruDecoder(input_size=1, hidden_size=5)

    decoder_input = x[:, 0, -1].reshape(batch_size, 1, 1)
    pred, hidden = decoder(decoder_input, prev_hidden)
    print(pred.shape, hidden.shape)

