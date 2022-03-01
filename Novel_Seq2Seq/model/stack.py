import torch.nn as nn
import torch
from . basic_block import BasicBlock as BasicBlock
from typing import List

Vector = List[int]
DEVICE = torch.device('cuda' if torch.cuda.is_available() else "cpu")


class Stack(nn.Module):
    def __init__(self, input_size: int = 3,
                 encoder_channels: Vector = [4, 6],
                 input_seqlen: int = 24,
                 forecast_seqlen: int = 6):
        super(Stack, self).__init__()

        self.estimate_seqlen = input_seqlen
        self.forecast_seqlen = forecast_seqlen
        self.input_size = input_size
        self.encoder_channels = encoder_channels

        self.block_1 = BasicBlock(input_size=self.input_size,
                                  encoder_num_channels=self.encoder_channels,
                                  forecast_seqlen=self.forecast_seqlen,
                                  estimate_seqlen=self.estimate_seqlen)

        self.block_2 = BasicBlock(input_size=1,
                                  encoder_num_channels=self.encoder_channels,
                                  forecast_seqlen=self.forecast_seqlen,
                                  estimate_seqlen=self.estimate_seqlen)
        '''
        self.block_3 = BasicBlock(input_size=1,
                                  encoder_num_channels=self.encoder_channels,
                                  forecast_seqlen=self.forecast_seqlen,
                                  estimate_seqlen=self.estimate_seqlen)

        self.block_4 = BasicBlock(input_size=1,
                                  encoder_num_channels=self.encoder_channels,
                                  forecast_seqlen=self.forecast_seqlen,
                                  estimate_seqlen=self.estimate_seqlen)
        '''

    def forward(self, inputs):
        '''
        params:
            inputs : 整个stack的输入 [batch_size, seq_len, feature=2]
        returns:
            forecast : 所有basicblock的预测输出之和  [batch_size, predict_seqlen, 1]
            input_estimate: 模型最后的输出，也就是输入 减  最佳估计 [batch_size, input_seqlen, 1]
        '''
        # forcast value ，之后的每一个block输出的预测值都加进来
        batch_size = inputs.shape[0]
        forecast = torch.zeros(size=(batch_size, self.forecast_seqlen, 1)).to(DEVICE)

        # forecast_ 和 x_ 都是临时变量 暂时存放当前 block 的输出
        forecast_, x_ = self.block_1(inputs)
        forecast += forecast_
        inputs = x_

        forecast_, x_ = self.block_2(inputs)
        forecast += forecast_
        inputs = x_
        '''
        forecast_, x_ = self.block_3(inputs)
        forecast += forecast_
        inputs = x_

        forecast_, x_ = self.block_4(inputs)
        forecast += forecast_
        inputs = x_
        '''
        return forecast, inputs


if __name__ == "__main__":
    batch_size = 128
    time_step = 24
    pred_horizion = 6
    features = 2
    # 测试 输入为 (batch_size, time_step, features) 是否可行
    x = torch.rand(size=(batch_size, time_step, features)).to(DEVICE)

    model = Stack(input_size=features, encoder_channels=[4, 6, 8, 12], input_seqlen=time_step, forecast_seqlen=pred_horizion).to(DEVICE)
    y, e = model(x)
    print(y.shape)
    print(e.shape)