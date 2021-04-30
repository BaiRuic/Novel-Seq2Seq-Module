import pandas as pd
import torch
from sklearn import preprocessing

# 定义全局变量
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_tensor_type('torch.DoubleTensor')

import os

class DataPrepare:
    '''1.数据归一化 2.将序列数据转化为监督模型数据 3.数据分割为训练集、测试集和验证集
    param: dataflie:数据集名称
            input_steps: [int ]样本的输入步数
            pred_horizion: [int] 样本的预测步数
            Split ratio: [List[float]] 样本划分比例，依次为 训练集、验证集、测试集
    return: train_io, train_op
            valid_io, valid_op
            test_io,test_op
    '''

    def __init__(self, datapath = '.\\prepare_data\\data', datafile='AT', input_steps=12, pred_horizion=6, split_ratio=[0.75, 0.15, 0.10]):

        if datafile in ['AT', 'CH', 'DE', 'NL', 'PL']:
            self.data = pd.read_csv(datapath + '\\' + datafile + '_data.csv', header=0, index_col=0)
        else:
            raise Exception("The parameter 'datafile' is wrong")
        # 该数据集的基本属性
        self.datafile = datafile
        self.pred_horizion = pred_horizion
        self.input_steps = input_steps
        self.features = self.data.shape[1]
        # 负载数据进行归一化的对象
        self.scaler_load = preprocessing.MinMaxScaler(feature_range=(0, 1))
        # 划分数据的比例
        self.split_ratio = split_ratio

    def __repr__(self):
        return f"The dataset of {self.datafile}"

    def _data_to_supervised(self):
        '''将归一化后的时间序列数据转换为供监督学习模型训练的数据
            return: the [DataFrame] data after reframed, shape(sample_num, features*input_steps, pred_horizion)
        '''
        column_data = []  # 存放数据
        column_name = []  # 存放column名字

        # 对输入步数的处理
        for i in range(self.input_steps, 0, -1):
            column_data.append(self.data.shift(periods=i, axis=0))
            if i != 1:
                column_name += ['load(t-' + str(i - 1) + ')', 'temperature(t-' + str(i - 1) + ')',
                                'windspeed(t-' + str(i - 1) + ')']
            else:
                column_name += ['load(t)', 'temp(t)', 'windspeed(t)']
        # 对预测范围的处理，预测值只有load变量！
        for i in range(0, self.pred_horizion):
            column_data.append(self.data['load'].shift(periods=-i, axis=0))
            column_name += ['load(t+' + str(i + 1) + ')']
        reframed_data = pd.concat(column_data, axis=1)
        reframed_data.columns = column_name
        reframed_data.dropna(how='any', axis=0, inplace=True)
        return reframed_data

    def _standardizeData(self):
        '''对数据进行归一化，使其范围在[0,1]之间, 直接对self.data进行操作
            param: 无
            return: 无
        '''
        self.data['load'] = self.scaler_load.fit_transform(self.data[['load']])
        # 定义对温度、风速归一化的对象，临时变量
        scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
        self.data['temperature'] = scaler.fit_transform(self.data[['temperature']])
        self.data['windspeed'] = scaler.fit_transform(self.data[['windspeed']])

    def un_standardize(self, x):
        '''将预测输出的数值反归一化，使其具有物理意义
        param: x: 待反归一化的数值 type: numpy.ndarray  shape:(n, 1)
        return : 反归一化后的数值 type:numpy.ndarray shape:(n, 1)
        '''
        return self.scaler_load.inverse_transform(x)

    def _split_data(self, reframed_data):
        '''将reframed后的数据按照 self.split_ratio 进行分割
        param: reframed_data [DataFrame]
        return: train_io, train_op
                valid_io, valid_op
                test_io,test_op
        '''
        # 先将输入输出分开
        inputs = reframed_data.values[:, :-self.pred_horizion]  # 输入
        targets = reframed_data.values[:, -self.pred_horizion:]  # 输出

        num_examples = inputs.shape[0]
        # 设置各数据集大小
        train_size = int(num_examples * self.split_ratio[0])
        valid_size = int(num_examples * self.split_ratio[1])
        test_size = int(num_examples * self.split_ratio[2])

        # 将训练集输入[sample_num, time_steps*features] reshape 为 [sample_num, time_steps, features]
        train_ip, train_op = inputs[:train_size, :], targets[: train_size, :]
        train_ip = train_ip.reshape(train_ip.shape[0], self.input_steps, self.features)

        valid_ip, valid_op = inputs[train_size: train_size + valid_size, :], targets[train_size: train_size + valid_size, :]
        valid_ip = valid_ip.reshape(valid_ip.shape[0], self.input_steps, self.features)

        test_ip, test_op = inputs[train_size + valid_size:train_size + valid_size + test_size , :], \
                           targets[train_size + valid_size:train_size + valid_size + test_size, :]
        test_ip = test_ip.reshape(test_ip.shape[0], self.input_steps, self.features)

        return (train_ip, train_op, valid_ip, valid_op, test_ip, test_op)

    def prepare_data(self):
        '''1. 对self.data归一化
           2. 转化为监督模型数据
           3. 分割数据
        return : (train_ip, train_op, valid_ip, valid_op, test_ip, test_op)
        '''
        self._standardizeData()
        reframed_data = self._data_to_supervised()
        return_data = self._split_data(reframed_data)
        return return_data


# Dataset
class Datasets(torch.utils.data.Dataset):
    def __init__(self, ip, op):
        super(Datasets, self).__init__()
        self.input = ip
        self.output = op
        self.len = ip.shape[0]

    def __getitem__(self,idx):
        return torch.tensor(self.input[idx]), torch.tensor(self.output[idx])

    def __len__(self):
        return self.len


if __name__ == '__main__':
    Data = DataPrepare(datafile='AT',input_steps=12, pred_horizion=6)

    temp = Data.prepare_data()   # temp (train_ip, train_op, valid_ip, valid_op, test_ip, test_op)
    train_dataset = Datasets(temp[0], temp[1]) # 训练集生成器
    for x,y in train_dataset:
        print(x.shape, y.shape)   # 每个样本的维度
        break