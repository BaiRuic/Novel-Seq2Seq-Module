# 1.structure of code
## 1.1prepare_data:
Consist of two class, Datasets and DataPrepare.

The function of DataPrepare as follows:
+ Data normalization.
+ Tranform series data into input and target pairs, which can be
uesd train supervised model.
+ Split sample into train datasets, valid datasets and test datasets.

The class of Datasets is a generator, which is  inherit by torch.utils.data.Dataset

The parameters of DataPrepare as follows:
1. datapath: 数据集文件路径 
2. dataflie: 数据集名称 
3. input_steps:  [int] 样本的输入步数 
4. pred_horizion: [int] 样本的预测步数
5. Split ratio: [Tuple[float]] 样本划分比例，依次为 训练集、验证集、测试集

After preprocess, the return value of Dataprepare is a tuple, which are `train_ip, train_op, valid_ip, valid_op, test_ip, test_op`
and the shape of ip is [sample_num, input_steps, features], and the shape of op is [pred_horizion]

