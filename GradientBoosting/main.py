from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
from prepare_data import prepare_data
from sklearn.model_selection import RandomizedSearchCV

# 定义超参数
HyperParams = {'datapath':'..\\prepare_data\\data',      # 数据集路径
               'datafile': 'AT',                         # 数据集文件
               'split_ratio':[0.3, 0.05, 0.05],         # 数据集分割比例
               "features": 3,
               "input_seqlen": 24,
               "pred_seqlen": 6
               }
# 保存模型指标的文件名
filename = f"GB_{HyperParams['datafile']}-{HyperParams['input_seqlen']} to {HyperParams['pred_seqlen']}"

# 构建多步预测模型
def multi_step_pred(model, input_timestep, pred_horizion, test_ip):
    '''test_ip [samples, time_steps]
        test_op [samples,]
    '''
    assert(test_ip.shape[1] == input_timestep)

    results = [] # 存放最后多步预测结果的数组
    for t in range(pred_horizion):
        x = test_ip[:,-input_timestep:] # 只取最后 input_timestep 个时间步来参与预测
        y_hat = model.predict(X=x).reshape(-1, 1)  # [samples, 1]
        results.append(y_hat)
        test_ip = np.hstack((test_ip, y_hat)) # 将当前预测单步的结果拼接到模型输入上，参与下一步预测
    return np.hstack(results)  # [samples, timesteps]


# 模型评估指标类
# 构建评估模型的类
class MyMetrics:
    def __init__(self, y_pred, y_true, un_std):
        '''
        y_hat.shape[samples, pred_horizion]
        un_std : 类 DataPrepare 的实例 ，计算mape的时候需要反归一化
        '''
        self.un_std = un_std
        self.y_pred = y_pred
        self.y_true = y_true
        self.metrics = {}
        self.mse()
        self.mape()
        self.smape()

    def mse(self):
        self.metrics['mse'] = np.mean((self.y_pred - self.y_true) ** 2)

    def mape(self):
        y_pred = self.un_std.un_standardize(x=self.y_pred)
        y_true = self.un_std.un_standardize(x=self.y_true)
        self.metrics['mape'] = np.mean(np.abs((y_pred - y_true) / (y_true))) * 100

    def smape(self):
        self.metrics['smape'] = 2.0 * np.mean(
            np.abs(self.y_pred - self.y_true) / (np.abs(self.y_pred) + np.abs(self.y_true))) * 100

    def print_metrics(self):
        for key, values in self.metrics.items():
            print(f"The {key} is:{values}.")


def main():
    # 构建数据集
    print("Preparing dataset")
    Data = prepare_data.DataPrepare(datapath=HyperParams["datapath"],
                                    datafile=HyperParams["datafile"],
                                   input_steps=HyperParams['input_seqlen'],
                                   pred_horizion=HyperParams['pred_seqlen'],
                                   split_ratio=HyperParams['split_ratio'],)
    temp = Data.prepare_data()  # temp (train_ip, train_op, valid_ip, valid_op, test_ip, test_op)
    train_ip, train_op, valid_ip, valid_op, test_ip, test_op = temp[0], temp[1], temp[2], temp[3], temp[4], temp[5]

    # 对训练集、验证集 提取数据集中的电力负载信息，对测试集 输出不做改变
    train_ip = train_ip[:, :, 0]
    train_op = train_op[:, 0]
    valid_ip = valid_ip[:, :, 0]
    valid_op = valid_op[:, 0]
    test_ip = test_ip[:, :, 0]

    # 建立模型 模型参数由网格搜索得到
    print('Training model')
    GB_grid = {
        'n_estimators': [100, 500],
        'learning_rate': [0.1, 0.05, 0.02],
        'max_depth': [4],
        'min_samples_leaf': [3],
        'max_features': [1.0]
    }
    GB = GradientBoostingRegressor()
    GB_search = RandomizedSearchCV(GB, GB_grid, cv=3)
    GB_search.fit(X=train_ip, y=train_op)    # 选择最佳模型
    model = GB_search.best_estimator_

    # 在测试集上多步预测
    print("Forecasting")
    y_hat= multi_step_pred(model, HyperParams['input_seqlen'], HyperParams['pred_seqlen'], test_ip)

    # 评估模型
    print("Evaluating model")
    my_metrics = MyMetrics(y_hat, test_op, Data)

    my_metrics.print_metrics()

    # 将模型指标保存下来
    np.save(filename, my_metrics.metrics)

if __name__ == "__main__":
    load_model = False
    if load_model:
        print(f"load {filename}.npy ing")
        temp_dict = np.load(f"{filename}.npy",allow_pickle=True)
        for key, values in temp_dict.item().items():
            print(f"The {key} is:{values}.")
    else:
        main()


