import matplotlib.pyplot as plt
from numpy.core.numeric import outer
import pandas as pd
import glob
from sklearn.preprocessing import MinMaxScaler
import torch
import numpy as np
from model import Lstm_model
import torch.utils.data as Data
from sklearn.metrics import mean_squared_error
import os
import time
def series_to_supervised(data, n_in=12, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # 输入数据的构造（t-n_in...t-1）
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))#.shift(i)将数据移动到对应的位数。
        names += [('%s(t-%d)' % (col, i)) for col in data.columns]
    # 预测数据的构造（t...t+n_out-1）
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('%s(t)' % (col)) for col in data.columns]
        else:
            names += [('%s(t+%d)' % (col, i)) for col in data.columns]
    # 数据融合
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # 去除空表格，得到最终数据
    if dropnan:
        agg.dropna(inplace=True)
    return agg  # [78457 rows x 26 columns]


def main():
    plt.rcParams['font.family'] = 'Microsoft YaHei'
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    rmse_list = pd.DataFrame(columns=(' ', 'train', 'test'))
    csv_list = glob.glob('./hours2/*.csv')
    # 使用GPU加速
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    print(device)
    print(csv_list)
  #  parameter_list=['f10.7_index','Dst-index','ap_index']
    # 下采样 /4

    for i in range(0, len(csv_list), 1):
        data = pd.read_csv('./hours2\\tec(12.5, 135).csv', parse_dates=[0], engine='python')#csv_list[i]
        data = data[data[data.columns[0]] >= '2000-01-01']
        data = data[data[data.columns[0]] < '2018-01-01']
        data = data[[data.columns[0], data.columns[1], ]]#parameter_list[j]
        # 将2017年数据作为测试集
        for year in range(2010,2011):
            data_st = str(year) + '-01-01'
            data_ed = str(year) + '-03-01'
            data_1 = data[data[data.columns[0]] >= '2000-01-01']
            data_1 = data_1[data_1[data_1.columns[0]] < data_st]
            data_2 = data[data[data.columns[0]] >= data_ed]
            data_2 = data_2[data_2[data_2.columns[0]] < '2018-01-01']
            train_data = data_1.append(data_2)
            train = train_data[[train_data.columns[0], train_data.columns[1], ]]#parameter_list[j]
            # 筛选2000-2017除year年份外的数据为训练集

            test_data = data[data[data.columns[0]] >= data_st]
            test_data = test_data[test_data[test_data.columns[0]] < data_ed]

            #print(test_data[test_data.columns[0]].iloc[1])
            test_2001 = test_data[[test_data.columns[0], test_data.columns[1], ]]#parameter_list[j]
            # 筛选year数据为测试集

            n_in = 12*30
            n_out = 12
            #print(data)

            series = series_to_supervised(train.drop(data.columns[0], axis=1), n_in=n_in, n_out=n_out)#drop 删除某列
            #print(series)
            series.drop(['%s(t)' % (col) for col in train.columns[2:]], axis=1,
                        inplace=True)  # 删除最后一列 [74137 rows x 25 columns]
            train_data1 = np.array(series)
            scaler = MinMaxScaler()
            train_data2 = scaler.fit_transform(train_data1)#，fit_transform(X_train) 意思是找出X_train的均值和标准差，并应用在X_train上
            train_x, train_y = train_data2[:, :-n_out], train_data2[:, -n_out:]

            #print(train_x.shape,train_y.shape)
            train_x = train_x.reshape(train_x.shape[0],1 ,n_in )  # (65415, 12, 29)
            train_y = train_y.reshape(train_y.shape[0],1, n_out )
            train_x_lstm = torch.from_numpy(train_x)
            train_y_lstm = torch.from_numpy(train_y)
            epochs = 15
            BATCH_SIZE = 48
            lr = 0.001

            torch_dataset = Data.TensorDataset(train_x_lstm, train_y_lstm)
            loader = Data.DataLoader(dataset=torch_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

            model = Lstm_model(inputsize=n_in, outputsize=n_out).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            criterion = torch.nn.MSELoss()
            lstm_history = []
            for epoch in range(epochs):
                since = time.time()
                train_loss = 0.0
                model.train()
                for _, data_train in enumerate(loader):
                    x, y = data_train
                    y=y.squeeze()
                    x, y = x.to(device), y.float().to(device)
                    optimizer.zero_grad()
                    outputs = model(x).squeeze()

                    optimizer.zero_grad()
                    outputs = model(x).squeeze()
                    y
                    # print(outputs,y)
                    batch_loss = criterion(outputs, y)

                    batch_loss.backward()
                    optimizer.step()
                    train_loss += batch_loss.item()
                lstm_history.append(train_loss / len(loader))
                print(f'epoch:{epoch}/30 in point{i}/285')
                time_elapsed = time.time() - since
                print('Training complete in {:.0f}m {:.0f}s'.format(
                    time_elapsed // 60, time_elapsed % 60))
                print(f'Training loss:{train_loss}')
            model.eval()
            test_data_2001 = series_to_supervised(test_2001.drop(data.columns[0], axis=1), n_in=n_in, n_out=n_out)
            test_data_2001.drop(['%s(t)' % (col) for col in train.columns[2:]], axis=1, inplace=True)
            test_data1_2001 = np.array(test_data_2001)
            print(test_data1_2001.shape)
            scaler = MinMaxScaler()
            test_data2_2001 = scaler.fit_transform(test_data1_2001)
            print(test_2001.shape)
            test_x_2001 = test_data2_2001[:, :-n_out]
            # print(test_y)
            test_x_lstm = torch.from_numpy(test_x_2001)  # torch.Size([4291, 348])
            test_x_lstm = test_x_lstm.reshape(test_x_lstm.shape[0], n_in, -1).permute(1, 0,
                                                                                      2)  # torch.Size([48,12, 4291, 29])
            '''print(test_x_lstm.shape)
            test_y = test_x_lstm[0, :,:]
            for col in range(11, n_out, test_x_2001.shape[0]):
                test_x = test_x_lstm[col, :,:]
                test_y = np.vstack((test_y, test_x))
            test_x_lstm = test_y
            test_x_lstm=torch.from_numpy(test_x_lstm)
            print(test_x_lstm.shape)
            '''

            model.eval()
            with torch.no_grad():
                test_lstm_pre_2001 = model(test_x_lstm.float().to(device)).cpu().squeeze().numpy()  # (4291,)
            print(test_lstm_pre_2001.shape)  # （260,12）
            test_data_lstm_2001 = np.c_[test_x_2001, test_lstm_pre_2001]  # (4291, 349)#左右组合。
            test_data_lstm_2001 = scaler.inverse_transform(test_data_lstm_2001)

            rmse_test_2001 = np.sqrt(mean_squared_error(test_data1_2001[:, -n_out:], test_data_lstm_2001[:, -n_out:]))
            print(rmse_test_2001)

            rmse_list = pd.DataFrame(columns=(data.columns[1], rmse_test_2001))
            all_data = pd.DataFrame(test_data_lstm_2001)


if __name__ == '__main__':
    main()
