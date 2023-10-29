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
from sklearn.metrics import mean_absolute_error

# 数据整理
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
    # 绘图字体设置
    plt.rcParams['font.family'] = 'Microsoft YaHei'
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    #rmse_list = pd.DataFrame(columns=(' ', 'train', 'test'))
    csv_list = glob.glob('./hours/*.csv')
    # 使用GPU加速
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    print(device)
    # parameter_list=['f10.7_index','Dst-index','ap_index']
    # 下采样 /4

    for i in range(0, len(csv_list), 1):
        data = pd.read_csv(csv_list[i], parse_dates=[0], engine='python')#csv_list[i]
        data = data[data[data.columns[0]] >= '2000-01-01']
        data = data[data[data.columns[0]] < '2018-01-01']
        data = data[[data.columns[0], data.columns[1], ]]#parameter_list[j]
        # 将2017年数据作为测试集
        # lzw: 可能之前有打算把更多的数据作为测试集，所以用了for
        for year in range(2017,2018):
            data_st = str(year) + '-07-01'
            data_ed = str(year+1) + '-08-01'
            # lzw: data_1 2000-01-01 到 2017-07-01
            data_1 = data[data[data.columns[0]] >= '2000-01-01']
            data_1 = data_1[data_1[data_1.columns[0]] < data_st]
            # lzw: data_2 2018-08-01 到 2018-01-01  感觉这里多半是写错了，恐怕就是NaN的原因
            data_2 = data[data[data.columns[0]] >= data_ed]
            data_2 = data_2[data_2[data_2.columns[0]] < '2018-01-01']
            # lzw: data_2 果然是空集
            train_data = data_1.append(data_2)
            train = train_data[[train_data.columns[0], train_data.columns[1], ]]#parameter_list[j]

            # 筛选2000-2017除year年份外的数据为训练集
            # lzw: test_data 2017-07-01 到 2018-08-01 ,我寻思下面的test_data 和 test_2001 不是一个东西嘛
            test_data = data[data[data.columns[0]] >= data_st]
            test_data = test_data[test_data[test_data.columns[0]] < data_ed]
            # print(test_data[test_data.columns[0]].iloc[1])
            test_2001 = test_data[[test_data.columns[0], test_data.columns[1], ]]#parameter_list[j]
            # 筛选year数据为测试集
            n_in = 120
            n_out = 1
            series = series_to_supervised(train.drop(data.columns[0], axis=1), n_in=n_in, n_out=n_out) # drop 删除某列

            # todo： 这一行疑似被错误调用，实际得出的series为76141 × 121，且没有起到删除最后一列功能
            series.drop(['%s(t)' % (col) for col in train.columns[2:]], axis=1,
                        inplace=True)  # 删除最后一列 [74137 rows x 25 columns

            train_data1 = np.array(series)
            scaler = MinMaxScaler()
            train_data2 = scaler.fit_transform(train_data1) # 归一化
            # 以上部分完全是数据格式的处理
            train_x, train_y = train_data2[:, :-1], train_data2[:, -1]
            train_x_lstm = train_x.reshape((train_x.shape[0], 1, train_x.shape[1]))
            train_x_lstm = torch.from_numpy(train_x_lstm)
            train_y = torch.from_numpy(train_y)

            epochs = 15
            BATCH_SIZE = 32
            lr = 0.001

            torch_dataset = Data.TensorDataset(train_x_lstm, train_y)
            loader = Data.DataLoader(dataset=torch_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
            model = Lstm_model(inputsize=n_in, outputsize=1).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            # lzw：误差采用方均误差
            criterion = torch.nn.MSELoss()
            # lzw: lstm_history 用于记录平均训练损失收敛过程
            lstm_history = []
            for epoch in range(epochs):
                since = time.time()
                train_loss = 0.0
                model.train()
                # lzw: 额，不用j计数为啥要用enumerate
                for _, data_train in enumerate(loader):
                    x, y = data_train
                    x, y = x.to(device), y.float().to(device)
                    optimizer.zero_grad()
                    outputs = model(x).squeeze()
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
            with torch.no_grad():
                pre_train_lstm = model(train_x_lstm.to(device)).cpu().squeeze().numpy()
            train_data_lstm = np.c_[train_x, pre_train_lstm]
            train_data_lstm = scaler.inverse_transform(train_data_lstm)
            rmse_train = np.sqrt(mean_squared_error(train_data1[:, -1], train_data_lstm[:, -1]))

            # 测试集

            # lzw: 这段主要是对齐数据格式
            test_data_2001 = series_to_supervised(test_2001.drop(data.columns[0], axis=1), n_in=n_in, n_out=n_out)
            test_data_2001.drop(['%s(t)' % (col) for col in train.columns[2:]], axis=1, inplace=True)
            test_data1_2001 = np.array(test_data_2001)
            test_data2_2001 = scaler.fit_transform(test_data1_2001)
            test_x_2001, test_y_2001 = test_data2_2001[:, :-1], test_data2_2001[:, -1]
            test_x_lstm_2001 = test_x_2001.reshape((test_x_2001.shape[0], 1, test_x_2001.shape[1]))
            test_x_lstm = torch.from_numpy(test_x_lstm_2001).to(device)
            tset_y = torch.from_numpy(test_y_2001).to(device)

            model.eval()
            with torch.no_grad():
                test_lstm_pre_2001 = model(test_x_lstm.to(device)).cpu().squeeze().numpy()
            test_data_lstm_2001 = np.c_[test_x_2001, test_lstm_pre_2001]
            test_data_lstm_2001 = scaler.inverse_transform(test_data_lstm_2001)
            rmse_test_2001 = np.sqrt(mean_squared_error(test_data1_2001[:, -1], test_data_lstm_2001[:, -1]))
            mae = mean_absolute_error(test_data1_2001[:, -1], test_data_lstm_2001[:, -1])

            print(rmse_test_2001, mae)
            rmse_list = pd.DataFrame(columns=(data.columns[1], rmse_test_2001))
            list0 = pd.DataFrame(np.array(test_2001.iloc[12:, 0]))
            list1 = pd.DataFrame(test_data1_2001[:, -1])
            list2 = pd.DataFrame(test_data_lstm_2001[:, -1])
            result_rp=pd.concat(([list0,list1,list2]),axis=1)
            result_rp.columns=['','real','pre']
            print(result_rp)
            #rmse_list=pd.DataFrame(columns=(data.columns[1],rmse_train,rmse_test_2001))
            #rmse_list.loc[i + 1] = rmse  # write the rmse to rmse_list
            #all_data = np.append(train_data_lstm, test_data_lstm_2001, axis=0)
            all_data = pd.DataFrame(test_data_lstm_2001)
            #print(all_data)
            #all_data=pd.DataFrame(test_data_lstm_2001)
            tru = test_data1_2001[:, -1]
            #tru = np.append(tru, test_data1_2001[:, -1])
            y = np.max(tru)
            tru = pd.DataFrame(tru)
            fig, ax = plt.subplots(2, 1, figsize=(15, 10))
            ax[0].plot(lstm_history)
            ax[1].plot(tru, label='true')
            # print(all_data)
            ax[1].plot(all_data.loc[:, all_data.shape[1] - 1], label='testdataset_pre')
            '''ax[1].annotate(text='test_dataset\nRMSE={0}'.format(round(rmse_test_2001, 4)),
                           xy=(len(test_data_lstm_2001) / 2, y - 15),
                           xytext=(len(test_data_lstm_2001) / 2, y - 10),
                           color='black', bbox=dict(boxstyle='round,pad=0.5', fc='yellow', ec='k', lw=1))'''
            paths = 'LSTM_1209'
            if not os.path.exists('./{0}'.format(paths)):
                os.makedirs('./{0}'.format(paths))
            plt.savefig('./{1}/{0}.png'.format(data.columns[1], paths), dpi=300)
            plt.close()
            # lzw：以下代码应该是绘图的废案
            '''fig, ax = plt.subplots(2, 1, figsize=(15, 10))
            ax[0].plot(lstm_history)
            ax[1].plot(tru, label='true')
            ax[1].plot(all_data.loc[0:len(train_data_lstm), 12], label='traindata_pre')
            ax[1].plot(all_data.loc[len(train_data_lstm):, 12], label='testdataset_pre')
            ax[1].axvline(len(train_data_lstm), c='r', ls='--', lw=3)
            ax[1].annotate(text=' fen\n jie \n xian', xy=(len(train_data_lstm), 30),
                           xytext=(len(train_data_lstm) - 2000, 28), color='black', weight='roman',
                           arrowprops=dict(facecolor='#ff7f50', shrink=0.2))
            ax[1].annotate(text='tiran_dataset\nRMSE={0}'.format(round(rmse_train, 4)),
                           xy=(len(train_data_lstm) / 2, y - 15),
                           xytext=(len(train_data_lstm) / 2, y - 10),
                           color='black', bbox=dict(boxstyle='round,pad=0.5', fc='yellow', ec='k', lw=1))
            ax[1].annotate(text='test_dataset\nRMSE={0}'.format(round(rmse_test_2001, 4)),
                           xy=(len(train_data_lstm) + len(test_data_lstm_2001) / 2, y - 15),
                           xytext=(len(train_data_lstm) - 1200 + len(test_data_lstm_2001) / 2, y - 10),
                           color='black', bbox=dict(boxstyle='round,pad=0.5', fc='yellow', ec='k', lw=1))
            plt.legend()
            plt.savefig('./{0}.png'.format(data.columns[1]), dpi=300)
            plt.close()'''


        #torch.save(model.state_dict(), './LSTM_2014/{0}_model/{1}.pth'.format(parameter_list[j],data.columns[1]))
        '''if not (os.path.exists('./LSTM_2014predict_result/{0}.csv'.format(data.columns[1]))):
            pd.DataFrame(columns=('site','real','pre')).to_csv('./LSTM_2014predict_result/{0}.csv'.format(data.columns[1]),float_format='%.4f', index=False)
        result_rp.to_csv('./LSTM_2014/{0}/{1}.csv'.format(parameter_list[j],data.columns[1]), mode='a',float_format='%.4f', index=False)
    rmse_list.to_csv('./LSTM2014.csv', float_format='%.4f', index=False)'''

        if not (os.path.exists('./lstm1209.csv')):
            pd.DataFrame(columns=('site', 'test_loss')).to_csv('./lstm1209.csv', float_format='%.4f', index=False)
        rmse_list.to_csv('./lstm1209.csv', mode='a', float_format='%.4f', index=False)
if __name__ == '__main__':
    main()
