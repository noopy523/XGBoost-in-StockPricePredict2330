import math
import matplotlib
import numpy as np
import pandas  as pd
import seaborn as sns
import time

from datetime import date
from matplotlib import pyplot as plt
from pylab import rcParams
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from xgboost import XGBRegressor

test_size = 0.2                # 訓練測試集80/20
N = 3                          

model_seed = 100

def get_mape(y_true, y_pred):
    """
    Compute mean absolute percentage error (MAPE)
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def _load_data():

    stk_path = "./2330TW_2019_20210831.csv"
    df = pd.read_csv(stk_path, sep=",")
    # 轉換時間至datetime
    df.loc[:, 'Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
    # 將所有行標題改為小寫，並刪除間距
    df.columns = [str(x).lower().replace(' ', '_') for x in df.columns]
    # 獲取每個樣本的月份
    df['month'] = df['date'].dt.month
    # 排序datetime
    df.sort_values(by='date', inplace=True, ascending=True)

    return df

def feature_engineer(df):

    df['range_hl'] = df['high'] - df['low']
    df['range_oc'] = df['open'] - df['close']

    lag_cols = ['adj_close', 'range_hl', 'range_oc', 'volume']
    shift_range = [x + 1 for x in range(N)]

    for col in lag_cols:
        for i in shift_range:

            new_col='{}_lag_{}'.format(col, i)   # 格式化字符串
            df[new_col]=df[col].shift(i)

    return df[N:]

def scale_row(row, feat_mean, feat_std):
    """
    Given a pandas series in row, scale it to have 0 mean and var 1 using feat_mean and feat_std
    Inputs
        row      : pandas series. Need to scale this.
        feat_mean: mean
        feat_std : standard deviation
    Outputs
        row_scaled : pandas series with same length as row, but scaled
    """
    # If feat_std = 0 (this happens if adj_close doesn't change over N days),
    # set it to a small number to avoid division by zero
    feat_std = 0.001 if feat_std == 0 else feat_std
    row_scaled = (row - feat_mean) / feat_std

    return row_scaled

def get_mov_avg_std(df, col, N):
    """
    Given a dataframe, get mean and std at timestep t using values from t-1, t-2, ..., t-N.
    Inputs
        df         : dataframe. 
        col        : name of the column  want to calculate mean and std 
        N          : get mean and std  at timestep t using values from t-1, t-2, ..., t-N
    Outputs
        df_out     : same as df but with additional column containing mean and std 
    """
    mean_list = df[col].rolling(window=N, min_periods=1).mean()  # len(mean_list) = len(df)
    std_list = df[col].rolling(window=N, min_periods=1).std()  # 第一個值是 NaN，因為 N-1 標準化

    # 為預測添加一個時間步長,這裡又移動了一步
    mean_list = np.concatenate((np.array([np.nan]), np.array(mean_list[:-1])))
    std_list = np.concatenate((np.array([np.nan]), np.array(std_list[:-1])))

    df_out = df.copy()
    df_out[col + '_mean'] = mean_list
    df_out[col + '_std'] = std_list

    return df_out

if __name__ == '__main__':

    # 第一步：載入數據
    data_df=_load_data()

    # 第二步：特徵工程
    df=feature_engineer(data_df)

    # 第三步：數據標準化，先計算出標準化的數據，在對其進行數據分割
    cols_list = [
        "adj_close",
        "range_hl",
        "range_oc",
        "volume"
    ]
    for col in cols_list:
        df = get_mov_avg_std(df, col, N)


    # 第四步：生成訓練資料和測試資料。因訓練資料和測試資料的標準化方式不同，因此需切分訓練和測試資料。
    num_test = int(test_size * len(df))
    num_train = len(df) - num_test
    train = df[:num_train]
    test = df[num_train:]

    # 第五步：標籤和特徵的標準化，此步的目的是為了對在訓練集不能代表總體的情況下，使樹模型正確運行的一種技巧
    cols_to_scale = [
        "adj_close"
    ]
    for i in range(1, N + 1):
        cols_to_scale.append("adj_close_lag_" + str(i))
        cols_to_scale.append("range_hl_lag_" + str(i))
        cols_to_scale.append("range_oc_lag_" + str(i))
        cols_to_scale.append("volume_lag_" + str(i))

    scaler = StandardScaler() # Note：標準化不應使用在測試集，以避免資訊在訓練時提前得知
    train_scaled = scaler.fit_transform(train[cols_to_scale])
    # Convert the numpy array back into pandas dataframe
    train_scaled = pd.DataFrame(train_scaled, columns=cols_to_scale)
    train_scaled[['date', 'month']] = train.reset_index()[['date', 'month']]

    test_scaled = test[['date']]
    for col in tqdm(cols_list):
        feat_list = [col + '_lag_' + str(shift) for shift in range(1, N + 1)]
        temp = test.apply(lambda row: scale_row(row[feat_list], row[col + '_mean'], row[col + '_std']), axis=1)
        test_scaled = pd.concat([test_scaled, temp], axis=1)

    # 第六步：建立樣本
    features = []
    for i in range(1, N + 1):
        features.append("adj_close_lag_" + str(i))
        features.append("range_hl_lag_" + str(i))
        features.append("range_oc_lag_" + str(i))
        features.append("volume_lag_" + str(i))

    target = "adj_close"

    X_train = train[features]
    y_train = train[target]
    X_sample = test[features]
    y_sample = test[target]

    X_train_scaled = train_scaled[features]
    y_train_scaled = train_scaled[target]
    X_sample_scaled = test_scaled[features]

    # 第七步：開始訓練
    from sklearn.model_selection import GridSearchCV
    parameters={'n_estimators':[90],
                'max_depth':[7],
                'learning_rate': [0.3],
                'min_child_weight':range(5, 21, 1),
                }
    #parameters={'max_depth':range(2,10,1)}
    model=XGBRegressor(seed=model_seed,
                         n_estimators=100,
                         max_depth=3,
                         eval_metric='rmse',
                         learning_rate=0.1,
                         min_child_weight=1,
                         subsample=1,
                         colsample_bytree=1,
                         colsample_bylevel=1,
                         gamma=0)
    gs=GridSearchCV(estimator= model,param_grid=parameters,cv=5,refit= True,scoring='neg_mean_squared_error')

    gs.fit(X_train_scaled,y_train_scaled)
    print ('最佳參數: ' + str(gs.best_params_))

    est_scaled = gs.predict(X_train_scaled)
    train['est'] = est_scaled * math.sqrt(scaler.var_[0]) + scaler.mean_[0]

    pre_y_scaled = gs.predict(X_sample_scaled)
    test['pre_y_scaled'] = pre_y_scaled
    test['pre_y']=test['pre_y_scaled'] * test['adj_close_std'] + test['adj_close_mean']

    plt.figure()
    ax = test.plot(x='date', y='adj_close', style='b-', grid=True)
    ax = test.plot(x='date', y='pre_y', style='r-', grid=True, ax=ax)
    plt.show()

    rmse=math.sqrt(mean_squared_error(y_sample, test['pre_y']))
    print("RMSE on dev set = %0.3f" % rmse)
    mape = get_mape(y_sample, test['pre_y'])
    print("MAPE on dev set = %0.3f%%" % mape)

    imp = list(zip(train[features], gs.best_estimator_.feature_importances_))
    imp.sort(key=lambda tup: tup[1])
    for i in range(-1,-10,-1):
        print(imp[i])

