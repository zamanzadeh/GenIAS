import numpy as np
import pandas as pd
import ast
import os

def load_data(ds_name, fname):

    datasets_path = '../../datasets/'

    if ds_name == 'MSL' or ds_name == 'SMAP':
        with open(datasets_path+'MSL_SMAP/labeled_anomalies.csv', 'r') as file:
            csv_reader = pd.read_csv(file, delimiter=',')
        data_info = csv_reader[csv_reader['chan_id'] == fname]

        train_path = os.path.join(datasets_path+'MSL_SMAP/train/', fname+'.npy')
        test_path = os.path.join(datasets_path+'MSL_SMAP/test/', fname+'.npy')
        train_data = np.load(train_path)
        test_data = np.load(test_path)

        if np.any(sum(np.isnan(train_data))!=0):
            print('Data contains NaN which replaced with zero')
            train_data = np.nan_to_num(train_data)

        if np.any(sum(np.isnan(test_data))!=0):
            print('Data contains NaN which replaced with zero')
            test_data = np.nan_to_num(test_data)

        labels = []
        for index, row in data_info.iterrows():
            anomalies = ast.literal_eval(row['anomaly_sequences'])
            length = row.iloc[-1]
            label = np.zeros([length], dtype=int)
            for anomaly in anomalies:
                label[anomaly[0]:anomaly[1] + 1] = 1
            labels.extend(label)
        test_label = np.asarray(labels)
    
    elif ds_name == 'SMD':
        train_path = os.path.join(datasets_path+'SMD/train/', fname)
        train_data = pd.read_csv(train_path)
        train_data = np.asarray(train_data)
        test_path = os.path.join(datasets_path+'SMD/test/', fname)
        test_data = pd.read_csv(test_path)
        test_data = np.asarray(test_data)

        if np.any(sum(np.isnan(train_data))!=0):
            print('Data contains NaN which replaced with zero')
            train_data = np.nan_to_num(train_data)
        if np.any(sum(np.isnan(test_data))!=0):
            print('Data contains NaN which replaced with zero')
            test_data = np.nan_to_num(test_data)

        labels = pd.read_csv(os.path.join(datasets_path+'SMD/test_label/', fname))
        test_label = np.asarray(labels)

    elif ds_name == 'GECCO':
        time_series_gec = pd.read_csv(datasets_path+"GECCO/1_gecco2018_water_quality.csv")
        temp = time_series_gec.iloc[:, 2:]
        mask = np.isnan(temp).any(axis=1)
        temp = temp[~mask]
        split_index = int(0.5 * len(temp))
        
        train_data = temp.iloc[0:split_index].drop(columns=['EVENT'])
        train_data = np.asarray(train_data)
        test_label = temp.iloc[split_index:, -1]
        test_data = temp.iloc[split_index:].drop(columns=['EVENT'])
        test_data = np.asarray(test_data)

        if np.any(sum(np.isnan(train_data))!=0):
            print('Data contains NaN which replaced with zero')
            train_data = np.nan_to_num(train_data)
        if np.any(sum(np.isnan(test_data))!=0):
            print('Data contains NaN which replaced with zero')
            test_data = np.nan_to_num(test_data)
    
    elif ds_name == 'swan':
        train_data = np.load(datasets_path+"Swan/Swan_train.npy")
        test_data = np.load(datasets_path+"Swan/Swan_test.npy")
        test_label = np.load(datasets_path+"Swan/Swan_test_label.npy")

        if np.any(sum(np.isnan(train_data))!=0):
            print('Data contains NaN which replaced with zero')
            train_data = np.nan_to_num(train_data)
        if np.any(sum(np.isnan(test_data))!=0):
            print('Data contains NaN which replaced with zero')
            test_data = np.nan_to_num(test_data)

    elif ds_name == 'swat':
        train_path = os.path.join(datasets_path+'SWAT/swat_train.csv')
        temp = pd.read_csv(train_path)
        train_data = np.asarray(temp.iloc[:, 1:52])
        test_path = os.path.join(datasets_path+'SWAT/swat_test.csv')
        temp = pd.read_csv(test_path)
        test_data = np.asarray(temp.iloc[:, 1:52])
        test_label = np.asarray(temp['attack'])

        if np.any(sum(np.isnan(train_data))!=0):
            print('Data contains NaN which replaced with zero')
            train_data = np.nan_to_num(train_data)
        if np.any(sum(np.isnan(test_data))!=0):
            print('Data contains NaN which replaced with zero')
            test_data = np.nan_to_num(test_data)

    elif ds_name == 'Yahoo':
        filename = os.path.join(datasets_path+'Yahoo/', fname)
        df = pd.read_csv(filename)
        data = np.array(df['value'])
        labels = np.array(df['is_anomaly'])

        l = len(data) // 2
        train_data = data[:l]
        test_data = data[l:]
        test_label= labels[l:]

    elif ds_name == 'KPI':
        train_path = os.path.join(datasets_path+'KPI/train/', fname)
        temp = pd.read_csv(train_path)
        temp = temp.set_index(['timestamp']).sort_index()
        train_data = np.asarray(temp['value'])
        
        test_path = os.path.join(datasets_path+'KPI/test/', fname)
        temp = pd.read_csv(train_path)
        temp = temp.set_index(['timestamp']).sort_index()
        test_data = np.asarray(temp['value'])
        test_label = np.asarray(temp['label'])

        if np.any(sum(np.isnan(train_data))!=0):
            print('Data contains NaN which replaced with zero')
            train_data = np.nan_to_num(train_data)

        if np.any(sum(np.isnan(test_data))!=0):
            print('Data contains NaN which replaced with zero')
            test_data = np.nan_to_num(test_data)

    elif ds_name == 'UCR':
        file_path = os.path.join(datasets_path+'UCR', fname)
        df = pd.read_csv(file_path)

        mask = np.isnan(df).any(axis=1)
        df = df[~mask]
        all_data = np.asarray(df)
        all_data = np.squeeze(all_data)

        sp_points = (fname.removesuffix('.txt')).split('_')
        tr_point = int(sp_points[-3])
        start_anom, end_anom = int(sp_points[-2]), int(sp_points[-1])
        d_sz = df.shape
        
        all_labels = np.zeros(d_sz[0])
        all_labels[(start_anom):(end_anom +1)] = 1

        train_data = all_data[:tr_point]
        test_data = all_data[tr_point:]
        labels = all_labels[tr_point:]
        test_label = np.asarray(labels)

    else: 
        print(f'No loader for {ds_name} dataset!')
    
    if train_data.ndim == 1:
        train_data = train_data.reshape(-1, 1)
        test_data = test_data.reshape(-1, 1)
        
    return train_data, test_data, test_label