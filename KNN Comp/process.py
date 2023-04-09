# -*- coding: utf-8 -*-
"""processing.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1PZmsVvRxooJEZM1W23DPhQejRZeaLaeP

# load libraries and data
"""

# import libraries
import pandas as pd
import numpy as np 
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

# Commented out IPython magic to ensure Python compatibility.
# %cd  /gdrive/MyDrive/364/
# %ls

"""# Feature Engineering"""

"""## [Done] Dealing with zeros in columns"""


def process_zero_col(data, method='drop-col'):
    # methods: drop col, drop row, replace with mean,
    data = np.copy(data)
    #   data = data.copy(deep=True)
    # zero_cols = ['A2', 'A3', 'A5']
    zero_cols = [5] # np indices
    if method == 'drop-col':
        return data.drop(columns=zero_cols)
    elif method == 'drop-row':
        return data.loc[~(data == 0).all(axis=1)]
    elif method == 'mean':
        for col in zero_cols:
            col_data = data[:, col]
            col_data[col_data == 0] = col_data.mean()
        return data


## Log Columns
def log_transform(data):
    data = np.copy(data)
    #   data = data.copy(deep=True)
    # col_list = list(data.columns)
    # categories = ['A8_1', 'A8_3', 'A8_7', 'A8_14', 'A8_28', 'A8_56', 'A8_90',
    #               'A8_91', 'A8_100', 'A8_120', 'A8_180', 'A8_270', 'A8_360', 'A8_365']
    # calc_columns = [i for i in col_list if i not in categories]
    
    # for col in calc_columns:
    #     if 'sqrt' in col:
    #         calc_columns.remove(col)
        
         
    for col in range(8): 
        col_data = data[:, col] 
        # col_data = np.reshape(col_data, [-1, 1])
        # print(col_data.shape)
        # print(np.log10(col_data).shape)
        data = np.append(data, np.log10(col_data).reshape([-1,1]), 1)
        # data[:, 8+col] = np.log10(data[col])
    return data




"""## [Done] Column Multiplication"""


def col_mult(data):
    # data = data.copy(deep=True)
    # col_list = list(data.columns)
    data = np.copy(data)
    #   data = data.copy(deep=True)
    # categories = ['A8_1', 'A8_3', 'A8_7', 'A8_14', 'A8_28', 'A8_56', 'A8_90',
    #               'A8_91', 'A8_100', 'A8_120', 'A8_180', 'A8_270', 'A8_360', 'A8_365']
    
    
    # calc_columns = [i for i in col_list if i not in categories]
    # calc_columns = col_list 
    num = data.shape[1]
    for col in range(num):
        for v in range(col + 1, num):
            if v != col: 
                data = np.append(data, np.multiply(data[:, col],data[:, v]).reshape(-1, 1), 1)
    # for idx, value in enumerate(calc_columns[:-1]):
    #     for v in calc_columns[idx + 1:]:
    #         if v != value:
    #             data[value + "x" + v] = data[value] * data[v]

    return data
    
    
def col_div(data):
   
    data = np.copy(data)

    num = data.shape[1]
    for col in range(num):
        for v in range(col + 1, num):
            if v != col: 
                data = np.append(data, np.divide(data[:, col],data[:, v]).reshape(-1, 1), 1)
    return data
    
    



# ## Sqrt Columns
# def sqrt_transform(data):
#     data = data.copy(deep=True)
#     col_list = list(data.columns)
#     categories = ['A8_1', 'A8_3', 'A8_7', 'A8_14', 'A8_28', 'A8_56', 'A8_90',
#                   'A8_91', 'A8_100', 'A8_120', 'A8_180', 'A8_270', 'A8_360', 'A8_365']
#     calc_columns = [i for i in col_list if i not in categories]
#     for col in calc_columns:
#         if 'log' in col:
#             calc_columns.remove(col)
        
#     for col in calc_columns:
#         data['sqrt' + col] = np.sqrt(data[col])
#     return data
# """## [Done] A8 Categorical"""


# def categorical_transform(data, keep_a8=False):
#     data = data.copy(deep=True)
#     s = pd.get_dummies(data, columns=['A8'])
#     # s.head()
#     if keep_a8:
#         s['A8'] = data['A8']

#     return s


def process(x: np.array) -> np.array:
    x = process_zero_col(x, method='mean') 
    x = log_transform(x)
    scaler = StandardScaler()   
    scaler.fit(x)
    x = scaler.transform(x)
    return x
