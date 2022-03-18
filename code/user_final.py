import os

import pandas as pd

from logger import Logger

data_path = os.path.join(os.path.dirname(os.getcwd()), 'data', '0315')
output_path = os.path.join(os.path.dirname(os.getcwd()), 'dataset')
if not os.path.exists(data_path):
    os.mkdir(data_path)
if not os.path.exists(output_path):
    os.mkdir(output_path)

log = Logger("user").getLogger()

user = pd.read_csv(os.path.join(data_path, '用户.csv'), sep=',', encoding='utf-8', header=0)
df = pd.DataFrame(user, columns=['id', 'sex', 'zan_count'])
df['id'] = pd.to_numeric(df['id']).fillna('0').astype('int64')
df['sex'] = pd.to_numeric(df['sex']).fillna('0').astype('int64')
df['zan_count'] = pd.to_numeric(df['zan_count']).fillna('0').astype('int64')

#  = df['sex'].astype('int64')
# df['zan_count'] = df['zan_count'].astype('int64')
df.to_csv(os.path.join(output_path, 'user.txt'), sep='\t', encoding='utf-8', index=False)
