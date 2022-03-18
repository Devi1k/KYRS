import os

import pandas as pd

from logger import Logger

data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', '0315')
output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dataset')
if not os.path.exists(data_path):
    os.mkdir(data_path)
if not os.path.exists(output_path):
    os.mkdir(output_path)

log = Logger("user").getLogger()

user = pd.read_csv(os.path.join(data_path, '用户.csv'), sep=',', error_bad_lines=False, encoding='utf-8', header=0)
df = pd.DataFrame(user, columns=['id', 'sex', 'zan_count'])
df['id'] = df['id'].astype('int64')
df['sex'] = df['sex'].astype('int64')
df['zan_count'] = df['zan_count'].astype('int64')
df.to_csv(os.path.join(output_path, 'user.txt'), sep='\t', encoding='utf-8', index=False)
