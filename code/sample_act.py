import gc
import os
import random

import pandas as pd

from logger import Logger

data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', '0315')
output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dataset')
if not os.path.exists(data_path):
    os.mkdir(data_path)
if not os.path.exists(output_path):
    os.mkdir(output_path)

log = Logger("inter").getLogger()
log.info(data_path)
log.info(output_path)
user_act = pd.read_csv(os.path.join(data_path, 'user_action.csv'), sep=',', encoding='utf-8',
                       header=0)
df1 = user_act.drop(labels=['id',
                            'device_id', 'idfa', 'os', 'os_version', 'version', 'system', 'platform', 'log_id',
                            'base_uri', 'pg_short_url',
                            'log_time', 'cal_dt', 'os_p', 'url_org', 'phase', 'pg_url'], axis=1)
del user_act
gc.collect()
user_id = set()
df_u = df1['user_id'].value_counts()
for i, v in df_u.items():
    if v < 10000:
        user_id.add(i)
# %%
user_id = list(user_id)
df5k_id = random.sample(user_id, 5000)
df_all = df1.iloc[0]
for id in user_id:
    df_all = pd.concat([df1[df1['user_id'] == id], df_all])
df_all = df_all.drop(labels=[0], axis=1)
df_all = df_all.dropna()
df_all['user_id'] = df_all['user_id'].astype('int64')
df_all['event_id'] = df_all['event_id'].astype('int64')
df_all['created_at'] = df_all['created_at'].astype('int64')
df_all.to_csv(os.path.join(data_path, 'useract_all.txt'), sep='\t', encoding='utf-8', index=False)
