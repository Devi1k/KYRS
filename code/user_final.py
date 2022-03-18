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

user_act = pd.read_csv(os.path.join(data_path, '用户.csv'), sep=',', error_bad_lines=False, encoding='utf-8', header=0)
df = pd.DataFrame(user_act, columns=['user_id', 'sex', 'zan_count'])
df.to_csv(os.path.join(output_path, 'user.txt'), sep='\t', encoding='utf-8', index=False)
