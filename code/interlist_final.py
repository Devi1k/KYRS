import gc
import os
import time

import pandas as pd
from functools import reduce
import json
import logging

data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dataset')
if not os.path.exists(data_path):
    os.mkdir(data_path)
if not os.path.exists(output_path):
    os.mkdir(output_path)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("interlist_final")
handler1 = logging.FileHandler("base-log.log")
handler1.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s|%(name)-12s+ %(levelname)-8s++%(message)s')
handler1.setFormatter(formatter)
logger.addHandler(handler1)

logger.info(data_path)
logger.info(output_path)
user_act = pd.read_csv(os.path.join(data_path, 'user_action.csv'), sep=',', error_bad_lines=False, encoding='utf-8',
                       header=0)
df1 = user_act.drop(labels=['id',
                            'device_id', 'os', 'os_version', 'version', 'system', 'platform', 'pg_short_url',
                            'log_time', 'cal_dt', 'duration', 'log_id'], axis=1)
df1.loc[((df1.event_id == 254) | (df1.event_id == 248)), 'follow'] = 'thumb'
df1.loc[((df1.event_id == 257) | (df1.event_id == 249)), 'follow'] = 'comment'
df1.loc[((df1.event_id == 258) | (df1.event_id == 256)), 'follow'] = 'forward'
df1.loc[((df1.event_id == 262) | (df1.event_id == 263)), 'follow'] = 'detail'
df1.loc[((df1.event_id == 264) | (df1.event_id == 310)), 'follow'] = 'detail'
df1['follow'] = df1['follow'].fillna(0)
# user_new['follow'] = user_new['follow'].fillna(0)
df1['follow'] = df1['follow'].replace(0, 'neg')
df1['follow'].value_counts()

# df1['content_id'] = df1.loc[(df1.event_id != 221), 'event_data']
df1['event_data'] = df1['event_data'].apply(json.loads)


def process(event_data):
    if not isinstance(event_data, dict):
        # print('list')
        return 0
    if 'content_id' in event_data.keys():
        return event_data['content_id']


df1['event_data'] = df1['event_data'].apply(process)
df1['event_data'] = pd.to_numeric(df1['event_data']).fillna('0').astype('int64')
df1['event_data'].value_counts()
df1['content_id'] = df1['event_data']

df1 = df1.sort_values(by=['user_id', 'created_at'], ascending=True)

content = pd.read_csv(os.path.join(data_path, '帖子.csv'), sep=',', error_bad_lines=False, encoding='utf-8', header=0)
topic = pd.read_csv(os.path.join(data_path, '话题.csv'), sep=',', error_bad_lines=False, encoding='utf-8', header=0)

gbr = df1.groupby('user_id')  # 用分组函数groupby()进行数据的分组，分组依据为'TYPE'这一属性

df_neg = df1[df1['follow'] == 'neg']
df_pos = df1[df1['follow'] != 'neg']

logger.info('-' * 5 + 'process sample' + '-' * 5)

df_pos.drop_duplicates(subset=['user_id', 'content_id', 'follow'], keep='last', inplace=True)
df_neg.drop_duplicates(subset=['user_id', 'content_id'], keep='last', inplace=True)
df_dic = df_neg['user_id'].value_counts().to_dict()

ratio = len(df_pos) / len(df_neg)
df_sample = dict()
sum = 0
for key, value in df_dic.items():
    v = round(value * ratio)
    df_sample[key] = v
    sum += v


def typicalsamling(group, typicalNDict):
    name = group.name
    n = typicalNDict[name]
    return group.sample(n=n)


df_neg_sample = df_neg.groupby('user_id').apply(typicalsamling, df_sample)

frames = [df_pos, df_neg_sample]
result = pd.concat(frames)
result = result.sort_values(by=['user_id', 'created_at'], ascending=True)

content.rename(columns={'subject_id': 'topic_id', 'id': 'content_id'}, inplace=True)
content = content[['content_id', 'topic_id', 'desc']]
topic = topic[['topic_id', 'genre_id', 'introduction']]
add_content = pd.merge(result, content, on='content_id', how='left')
add_topic = pd.merge(add_content, topic, on='topic_id', how='left')


df_final = add_topic.join(pd.get_dummies(add_topic.follow))
del add_content, add_topic, content, topic, df_neg_sample, df_neg, df_pos, result
gc.collect()
df_final = df_final[
    ['user_id', 'topic_id', 'desc', 'genre_id', 'introduction', 'comment', 'forward', 'thumb', 'detail', 'neg',
     'follow', 'created_at']]

df_final['rep'] = df_final[df_final['follow'] != 'neg'].duplicated(subset=['user_id', 'desc'], keep=False)

df_final['rep'] = df_final['rep'].fillna(False)

start = time.time()


def de_duplicate_comment(flag, user_id, desc):
    if flag is True:
        comment = df_final.loc[((df_final.user_id == user_id) & (df_final.desc == desc)), 'comment'].sum()
        return min(comment, 1)



def de_duplicate_forward(flag, user_id, desc):
    if flag is True:
        forward = df_final.loc[((df_final.user_id == user_id) & (df_final.desc == desc)), 'forward'].sum()
        return min(forward, 1)



def de_duplicate_thumb(flag, user_id, desc):
    if flag is True:
        thumb = df_final.loc[((df_final.user_id == user_id) & (df_final.desc == desc)), 'thumb'].sum()
        return min(thumb, 1)



def de_duplicate_detail(flag, user_id, desc):
    if flag is True:
        detail = df_final.loc[((df_final.user_id == user_id) & (df_final.desc == desc)), 'detail'].sum()
        return min(detail, 1)


logger.info('-' * 5 + 'process comment' + '-' * 5)

df_final['comment'] = df_final.apply(
    lambda row: de_duplicate_comment(row['rep'], row['user_id'], row['desc']), axis=1)
logger.info('-' * 5 + 'process forward' + '-' * 5)

df_final['forward'] = df_final.apply(
    lambda row: de_duplicate_forward(row['rep'], row['user_id'], row['desc']), axis=1)
logger.info('-' * 5 + 'process thumb' + '-' * 5)

df_final['thumb'] = df_final.apply(
    lambda row: de_duplicate_thumb(row['rep'], row['user_id'], row['desc']), axis=1)
logger.info('-' * 5 + 'process detail' + '-' * 5)

df_final['detail'] = df_final.apply(
    lambda row: de_duplicate_detail(row['rep'], row['user_id'], row['desc']), axis=1)
end = time.time()
logger.info('count cost' + str(end - start))
df_final_pos = df_final[df_final['follow'] != 'neg']
df_final_pos.drop_duplicates(subset=['user_id', 'desc'], keep='last', inplace=True)

df_final_neg = df_final[df_final['follow'] == 'neg']

df_finalf = pd.concat([df_final_pos, df_final_neg])
df_finalf.sort_values(by=['user_id', 'created_at'], ascending=True)

df_finalf.drop(labels=['follow', 'created_at', 'rep'], axis=1, inplace=True)
logger.info('-' * 5 + 'process data type' + '-' * 5)

df_finalf['topic_id'] = df_finalf['topic_id'].fillna('0').astype('int64')
df_finalf['genre_id'] = df_finalf['genre_id'].fillna('0').astype('int64')
df_finalf['comment'] = df_finalf['comment'].fillna('0').astype('int64')
df_finalf['forward'] = df_finalf['forward'].fillna('0').astype('int64')
df_finalf['thumb'] = df_finalf['thumb'].fillna('0').astype('int64')
df_finalf['detail'] = df_finalf['detail'].fillna('0').astype('int64')


def process1(desc):
    try:
        return desc.replace('\n', '').replace('\r', '').replace(' ', '')
    except AttributeError:
        return None


df_finalf['desc'] = df_finalf['desc'].apply(process1)
df_finalf['introduction'] = df_finalf['introduction'].apply(process1)

df2 = df1[df1['follow'] != 'neg']
# df2[df2['user_id']==1633329256893853]


logger.info('-' * 5 + 'process content list' + '-' * 5)
df2['content_list'] = pd.NA


def count_content_list(user_id):
    inter = list(df2.loc[(df2.user_id == user_id), 'content_id'])
    func = lambda x, y: x if y in x else x + [y]
    list1 = reduce(func, [[], ] + inter)
    content_list = map(lambda x: str(x), list1)
    str1 = ",".join(list(content_list))
    return str1


def lent(data):
    try:
        list = data.split(",")
        return len(list)
    except AttributeError:
        return 0


df2['content_num'] = df2['content_list'].apply(lent)

df2['content_list'] = df2.apply(lambda row: count_content_list(row['user_id']), axis=1)

logger.info(df_finalf.dtypes)
logger.info(df2.dtypes)
df_finalf.to_csv(os.path.join(output_path, 'content_with_topiccnt.txt'), sep='\t', encoding='utf-8', index=False)
df2.to_csv(os.path.join(output_path, 'content_with_interlist.txt'), sep='\t', encoding='utf-8', index=False)
df_inter1 = pd.merge(df_finalf, df2, on='user_id', how='left')
del df_final, df_finalf, df_final_neg, df_final_pos
gc.collect()
logger.info('-' * 5 + 'process split' + '-' * 5)
bt1 = df_inter1[df_inter1['content_num'] > 1]
bt2 = df_inter1[df_inter1['content_num'] > 2]
bt3 = df_inter1[df_inter1['content_num'] > 3]
bt4 = df_inter1[df_inter1['content_num'] > 4]
bt5 = df_inter1[df_inter1['content_num'] > 5]

bt1.to_csv(os.path.join(output_path, 'bt1.txt'), sep='\t', encoding='utf-8', index=False)
bt2.to_csv(os.path.join(output_path, 'bt2.txt'), sep='\t', encoding='utf-8', index=False)
bt3.to_csv(os.path.join(output_path, 'bt3.txt'), sep='\t', encoding='utf-8', index=False)
bt4.to_csv(os.path.join(output_path, 'bt4.txt'), sep='\t', encoding='utf-8', index=False)
bt5.to_csv(os.path.join(output_path, 'bt5.txt'), sep='\t', encoding='utf-8', index=False)

df3 = df2.drop_duplicates(subset=['user_id'], keep='first')

df4 = df3.drop(labels=['event_id', 'event_data', 'follow', 'content_id', 'rep', 'created_at'], axis=1)
del df2,df3
gc.collect()

def split_dataset(data):
    train_ratio = 0.8
    dev_ratio = 0.1
    train_length = int(len(data) * train_ratio)
    dev_length = int(len(data) * dev_ratio)
    train = data[:train_length]
    train = train.drop(
        labels=['content_list', 'content_num'], axis=1)
    dev = data[train_length:train_length + dev_length]
    dev = dev.drop(
        labels=['content_list', 'content_num'], axis=1)
    test = data[train_length + dev_length:]
    test = test.drop(
        labels=['content_id', 'desc', 'topic_id', 'genre_id', 'introduction', 'comment', 'forward', 'thumb', 'detail',
                'neg', 'created_at', 'content_num'], axis=1)
    return train, dev, test


train_list = []
dev_list = []
test_list = []

train_bt1, dev_bt1, test_bt1 = split_dataset(bt1)
train_bt2, dev_bt2, test_bt2 = split_dataset(bt2)
train_bt3, dev_bt3, test_bt3 = split_dataset(bt3)
train_bt4, dev_bt4, test_bt4 = split_dataset(bt4)
train_bt5, dev_bt5, test_bt5 = split_dataset(bt5)

train_bt1 = train_bt1.drop(train_bt1[train_bt1['user_id'] == train_bt1.iloc[-1]['user_id']].index)
dev_bt1 = dev_bt1.drop(dev_bt1[dev_bt1['user_id'] == dev_bt1.iloc[0]['user_id']].index)
test_bt1 = test_bt1.drop(test_bt1[test_bt1['user_id'] == test_bt1.iloc[0]['user_id']].index)

train_bt2 = train_bt2.drop(train_bt2[train_bt2['user_id'] == train_bt2.iloc[-1]['user_id']].index)
dev_bt2 = dev_bt2.drop(dev_bt2[dev_bt2['user_id'] == dev_bt2.iloc[0]['user_id']].index)
test_bt2 = test_bt2.drop(test_bt2[test_bt2['user_id'] == test_bt2.iloc[0]['user_id']].index)

train_bt3 = train_bt3.drop(train_bt3[train_bt3['user_id'] == train_bt3.iloc[-1]['user_id']].index)
dev_bt3 = dev_bt3.drop(dev_bt3[dev_bt3['user_id'] == dev_bt3.iloc[0]['user_id']].index)
test_bt3 = test_bt3.drop(test_bt3[test_bt3['user_id'] == test_bt3.iloc[0]['user_id']].index)

train_bt4 = train_bt4.drop(train_bt4[train_bt4['user_id'] == train_bt4.iloc[-1]['user_id']].index)
dev_bt4 = dev_bt4.drop(dev_bt4[dev_bt4['user_id'] == dev_bt4.iloc[0]['user_id']].index)
test_bt4 = test_bt4.drop(test_bt4[test_bt4['user_id'] == test_bt4.iloc[0]['user_id']].index)

train_bt5 = train_bt5.drop(train_bt5[train_bt5['user_id'] == train_bt5.iloc[-1]['user_id']].index)
dev_bt5 = dev_bt5.drop(dev_bt5[dev_bt5['user_id'] == dev_bt5.iloc[0]['user_id']].index)
test_bt5 = test_bt5.drop(test_bt5[test_bt5['user_id'] == test_bt5.iloc[0]['user_id']].index)

logger.info('-' * 5 + 'export' + '-' * 5)
train_bt1.to_csv(os.path.join(output_path, 'bt1', 'train.txt'), sep='\t', encoding='utf-8', index=False)
dev_bt1.to_csv(os.path.join(output_path, 'bt1', 'dev.txt'), sep='\t', encoding='utf-8', index=False)
test_bt1.to_csv(os.path.join(output_path, 'bt1', 'test.txt'), sep='\t', encoding='utf-8', index=False)

train_bt2.to_csv(os.path.join(output_path, 'bt2', 'train.txt'), sep='\t', encoding='utf-8', index=False)
dev_bt2.to_csv(os.path.join(output_path, 'bt2', 'dev.txt'), sep='\t', encoding='utf-8', index=False)
test_bt2.to_csv(os.path.join(output_path, 'bt2', 'test.txt'), sep='\t', encoding='utf-8', index=False)

train_bt3.to_csv(os.path.join(output_path, 'bt3', 'train.txt'), sep='\t', encoding='utf-8', index=False)
dev_bt3.to_csv(os.path.join(output_path, 'bt3', 'dev.txt'), sep='\t', encoding='utf-8', index=False)
test_bt3.to_csv(os.path.join(output_path, 'bt3', 'test.txt'), sep='\t', encoding='utf-8', index=False)

train_bt4.to_csv(os.path.join(output_path, 'bt4', 'train.txt'), sep='\t', encoding='utf-8', index=False)
dev_bt4.to_csv(os.path.join(output_path, 'bt4', 'dev.txt'), sep='\t', encoding='utf-8', index=False)
test_bt4.to_csv(os.path.join(output_path, 'bt4', 'test.txt'), sep='\t', encoding='utf-8', index=False)

train_bt5.to_csv(os.path.join(output_path, 'bt5', 'train.txt'), sep='\t', encoding='utf-8', index=False)
dev_bt5.to_csv(os.path.join(output_path, 'bt5', 'dev.txt'), sep='\t', encoding='utf-8', index=False)
test_bt5.to_csv(os.path.join(output_path, 'bt5', 'test.txt'), sep='\t', encoding='utf-8', index=False)
