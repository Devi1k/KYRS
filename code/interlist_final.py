import gc
import json
import os
import time
from functools import reduce

import pandas as pd

from logger import Logger

data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dataset')
if not os.path.exists(data_path):
    os.mkdir(data_path)
if not os.path.exists(output_path):
    os.mkdir(output_path)

log = Logger().getLogger()
log.info(data_path)
log.info(output_path)
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

log.info('-' * 5 + 'process sample' + '-' * 5)

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


def de_duplicate_comment(flag, user_id, desc, comment):
    if flag is True:
        comment = df_final.loc[((df_final.user_id == user_id) & (df_final.desc == desc)), 'comment'].sum()
        return int(min(comment, 1))
    else:
        return comment


def de_duplicate_forward(flag, user_id, desc, forward):
    if flag is True:
        forward = df_final.loc[((df_final.user_id == user_id) & (df_final.desc == desc)), 'forward'].sum()
        return int(min(forward, 1))
    else:
        return forward


def de_duplicate_thumb(flag, user_id, desc, thumb):
    if flag is True:
        thumb = df_final.loc[((df_final.user_id == user_id) & (df_final.desc == desc)), 'thumb'].sum()
        return int(min(thumb, 1))
    else:
        return thumb


def de_duplicate_detail(flag, user_id, desc, detail):
    if flag is True:
        detail = df_final.loc[((df_final.user_id == user_id) & (df_final.desc == desc)), 'detail'].sum()
        return int(min(detail, 1))
    else:
        return detail


log.info('-' * 5 + 'process comment' + '-' * 5)

df_final['comment'] = df_final.apply(
    lambda row: de_duplicate_comment(row['rep'], row['user_id'], row['desc'], row['comment']), axis=1)
log.info('-' * 5 + 'process forward' + '-' * 5)

df_final['forward'] = df_final.apply(
    lambda row: de_duplicate_forward(row['rep'], row['user_id'], row['desc'], row['forward']), axis=1)
log.info('-' * 5 + 'process thumb' + '-' * 5)

df_final['thumb'] = df_final.apply(
    lambda row: de_duplicate_thumb(row['rep'], row['user_id'], row['desc'], row['thumb']), axis=1)
log.info('-' * 5 + 'process detail' + '-' * 5)

df_final['detail'] = df_final.apply(
    lambda row: de_duplicate_detail(row['rep'], row['user_id'], row['desc'], row['detail']), axis=1)
end = time.time()
log.info('count cost' + str(end - start))
df_final_pos = df_final[df_final['follow'] != 'neg']
df_final_pos.drop_duplicates(subset=['user_id', 'desc'], keep='last', inplace=True)

df_final_neg = df_final[df_final['follow'] == 'neg']

df_finalf = pd.concat([df_final_pos, df_final_neg])
df_finalf.sort_values(by=['user_id', 'created_at'], ascending=True)

df_finalf.drop(labels=['follow', 'created_at', 'rep'], axis=1, inplace=True)
log.info('-' * 5 + 'process data type' + '-' * 5)

df_finalf['topic_id'] = df_finalf['topic_id'].fillna('0').astype('int64')
df_finalf['genre_id'] = df_finalf['genre_id'].fillna('0').astype('int64')
df_finalf['comment'] = df_finalf['comment'].astype('int64')
df_finalf['forward'] = df_finalf['forward'].astype('int64')
df_finalf['thumb'] = df_finalf['thumb'].astype('int64')
df_finalf['detail'] = df_finalf['detail'].astype('int64')


def process1(desc):
    try:
        return desc.replace('\n', '').replace('\r', '').replace(' ', '')
    except AttributeError:
        return None


df_finalf['desc'] = df_finalf['desc'].apply(process1)
df_finalf['introduction'] = df_finalf['introduction'].apply(process1)

df2 = df1[df1['follow'] != 'neg']
# df2[df2['user_id']==1633329256893853]


log.info('-' * 5 + 'process content list' + '-' * 5)
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


df2['content_list'] = df2.apply(lambda row: count_content_list(row['user_id']), axis=1)
df2['content_num'] = df2['content_list'].apply(lent)

df2_ = df2[['user_id', 'content_list', 'content_num']]

df_finalf.to_csv(os.path.join(output_path, 'content_with_topiccnt.txt'), sep='\t', encoding='utf-8', index=False)
df2_.to_csv(os.path.join(output_path, 'content_with_interlist.txt'), sep='\t', encoding='utf-8', index=False)
log.info(df_finalf.dtypes)
log.info(df2_.dtypes)
