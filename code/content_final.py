import gc
import json
import os
import warnings
from functools import reduce

import pandas as pd

data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', '0315')
output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dataset')
if not os.path.exists(data_path):
    os.mkdir(data_path)
if not os.path.exists(output_path):
    os.mkdir(output_path)
from logger import Logger

log = Logger("content").getLogger()
log.info(data_path)
log.info(output_path)
warnings.filterwarnings("ignore")
content = pd.read_csv(os.path.join(data_path, '帖子.csv'), sep=',', encoding='utf-8', header=0)
topic = pd.read_csv(os.path.join(data_path, '话题.csv'), sep=',', encoding='utf-8', header=0)
content.rename(columns={'subject_id': 'topic_id'}, inplace=True)
topic.rename(columns={'id': 'topic_id'}, inplace=True)
concat = pd.merge(content, topic, on='topic_id', how='left')
# df1 = concat.drop(['group_id',  'type', 'topic_type','follow_sort_time','is_subject_follow'
#                    'status_x', 'common_praise_count',
#                    'hug_count', 'touch_count', 'hand_num',
#                    'reply_user_count', 'read_count', 'uv_x', 'reason', 'image', 'images',
#                    'reply_time',
#                    'top_time', 'weight_x', 'week_weight', 'month_weight', 'hot_weight',
#                    'is_show', 'is_plan', 'biz_content', 'online_time_x', 'pick_time',
#                    'weight_json', 'anonymous', 'link_topic_id', 'linked_topic_count',
#                    'update_time_x', 'genre_id', 'extra_contents', 'tag', 'is_vote', 'vote',
#                    'sort', 'star_num',
#                    'topic_num', 'last_pick_time', 'pick_count', 'uv_y', 'pv', 'status_y',
#                    'btime', 'etime', 'clock_id', 'weight_y', 'tool_entrance',
#                    'tool_entrances', 'online_time_y', 'update_time_y'], axis=1)
df1 = concat.rename(
    columns={'title_x': 'content_title', 'title_y': 'topic_title', 'create_time_x': 'content_create_time',
             'create_time_y': 'topic_create_time'})

df = df1[
    ['id', 'content_title', 'desc', 'user_id', 'topic_id', 'topic_title', 'introduction', 'praise_count', 'reply_count',
     'forward_count', 'star_count', 'share_count', 'content_create_time', 'topic_create_time']]
df['topic_create_time'] = df['topic_create_time'].fillna(0).astype('int64')
del df1, content, topic
gc.collect()

log.info(df.dtypes)


def process1(desc):
    try:
        return desc.replace('\n', '').replace('\r', '').replace(' ', '')
    except AttributeError:
        return None


df['desc'] = df['desc'].apply(process1)
df['introduction'] = df['introduction'].apply(process1)
df['content_title'] = df['content_title'].apply(process1)
df['topic_title'] = df['topic_title'].apply(process1)

log.info('-' * 5 + 'process count' + '-' * 5)
df.rename(columns={'id': 'content_id'}, inplace=True)
df['topic_praise_count'] = 0
df['topic_reply_count'] = 0
df['topic_forward_count'] = 0


def count_praise(topic_id):
    topic_praise_count = df.loc[(df.topic_id == topic_id), 'praise_count'].sum()
    return topic_praise_count


def count_reply(topic_id):
    topic_reply_count = df.loc[(df.topic_id == topic_id), 'reply_count'].sum()
    return topic_reply_count


def count_forward(topic_id):
    topic_forward_count = df.loc[(df.topic_id == topic_id), 'forward_count'].sum()
    return topic_forward_count


df['topic_praise_count'] = df.apply(lambda row: count_praise(row['topic_id']), axis=1)
df['topic_reply_count'] = df.apply(lambda row: count_reply(row['topic_id']), axis=1)
df['topic_forward_count'] = df.apply(lambda row: count_forward(row['topic_id']), axis=1)
df.to_csv(os.path.join(output_path, 'content_raw.txt'), sep='\t', encoding='utf-8', index=False)
log.info(df.dtypes)
del df
gc.collect()
log.info('-' * 5 + 'process event_id' + '-' * 5)
user_act = pd.read_csv(os.path.join(data_path, 'user_action.csv'), sep=',', encoding='utf-8',
                       header=0)
df3 = user_act.drop(labels=['id',
                            'device_id', 'idfa', 'os', 'os_version', 'version', 'system', 'platform', 'log_id',
                            'base_uri', 'pg_short_url',
                            'log_time', 'cal_dt', 'os_p', 'url_org', 'phase', 'pg_url'], axis=1)
del user_act
gc.collect()
df3.loc[((df3.event_id == 254) | (df3.event_id == 248)), 'follow'] = 'thumb'
df3.loc[((df3.event_id == 257) | (df3.event_id == 249)), 'follow'] = 'comment'
df3.loc[((df3.event_id == 258) | (df3.event_id == 256)), 'follow'] = 'forward'
df3.loc[((df3.event_id == 262) | (df3.event_id == 263)), 'follow'] = 'detail'
df3.loc[((df3.event_id == 264) | (df3.event_id == 310)), 'follow'] = 'detail'

df3['follow'] = df3['follow'].fillna(0)

df3['follow'] = df3['follow'].replace(0, 'neg')
df3['follow'].value_counts()

df3['event_data'] = df3['event_data'].apply(json.loads)


def process(event_data):
    if not isinstance(event_data, dict):
        # print('list')
        return 0
    if 'content_id' in event_data.keys():
        return int(event_data['content_id'])


df3['event_data'] = df3['event_data'].apply(process)
df3['event_data'] = pd.to_numeric(df3['event_data']).fillna('0').astype('int64')
df3['event_data'].value_counts()
df3['content_id'] = df3['event_data']
df3 = df3.sort_values(by=['user_id', 'created_at'], ascending=True)

df4 = df3[df3['follow'] != 'neg']
del df3
gc.collect()
df4['inter_list'] = pd.NA

log.info('-' * 5 + 'process inter_list' + '-' * 5)


def count_inter_list(content_id):
    inter = list(df4.loc[(df4.content_id == content_id), 'user_id'])
    func = lambda x, y: x if y in x else x + [y]
    list1 = reduce(func, [[], ] + inter)
    inter_list = map(lambda x: str(x), list1)
    str1 = ",".join(list(inter_list))
    return str1


df4['inter_list'] = df4.apply(lambda row: count_inter_list(row['content_id']), axis=1)
df4.drop_duplicates(subset=['content_id'], keep='first', inplace=True)
df4 = df4.drop(labels=['event_id', 'event_data', 'follow', 'user_id', 'created_at'], axis=1)

df4.to_csv(os.path.join(output_path, 'inter_list.txt'), sep='\t', encoding='utf-8', index=False)

log.info(df4.dtypes)

log.info('-' * 5 + 'process merge' + '-' * 5)
df_filter = df4[df4['content_id'] != 0]
typedict_content = {'content_id': int, 'content_title': object, 'desc': object, 'topic_id': int, 'topic_title': object,
                    'introduction': object, 'praise_count': int, 'reply_count': int,
                    'forward_count': int,
                    'topic_praise_count': int,
                    'topic_reply_count': int,
                    'topic_forward_count': int, 'inter_list': object}
df = pd.read_csv(os.path.join(output_path, 'content_raw.txt'), sep='\t', encoding='utf-8',
                 dtype=typedict_content)
df7 = pd.merge(df, df_filter, on='content_id', how='left')
df7.to_csv(os.path.join(output_path, 'content.txt'), sep='\t', encoding='utf-8', index=False)
