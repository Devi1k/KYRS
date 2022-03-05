import gc
import os

import pandas as pd
import warnings
import logging
import json
from functools import reduce

data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dataset')
if not os.path.exists(data_path):
    os.mkdir(data_path)
if not os.path.exists(output_path):
    os.mkdir(output_path)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("content_list")
handler1 = logging.FileHandler("base-log-content.log")
handler1.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s|%(name)-12s-- %(levelname)-8s--%(message)s')
handler1.setFormatter(formatter)
logger.addHandler(handler1)

logger.info(data_path)
logger.info(output_path)
warnings.filterwarnings("ignore")
content = pd.read_csv(os.path.join(data_path, '帖子.csv'), sep=',', error_bad_lines=False, encoding='utf-8', header=0)
topic = pd.read_csv(os.path.join(data_path, '话题.csv'), sep=',', error_bad_lines=False, encoding='utf-8', header=0)
content.rename(columns={'subject_id': 'topic_id'}, inplace=True)
concat = pd.merge(content, topic, on='topic_id', how='left')
df1 = concat.drop(['group_id', 'user_id', 'type', 'topic_type',
                   'status_x', 'common_praise_count',
                   'hug_count', 'touch_count', 'hand_num',
                   'reply_user_count', 'star_count', 'read_count', 'share_count', 'uv_x', 'reason', 'image', 'images',
                   'reply_time',
                   'top_time', 'weight_x', 'week_weight', 'month_weight', 'hot_weight',
                   'is_show', 'is_plan', 'biz_content', 'online_time_x', 'pick_time',
                   'weight_json', 'anonymous', 'link_topic_id', 'linked_topic_count',
                   'create_time_x', 'update_time_x', 'genre_id', 'extra_contents', 'tag', 'is_vote', 'vote',
                   'sort', 'star_num',
                   'topic_num', 'last_pick_time', 'pick_count', 'uv_y', 'pv', 'status_y',
                   'btime', 'etime', 'clock_id', 'weight_y', 'tool_entrance',
                   'tool_entrances', 'online_time_y', 'create_time_y', 'update_time_y'], axis=1)
df1 = df1.rename(columns={'title_x': 'content_title', 'title_y': 'topic_title'})

df = df1[['id', 'content_title', 'desc', 'topic_id', 'topic_title', 'introduction', 'praise_count', 'reply_count',
          'forward_count']]
del df1, content, topic
gc.collect()


def process1(desc):
    try:
        return desc.replace('\n', '').replace('\r', '').replace(' ', '')
    except AttributeError:
        return None


df['desc'] = df['desc'].apply(process1)
df['introduction'] = df['introduction'].apply(process1)
df['content_title'] = df['content_title'].apply(process1)
df['topic_title'] = df['topic_title'].apply(process1)

logger.info('-' * 5 + 'process count' + '-' * 5)
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
logger.info('-' * 5 + 'process event_id' + '-' * 5)
user_act = pd.read_csv(os.path.join(data_path, 'user_action.csv'), sep=',', error_bad_lines=False, encoding='utf-8',
                       header=0)
df3 = user_act.drop(labels=['id',
                            'device_id', 'os', 'os_version', 'version', 'system', 'platform', 'pg_short_url',
                            'log_time', 'cal_dt', 'duration', 'log_id'], axis=1)
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
        # logger.info('list')
        return 0
    if 'content_id' in event_data.keys():
        return event_data['content_id']


df3['event_data'] = df3['event_data'].apply(process)
df3['event_data'] = pd.to_numeric(df3['event_data']).fillna('0').astype('int64')
df3['event_data'].value_counts()
df3['content_id'] = df3['event_data']
df3 = df3.sort_values(by=['user_id', 'created_at'], ascending=True)


df4 = df3[df3['follow'] != 'neg']
df4['inter_list'] = pd.NA
del df3
gc.collect()

logger.info('-' * 5 + 'process inter_list' + '-' * 5)


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
df.to_csv(os.path.join(output_path, 'content_raw.txt'), sep='\t', encoding='utf-8', index=False)
logger.info(df4.dtypes)
logger.info(df.dtypes)

logger.info('-' * 5 + 'process merge' + '-' * 5)
df_filter = df4[df4['content_id'] != 0]
df7 = pd.merge(df, df_filter, on='content_id', how='left')
df7.to_csv(os.path.join(output_path, 'content.txt'), sep='\t', encoding='utf-8', index=False)
