import os

import pandas as pd

from logger import Logger

log = Logger().getLogger()
log.info('-' * 5 + 'process merge & split' + '-' * 5)

output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dataset')

typedict_inter_de_du = {'user_id': int, 'topic_id': int, 'desc': object,
                        'genre_id': int, 'introduction': object, 'comment': int,
                        'forward': int,
                        'thumb': int,
                        'detail': int,
                        'neg': int}

typedict_content_list = {'user_id': int, 'content_list': object, 'content_num': object}

content_list = pd.read_csv(os.path.join(output_path, 'content_with_interlist.txt'), sep='\t', encoding='utf-8',
                           dtype=typedict_content_list)
inter_dedu = pd.read_csv(os.path.join(output_path, 'content_with_topiccnt.txt'), sep='\t', encoding='utf-8',
                         dtype=typedict_inter_de_du)

df_inter1 = pd.merge(inter_dedu, content_list, on='user_id', how='left')
log.info(df_inter1.dtypes)
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
        labels=['desc', 'topic_id', 'genre_id', 'introduction', 'comment', 'forward', 'thumb', 'detail',
                'neg', 'content_num'], axis=1)
    test.drop_duplicates(subset=['user_id'], inplace=True)
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

log.info('-' * 5 + 'export' + '-' * 5)
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
