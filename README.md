# KYRS Data
### 数据类型
```python
typedict_content = {'content_id': int, 'content_title': object, 'desc': object, 'topic_id': int, 'topic_title': object,
                    'introduction': object, 'praise_count': int, 'reply_count': int,
                    'forward_count': int,
                    'topic_praise_count': int,
                    'topic_reply_count': int,
                    'topic_forward_count': int, 'inter_list': object}
typedict_inter = {'user_id': int, 'topic_id': int, 'desc': object,
                  'genre_id': int, 'introduction': object, 'comment': int,
                  'forward': int,
                  'thumb': int,
                  'detail': int,
                  'neg': int, 'content_list': object, 'content_num': object}
typedict_user = {'user_id': int, 'sex': int, 'zan_count': int}

```

```shell
nohup python interlist_final.py > inter.log 2>&1
nohup python content_final.py > content.log 2>&1
nohup python user_final.py > user.log 2>&1
```