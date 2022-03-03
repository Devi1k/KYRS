import pandas as pd

user_act = pd.read_csv("../data/user_action.csv", sep=',', error_bad_lines=False, encoding='utf-8', header=0)
df = pd.DataFrame(user_act, columns=['user_id', 'sex', 'zan_count'])
df.to_csv('../dataset/user.txt', sep='\t', encoding='utf-8', index=False)
