import pandas as pd
import math
from score_cfg import reward_type_cfg, patent_type_cfg, paper_type_cfg, book_type_cfg
pd.set_option('display.max_columns', None)


def _read_data():
    user_base_df = pd.read_csv('../data/base_info/user_base_info.csv')
    patent_df = pd.read_csv('../data/base_info/patent_info.csv')
    reward_df = pd.read_csv('../data/base_info/reward_info.csv')
    paper_df = pd.read_csv('../data/base_info/paper_info.csv')
    book_df = pd.read_csv('../data/base_info/book_info.csv')

    return user_base_df, patent_df, reward_df, paper_df, book_df


def _user_base_score(user_base_df):
    edu_config = {0: 300, 1: 210, 2: 160}
    user_base_df['edu_score'] = user_base_df['edu_background'].apply(lambda x: edu_config[x])

    def _after_senior_score(x):
        if x >= 20:
            return 300
        elif x < 5:
            return 0
        else:
            return 300 - (255/15)*(20-x)
    user_base_df['after_senior_score'] = user_base_df['year_after_senior'].apply(lambda x: _after_senior_score(x))
    # print(user_base_df)
    # print(user_base_df.columns)
    return user_base_df[['id', 'name', 'edu_score', 'after_senior_score']]


def _score_func(finish_order, max_order, base_score, divid_score):
    if max_order == 1:
        return 1
    return round(base_score / (math.pow(base_score / divid_score, 1/(max_order-1)) ** (finish_order-1)), 3)


def _reward_score(reward_df):
    # 分数 * math

    reward_score = []
    for ind, row in reward_df.iterrows():

        base_score = reward_type_cfg[row['reward_type']][int(row['reward_class'])][0]
        divid_score = reward_type_cfg[row['reward_type']][int(row['reward_class'])][1]

        # print('base_score ', base_score, divid_score)
        finish_order, max_order = row['finish_order'], row['max_order']
        reward_score.append(_score_func(finish_order, max_order, base_score, divid_score))

    # print(reward_score)
    reward_df['reward_score'] = reward_score

    # print(reward_df)
    return reward_df[['id', 'reward_name', 'reward_score', 'finish_year']]


def _patent_score(patent_df):
    patent_score = []
    for ind, row in patent_df.iterrows():

        base_score = patent_type_cfg[row['patent_type']][0]
        divid_score = patent_type_cfg[row['patent_type']][1]

        # print('base_score ', base_score, divid_score)
        finish_order, max_order = row['finish_order'], row['max_order']
        patent_score.append(_score_func(finish_order, max_order, base_score, divid_score))

    # print(patent_score)
    patent_df['patent_score'] = patent_score

    # print(patent_df)
    return patent_df[['id', 'patent_name', 'patent_score', 'finish_year']]


def _book_score(book_df):
    book_score = []
    for ind, row in book_df.iterrows():

        base_score = book_type_cfg[row['book_type']][0]
        divid_score = book_type_cfg[row['book_type']][1]

        # print('base_score ', base_score, divid_score)
        finish_order, max_order = row['finish_order'], row['max_order']
        book_score.append(_score_func(finish_order, max_order, base_score, divid_score))

    # print(book_score)
    book_df['book_score'] = book_score

    # print(book_df)
    return book_df[['id', 'book_name', 'book_score', 'finish_year']]


def _paper_score(paper_df):
    paper_score = []
    for ind, row in paper_df.iterrows():

        base_score = paper_type_cfg[row['paper_type']][0]
        divid_score = paper_type_cfg[row['paper_type']][1]

        # print('base_score ', base_score, divid_score)
        finish_order, max_order = row['finish_order'], row['max_order']
        paper_score.append(_score_func(finish_order, max_order, base_score, divid_score))

    # print(paper_score)
    paper_df['paper_score'] = paper_score

    # print(paper_df)
    return paper_df[['id', 'paper_name', 'paper_score', 'finish_year']]


def _calculate_score():
    user_base_df, patent_df, reward_df, paper_df, book_df = _read_data()
    user_base_score_df = _user_base_score(user_base_df)
    reward_score_df = _reward_score(reward_df)
    patent_score_df = _patent_score(patent_df)
    book_score_df = _book_score(book_df)
    paper_score_df = _paper_score(paper_df)

    reward_sum_df = reward_score_df.groupby(['id', 'finish_year']).agg(reward_score=('reward_score', 'sum')).reset_index()
    patent_sum_df = patent_score_df.groupby(['id', 'finish_year']).agg(patent_score=('patent_score', 'sum')).reset_index()
    book_sum_df = book_score_df.groupby(['id', 'finish_year']).agg(book_score=('book_score', 'sum')).reset_index()
    paper_sum_df = paper_score_df.groupby(['id', 'finish_year']).agg(paper_score=('paper_score', 'sum')).reset_index()

    print(paper_sum_df, paper_sum_df.columns)

    res_df = user_base_score_df.merge(reward_sum_df, on=['id'], how='left')\
                               .merge(patent_sum_df, on=['id', 'finish_year'], how='left')\
                               .merge(book_sum_df, on=['id', 'finish_year'], how='left')\
                               .merge(paper_sum_df, on=['id', 'finish_year'], how='left')

    res_df['sum_score'] = res_df['edu_score'] + res_df['after_senior_score'] + res_df['reward_score'] \
                          + res_df['patent_score'] + res_df['book_score'] + res_df['paper_score']

    res_df = res_df.sort_values(by=['finish_year', 'sum_score'], ascending=[False, False])
    print(res_df)
    print(res_df.columns)


if __name__ == '__main__':
    _calculate_score()


