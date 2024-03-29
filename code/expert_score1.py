import pandas as pd
import math

import score_cfg
from score_cfg import reward_type_cfg, patent_type_cfg, paper_type_cfg, book_type_cfg, score_weight, edu_config, standard_cfg
pd.set_option('display.max_columns', None)


# 第一版规则计算分数
def _read_data():
    user_base_df = pd.read_csv('../data/base_info/user_base_info.csv')
    patent_df = pd.read_csv('../data/base_info/patent_info.csv')
    reward_df = pd.read_csv('../data/base_info/reward_info.csv')
    paper_df = pd.read_csv('../data/base_info/paper_info.csv')
    book_df = pd.read_csv('../data/base_info/book_info.csv')
    standard_df = pd.read_csv('../data/base_info/standard_info.csv')

    return user_base_df, patent_df, reward_df, paper_df, book_df, standard_df


def _user_base_score(user_base_df):
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
    return round(base_score / (math.pow(base_score / divid_score, 1/(max_order-1)) ** (finish_order-1)), 2)


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

        base_score = patent_type_cfg[row['patent_type']][1][0]
        divid_score = patent_type_cfg[row['patent_type']][1][1]

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


def _standard_score(standard_df):
    # id,standard_name,standard_class,finish_order,max_order,finish_year
    paper_score = []
    for ind, row in standard_df.iterrows():

        base_score = standard_cfg[row['standard_class']][0]
        divid_score = standard_cfg[row['standard_class']][1]

        # print('base_score ', base_score, divid_score)
        finish_order, max_order = row['finish_order'], row['max_order']
        paper_score.append(_score_func(finish_order, max_order, base_score, divid_score))

    # print(paper_score)
    standard_df['standard_score'] = paper_score

    return standard_df[['id', 'standard_name', 'standard_score', 'finish_year']]


def _calculate_score():
    user_base_df, patent_df, reward_df, paper_df, book_df, standard_df = _read_data()
    user_base_score_df = _user_base_score(user_base_df)
    reward_score_df = _reward_score(reward_df)
    patent_score_df = _patent_score(patent_df)
    book_score_df = _book_score(book_df)
    paper_score_df = _paper_score(paper_df)
    standard_score_df = _standard_score(standard_df)

    reward_sum_df = reward_score_df.groupby(['id', 'finish_year']).agg(reward_score=('reward_score', 'sum')).reset_index()
    patent_sum_df = patent_score_df.groupby(['id', 'finish_year']).agg(patent_score=('patent_score', 'sum')).reset_index()
    book_sum_df = book_score_df.groupby(['id', 'finish_year']).agg(book_score=('book_score', 'sum')).reset_index()
    paper_sum_df = paper_score_df.groupby(['id', 'finish_year']).agg(paper_score=('paper_score', 'sum')).reset_index()
    standard_sum_df = standard_score_df.groupby(['id', 'finish_year']).agg(standard_score=('standard_score', 'sum')).reset_index()

    res_df = user_base_score_df.merge(reward_sum_df, on=['id'], how='left')\
                               .merge(patent_sum_df, on=['id', 'finish_year'], how='left')\
                               .merge(book_sum_df, on=['id', 'finish_year'], how='left')\
                               .merge(paper_sum_df, on=['id', 'finish_year'], how='left')\
                               .merge(standard_sum_df, on=['id', 'finish_year'], how='left')

    edu_weight, senior_weight, reward_weight, patent_weight, book_and_paper_weight = score_weight['base_score'], score_weight['after_senior_score'], score_weight['reward_score'], score_weight['patent_score'], score_weight['book_and_paper']
    res_df['sum_score'] = res_df['edu_score'] * edu_weight + res_df['after_senior_score'] * senior_weight + \
                          res_df['reward_score'] * reward_weight + res_df['patent_score'] * patent_weight \
                          + (res_df['book_score'] + res_df['paper_score'] + res_df['standard_score']) * book_and_paper_weight

    res_df = res_df.sort_values(by=['finish_year', 'sum_score'], ascending=[False, False])
    print(res_df)
    print(res_df.columns)

    res_df = res_df.groupby(['id', 'name']).agg(sum_score=('sum_score', 'sum')).reset_index()
    res_df = res_df.sort_values(by=['sum_score'], ascending=False)
    res_df['rank'] = res_df.reset_index().index + 1
    print(res_df)


if __name__ == '__main__':
    _calculate_score()


