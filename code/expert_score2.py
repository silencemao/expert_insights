import pandas as pd
import math
from score_cfg import reward_type_cfg, patent_type_cfg, paper_type_cfg, book_type_cfg
pd.set_option('display.max_columns', None)


# 只根据论文计算分数
def _read_data():
    paper_author_rela_df = pd.read_csv('../data/paper_author_rela.csv', dtype={'paper_id':str, 'author_id':str})

    return paper_author_rela_df


def _score_func(finish_order, max_order, base_score, divid_score):
    if max_order == 1:
        return 1
    return round(base_score / (math.pow(base_score / divid_score, 1/(max_order-1)) ** (finish_order-1)), 3)


def _paper_score(paper_df):
    paper_score = []

    max_author_df = paper_df.groupby(['paper_id']).agg(max_author=('author_id', 'count')).reset_index()
    print(max_author_df[:5])

    paper_df = pd.merge(paper_df, max_author_df, on=['paper_id'], how='left')
    paper_df = paper_df.rename(columns={'max_author': 'max_order', 'author_order': 'finish_order'})
    print(paper_df.columns)
    for ind, row in paper_df.iterrows():

        base_score = paper_type_cfg[row['journal_class1']][0]
        divid_score = paper_type_cfg[row['journal_class1']][1]

        # print('base_score ', base_score, divid_score)
        finish_order, max_order = row['finish_order'], row['max_order']
        paper_score.append(_score_func(finish_order, max_order, base_score, divid_score))

    # print(paper_score)
    paper_df['paper_score'] = paper_score
    print(paper_df.columns)

    # print(paper_df[['paper_id', 'paper_name', 'author_id', 'author_name', 'paper_score', 'pub_year']][10:20])

    author_score_df = paper_df.groupby(['author_id', 'author_name']).agg(author_score_sum=('paper_score', 'sum'),
                                                          author_paper_cnt=('paper_id', 'count')).reset_index()
    author_score_df = author_score_df.sort_values(by=['author_score_sum', 'author_paper_cnt'], ascending=[False, False])
    print(author_score_df[:50])
    return paper_df[['paper_id', 'paper_name', 'author_id', 'author_name', 'paper_score', 'pub_year']]


if __name__ == '__main__':
    paper_author_rela_df = _read_data()
    _paper_score(paper_author_rela_df)
