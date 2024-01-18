import pandas as pd
import math
from score_cfg import reward_type_cfg, patent_type_cfg, paper_type_cfg, book_type_cfg
pd.set_option('display.max_columns', None)


# 只根据论文计算分数
def _read_data():
    paper_author_rela_df = pd.read_csv('../data/paper_author_rela.csv', dtype={'paper_id':str, 'author_id':str})

    return paper_author_rela_df


paper_author_rela_df = _read_data()
paper_key_word_df = pd.read_csv('../data/paper_key_word.csv')


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
    author_score_df['author_score_sum'] = author_score_df['author_score_sum'].round(1)
    author_score_df = author_score_df.sort_values(by=['author_score_sum', 'author_paper_cnt'], ascending=[False, False]).reset_index(drop=True)

    author_score_df['author_rank'] = (author_score_df.index + 1).astype(str)

    # print(author_score_df[:5])
    print(author_score_df.columns)
    return author_score_df[['author_id', 'author_name',  'author_score_sum',  'author_paper_cnt', 'author_rank']]
    # return paper_df[['paper_id', 'paper_name', 'author_id', 'author_name', 'paper_score', 'pub_year']]


def _get_author_info(author_id):

    res_df = paper_author_rela_df[paper_author_rela_df['author_id'] == author_id]
    cols = ['author_id', 'author_name', 'paper_id', 'paper_name', 'author_order', 'journal_name', 'pub_year']
    res_df = res_df[cols]

    paper_key_word_df = pd.read_csv('../data/paper_key_word.csv')

    res_df = pd.merge(res_df, paper_key_word_df, on=['paper_id', 'paper_name'], how='left')
    res_df = res_df.sort_values(by=['pub_year'], ascending=[False])
    res_df = res_df.drop(columns=['pub_year', 'paper_id'])
    print(res_df[:5])
    return res_df


def _get_author_key_word(author_id):
    paper_ids_df = paper_author_rela_df[paper_author_rela_df['author_id'] == author_id][['paper_id', 'author_id', 'author_name']]
    res_df = pd.merge(paper_ids_df, paper_key_word_df, on=['paper_id'], how='left')

    key_words = res_df['key_word'].tolist()
    key_words = [words.split(',') for words in key_words]
    key_words = list(set([word for words in key_words for word in words]))

    key_words.insert(0, res_df['author_name'].tolist()[0])
    print(key_words)
    return key_words


def _main():
    return _paper_score(paper_author_rela_df)


if __name__ == '__main__':
    _main()
    # _get_author_info('00000011')
    _get_author_key_word('00000011')
