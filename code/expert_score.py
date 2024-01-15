import pandas as pd


def _read_data():
    user_base_df = pd.read_csv('../data/base_info/user_base_info.csv')
    patent_df = pd.read_csv('../data/patent_info.csv')
    reward_df = pd.read_csv('../data/base_info/reward_info.csv')
    paper_df = pd.read_csv('../data/base_info/paper_info.csv')
    book_df = pd.read_csv('../data/base_info/book_info.csv')

    return user_base_df, patent_df, reward_df, paper_df, book_df


def _user_base_score(user_base_df):
    edu_config = {0: 300, 1: 210, 2: 160}
    user_base_df['edu_score'] = user_base_df['edu_background'].apply(lambda x: edu_config[x])


def _calculate_score():
    user_base_df, patent_df, reward_df, paper_df, book_df = _read_data()
    pass


