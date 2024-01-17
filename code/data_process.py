import pandas as pd
import jieba
import hanlp
from sklearn.feature_extraction.text import TfidfVectorizer
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


def remove_stop_word(matrix, array):
    result_matrix = [[element for element in row if element not in array] for row in matrix]
    return result_matrix


def _get_title_key_words(df):
    # 假设有多篇文章的标题

    # 分词、去除停用词
    data = open('../data/中文停用词库.txt', 'r').readlines()
    stop_words = [word.strip() for word in data]

    titles, paper_ids = df['paper_name'].tolist(), df['paper_id'].tolist()

    # 分词并抽取关键词
    # tokenized_titles = [" ".join(jieba.cut(title)) for title in titles]
    # words = [title.split(' ') for title in tokenized_titles]

    tok_fine = hanlp.load(hanlp.pretrained.tok.FINE_ELECTRA_SMALL_ZH)
    words = [tok_fine(title) for title in titles]
    print(words)

    words = remove_stop_word(words, stop_words)
    print(words)
    tokenized_titles = [" ".join(item) for item in words]
    # print('恢复格式 ', tokenized_titles)

    # 使用 TF-IDF 计算每个词的重要性
    vectorizer = TfidfVectorizer(use_idf=True)
    tfidf_matrix = vectorizer.fit_transform(tokenized_titles)

    # 获取特征词（关键词）
    feature_names = vectorizer.get_feature_names_out()

    # 将 TF-IDF 结果存储为 DataFrame，每篇文章的关键词作为一行
    df_tfidf = pd.DataFrame(data=tfidf_matrix.toarray(), columns=feature_names, index=titles)

    # 对每篇文章的 TF-IDF 权重进行排序，选取权重最高的三个关键词
    top_keywords_per_article = []
    for title in titles:
        sorted_indices = df_tfidf.loc[title].sort_values(ascending=False).index
        top_keywords = sorted_indices[:3].tolist()
        top_keywords_per_article.append(",".join(top_keywords))

    # 将结果存储为 DataFrame
    # df_result = pd.DataFrame(data=top_keywords_per_article, columns=["title_key_word"])
    df_result = pd.DataFrame({'paper_id': paper_ids, 'paper_name': titles, 'key_word': top_keywords_per_article})
    # df_result["Title"] = titles  # 将文章标题添加到 DataFrame 中

    return df_result


def _paper_process_():
    '''
    论文基础数据预处理，

    1、论文添加id、时间处理、期刊类型拆分（0SCI 1EI 2其他）、人名拆分，去除人名中非汉字部分 -》paper_format
    2、题目关键词提取 -》paper_key_word
    paper_id, paper_name, pub_year, journal_name, journal_class, author_name, author_order
    :return:
    '''
    df = pd.read_csv('../data/paper.csv')
    print(df.columns)

    df['paper_name'] = df['paper_name'].apply(lambda x: x.strip())
    df['paper_id'] = 'lw' + (df.index + 1).astype(str).str.zfill(8)

    # 自定义函数
    def label_text(text):
        if 'SCI' in text:
            return 0
        elif 'EI' in text:
            return 1
        else:
            return 2
    df['journal_class1'] = df['journal_class'].apply(lambda x: label_text(x))

    df['pub_date'] = pd.to_datetime(df['pub_date'], errors='coerce')
    df['pub_year'] = ((df['pub_date'].dt.year.fillna(df['register_date'])).fillna(0).astype(int)).astype(str)

    df['author_name'] = df['author_name'].str.replace(r'[^\w\s\u4e00-\u9fff]', ' ')  # 用空格替换除了字母、数字、空格和中文字符之外的字符
    df['author_name'] = df['author_name'].str.replace(r'[a-zA-Z]', '')  # 去除字母
    df['author_name'] = df['author_name'].str.replace(r'\d', '')  # 去除数字
    df['author_name'] = df['author_name'].str.replace(r'\s+', ' ')  # 将多个空格替换为一个空格
    df['author_name'] = df['author_name'].str.strip()  # 去除开头和结尾的空格
    df['author_name'] = df['author_name'].str.split(' ').apply(lambda x: ' '.join(x))  # 将人名按空格连接起来

    print(df.columns)

    # 题目提取关键词
    paper_key_word_df = _get_title_key_words(df)
    print(paper_key_word_df[:5])
    print(paper_key_word_df.to_csv('../data/paper_key_word.csv', index=False))

    cols = ['paper_id', 'paper_name', 'author_name', 'author_company', 'journal_name', 'journal_class', 'journal_class1',
            'pub_year']
    print(df[cols])
    df[cols].to_csv('../data/paper_format.csv', index=False)


def _author_split():
    # 1、在paper_format的基础上，将一篇文章的人名拆分成多个，构建论文-作者的关系，并标记作者排序 -》paper_author_rela
    # 2、将所有作者抽取出俩，去重 然后作者编号 -》author_id
    df = pd.read_csv('../data/paper_format.csv')

    df['author_name'] = df['author_name'].str.split(' ')

    # 使用 explode 方法将列表拆分成多行
    df = df.explode('author_name')

    # 添加 author_order 列，表示作者顺序
    df['author_order'] = df.groupby('paper_id').cumcount() + 1

    # 重新设置索引
    df = df.reset_index(drop=True)

    author_df = df[['author_name']].drop_duplicates()
    author_df['author_id'] = (author_df.index + 1).astype(str).str.zfill(8)
    print(author_df.columns)
    author_df[['author_id', 'author_name']].to_csv('../data/author_id.csv', index=False)

    res_df = pd.merge(df, author_df, on=['author_name'], how='left')
    cols = ['paper_id', 'paper_name', 'author_name', 'author_order', 'author_id', 'author_company', 'journal_name',
            'journal_class', 'journal_class1', 'pub_year']
    res_df[cols].to_csv('../data/paper_author_rela.csv', index=False)


if __name__ == '__main__':
    # _paper_process_()
    _author_split()
