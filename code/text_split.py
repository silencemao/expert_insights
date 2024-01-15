import jieba
import pandas as pd
import json
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import hanlp
import re
import cfg
from pyhanlp import HanLP


def read_docx(file_path):
    doc = Document(file_path)
    content = []

    for paragraph in doc.paragraphs:
        if paragraph.text.strip():
            text = paragraph.text.strip()
            text = re.sub(r'[a-zA-Z]+', '', text)
            text = re.sub(r'[()（）【】[\]<>《》{}\\/\\\\_\-+*=]', '', text)
            content.append(text)

    return '\n'.join(content)


def remove_stop_word(matrix, array):
    result_matrix = [[element for element in row if element not in array] for row in matrix]
    return result_matrix


def tokenize_doc(doc_content, method='jieba'):
    # 分词、去除停用词
    data = open('../data/中文停用词库.txt', 'r').readlines()
    stop_words = [word.strip() for word in data]

    if method == 'jieba':
        # 使用精确模式分词
        words = jieba.cut(doc_content, cut_all=False)
        # 过滤掉无关字符（空格、回车等）
    else:
        tok_fine = hanlp.load(hanlp.pretrained.tok.FINE_ELECTRA_SMALL_ZH)
        words = tok_fine(doc_content)
        # print("直接提取关键词的结果 ", HanLP.extractKeyword(doc_content, 5))

    print('分词之后 ', words)
    words = [word.strip() for word in words if word.strip()]

    intersection = set(words).intersection(stop_words)
    result = [word for word in words if word not in intersection]
    print('去除停用 ', result)

    result = " ".join(result)
    return result


def calculate_tfidf(doc_content_seg):
    vectorizer = TfidfVectorizer()     # 创建TF-IDF向量化器

    tfidf_matrix = vectorizer.fit_transform([doc_content_seg])  # 对文档进行TF-IDF转换

    # 获取特征词（关键词）
    feature_names = vectorizer.get_feature_names_out()
    # print(feature_names)

    # 根据TF-IDF值排序获取关键词
    keywords_indices = tfidf_matrix.sum(axis=0).argsort()[0, ::-1]

    top_n_keywords = [feature_names[idx] for idx in keywords_indices[0, :5]]  # 取前N个关键词
    top_n_keywords = top_n_keywords[0].tolist()[0]
    print("关键词:", top_n_keywords)  # 输出关键词
    # print(tfidf_matrix.toarray()[0, :])  # 输出TF-IDF矩阵
    # print(vectorizer.get_feature_names())  # 输出特征词

    # df = pd.DataFrame({'tf-idf': tfidf_matrix.toarray()[0, :], 'keywords': vectorizer.get_feature_names()})
    # df.to_csv("../data/tfidf_results.csv", index=False)
    return ",".join(top_n_keywords)


def _get_title_key_words(titles):
    # 假设有多篇文章的标题

    # 分词、去除停用词
    data = open('../data/中文停用词库.txt', 'r').readlines()
    stop_words = [word.strip() for word in data]

    # 分词并抽取关键词
    tokenized_titles = [" ".join(jieba.cut(title)) for title in titles]

    words = [title.split(' ') for title in tokenized_titles]
    words = remove_stop_word(words, stop_words)
    tokenized_titles = [" ".join(item) for item in words]
    print('恢复格式 ', tokenized_titles)

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
    df_result = pd.DataFrame(data=top_keywords_per_article, columns=["title_key_word"])
    df_result["Title"] = titles  # 将文章标题添加到 DataFrame 中

    # 打印结果
    print(df_result)
    return df_result


def _token_seg_main():
    '''
    todo
    1、题目分词之后可以考虑加到关键词里面
    2、hanlp
    '''

    file_type, file_names, key_words = [], [], []

    dir_path = os.path.join(cfg.data_path, '抽取文件')
    for sub_dir in os.listdir(dir_path):
        sub_dir = os.path.join(dir_path, sub_dir)
        print(sub_dir)
        if not os.path.isdir(sub_dir):
            continue

        for file_name in os.listdir(sub_dir):
            if file_name.split('.')[1] not in ["doc", "docx"]:
                continue

            print(file_name)
            file_path = os.path.join(sub_dir, file_name)
            print(file_path)

            # 内容关键词提取
            doc_content = read_docx(file_path)

            doc_content_seg = tokenize_doc(doc_content, method='hanlp')
            print('计算tf-idf')
            top_n_keywords = calculate_tfidf(doc_content_seg)

            # 写入结果
            file_type.append(file_path.split('/')[-2])
            file_names.append(file_path.split('/')[-1].split('.')[0])
            key_words.append(top_n_keywords)
            print("=" * 10, " done")

    # 稳藏关键词提取
    title_key_word_df = _get_title_key_words(file_names)
    print(title_key_word_df)

    df = pd.DataFrame({'file_type': file_type, 'file_names': file_names, 'key_words': key_words,
                       'title_key_word': title_key_word_df['title_key_word']})
    df.to_csv('../data/tocken_result2.csv', index=False)
    print(df)


def _transfer_word2vec_format_():
    from gensim.models import KeyedVectors
    import re
    weight_path = os.path.join(cfg.weight_path, 'sgns.wiki.bigram-char.bz2')
    wv_from_text = KeyedVectors.load_word2vec_format(weight_path, binary=False, encoding="utf8", unicode_errors='ignore')

    chinese_words = []
    for word, index in wv_from_text.key_to_index.items():
        if re.search("[\u4e00-\u9fff]", word):
            chinese_words.append(word)

    res = {}
    for word in chinese_words:
        if word in wv_from_text:
            res[word] = wv_from_text[word].tolist()

    df = pd.DataFrame(list(res.items()), columns=['key_word', 'vector'])
    df.to_csv('../data/data1.csv', index = False, encoding='utf-8')

    f = open('../data/data1.json', 'w', encoding='utf-8')
    json.dump(res, f, ensure_ascii=False)
    f.close()


def _word2vec_():
    from gensim.models import KeyedVectors
    weight_path = os.path.join(cfg.weight_path, 'sgns.wiki.bigram-char.bz2')
    wv_from_text = KeyedVectors.load_word2vec_format(weight_path, binary=False, encoding="utf8", unicode_errors='ignore')

    # wv_from_text = pd.read_csv('../data/data1.csv.csv')

    file_key_words = pd.read_csv('../data/tocken_result1.csv')
    print(file_key_words)

    for ind, row in file_key_words.iterrows():
        if ind > 1:
            break
        file_name_word = tokenize_doc(row['file_names'].split('.')[0], method='hanlp').split(' ')
        print(file_name_word)
        key_words = row['key_words']
        word_vec = []
        for word in key_words.split(','):
            if word in wv_from_text:
                word_vec.append(wv_from_text[word].tolist())
        print(word_vec)

'''
1、分词、抽取关键词
2、构建词向量  -》 构建文章向量 向量拼接 or 均值 or 叠加
3、文章相似度  -》 向量相似度？
4、作者抽取 -》 需要调研 
5、
'''
if __name__ == '__main__':
    _token_seg_main()
    # _word2vec_()
    # _transfer_word2vec_format_()
