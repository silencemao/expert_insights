import jieba
import pandas as pd
import json
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import hanlp
import re
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


def tokenize_doc(doc_content, method='jieba'):
    if method == 'jieba':
        # 使用精确模式分词
        words = jieba.cut(doc_content, cut_all=False)
        # 过滤掉无关字符（空格、回车等）
        words = [word.strip() for word in words if word.strip()]
        # 将分词结果拼接为字符串
        result = " ".join(words)
    else:
        tok_fine = hanlp.load(hanlp.pretrained.tok.FINE_ELECTRA_SMALL_ZH)
        words = tok_fine(doc_content)
        result = " ".join(words)

        # print("直接提取关键词的结果 ", HanLP.extractKeyword(doc_content, 5))

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


def _token_seg_main():
    '''
    todo
    1、题目分词之后可以考虑加到关键词里面
    2、hanlp
    '''

    file_type, file_names, key_words = [], [], []

    dir_path = "~/Desktop/抽取文件/"
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

            doc_content = read_docx(file_path)

            doc_content_seg = tokenize_doc(doc_content, method='hanlp')
            # print('分词完毕', doc_content_seg)
            print('计算tf-idf')
            top_n_keywords = calculate_tfidf(doc_content_seg)

            # 写入结果
            file_type.append(file_path.split('/')[-2])
            file_names.append(file_path.split('/')[-1])
            key_words.append(top_n_keywords)
            print("=" * 10, " done")

    df = pd.DataFrame({'file_type': file_type, 'file_names': file_names, 'key_words': key_words})
    df.to_csv('../data/tocken_result1.csv', index=False)
    print(df)


if __name__ == '__main__':
    _token_seg_main()
