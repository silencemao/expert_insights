import jieba
import pandas as pd
import json
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer


def read_docx(file_path):
    doc = Document(file_path)
    content = []

    for paragraph in doc.paragraphs:
        if paragraph.text.strip():
            content.append(paragraph.text.strip())
    return '\n'.join(content)


def tokenize_doc(doc_content):
    # 使用精确模式分词
    words = jieba.cut(doc_content, cut_all=False)
    # 过滤掉无关字符（空格、回车等）
    words = [word.strip() for word in words if word.strip()]
    # 将分词结果拼接为字符串
    result = " ".join(words)

    return result


def calculate_tfidf(doc_content_seg):
    vectorizer = TfidfVectorizer()     # 创建TF-IDF向量化器

    tfidf_matrix = vectorizer.fit_transform([doc_content_seg])  # 对文档进行TF-IDF转换

    # 获取特征词（关键词）
    feature_names = vectorizer.get_feature_names_out()
    print(feature_names)

    # 根据TF-IDF值排序获取关键词
    keywords_indices = tfidf_matrix.sum(axis=0).argsort()[0, ::-1]

    top_n_keywords = [feature_names[idx] for idx in keywords_indices[0, :5]]  # 取前N个关键词

    print("关键词:", top_n_keywords)  # 输出关键词
    # print(tfidf_matrix.toarray()[0, :])  # 输出TF-IDF矩阵
    # print(vectorizer.get_feature_names())  # 输出特征词

    df = pd.DataFrame({'tf-idf': tfidf_matrix.toarray()[0, :], 'keywords': vectorizer.get_feature_names()})
    df.to_csv("../data/tfidf_results.csv", index=False)


def _token_seg_main():
    file_path = ""
    # tokenize_doc(file_path)
    doc_content = read_docx(file_path)

    print("="*10)
    doc_content_seg = tokenize_doc(doc_content)
    print('分词完毕', doc_content_seg)
    print('计算tf-idf')
    calculate_tfidf(doc_content_seg)


if __name__ == '__main__':
    _token_seg_main()
