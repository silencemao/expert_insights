import pandas as pd
import random

# 创建数据
user_data = {
    'id': ['10000001', '10000002', '10000003', '10000004', '10000005', '10000006', '10000007', '10000008'],
    'name': ['张三', '李四', '王五', '赵六', '陈胜', '吴广', '苏轼', '赵括'],
    'edu_background': [2, 1, 0, 1, 0, 1, 2, 1],  # 0代表博士，1代表硕士，2代表学士
    'year_after_senior': [5, 8, 3, 10, 5, 2, 6, 10]  # 假设单位为年
}

# 创建DataFrame
user_base_df = pd.DataFrame(user_data)

# 打印DataFrame
print(user_base_df)

user_base_df.to_csv('../data/base_info/user_base_info.csv', index=False)


def generate_random_chinese(length):
    result = ''
    for _ in range(length):
        # 生成一个随机的unicode编码，范围在19968到40869之间（汉字的unicode编码范围）
        char = chr(random.randint(19968, 40869))
        result += char
    return result


def generate_reward_info():
    user_num = user_base_df['id']
    years = ['2021', '2022', '2023']
    reward = {
        'id': [], 'reward_name': [], 'reward_type': [], 'reward_class': [], 'finish_order': [], 'max_order': [],
        'finish_year': []
    }
    for year in years:
        for user in user_num:
            reward['finish_year'].append(year)
            reward['reward_name'].append(generate_random_chinese(random.randint(5, 16)))
            reward['id'].append(user)
            reward['reward_type'].append(random.randint(0, 2))
            reward['reward_class'].append(random.randint(0, 2))

            finish_order = random.randint(1, 20)
            reward['finish_order'].append(finish_order)
            reward['max_order'].append(random.randint(min(finish_order, min(finish_order+1, 20)), 20))
    reward_df = pd.DataFrame(reward)
    print(reward_df)
    reward_df.to_csv('../data/base_info/reward_info.csv', index=False)


def generate_patent_info():
    # 姓名	专利名称	专利类型（0发明 1实用新型）	是否授权（0 否 1是）	完成人排序(从1开始）	授权年份
    user_num = user_base_df['id']
    years = ['2019', '2020', '2021', '2022', '2023']
    patent = {
        'id': [], 'patent_name': [], 'patent_type': [], 'is_auth': [], 'finish_order': [], 'max_order': [], 'finish_year': []
    }
    for year in years:
        for user in user_num:
            patent['id'].append(user)
            patent['patent_name'].append(generate_random_chinese(random.randint(5, 16)))
            patent['patent_type'].append(random.randint(0, 1))

            finish_order = random.randint(1, 20)
            patent['finish_order'].append(finish_order)
            patent['max_order'].append(random.randint(min(finish_order, min(finish_order+1, 20)), 20))

            patent['is_auth'].append(random.randint(0, 1))
            patent['finish_year'].append(year)

    patent_df = pd.DataFrame(patent)
    print(patent_df)
    patent_df.to_csv('../data/base_info/patent_info.csv', index=False)


def generate_book_info():
    # 姓名	著作名称	著作类型(0合著、1专著、2独著)	出版类型(0 百佳图书出版 1 其他)	完成人排序(从1开始）	完成年份
    user_num = user_base_df['id']
    years = ['2019', '2020', '2021', '2022', '2023']
    book = {
        'id': [], 'book_name': [], 'book_type': [], 'publish_type': [], 'finish_order': [], 'max_order': [],
           'finish_year': []
    }
    for year in years:
        for user in user_num:
            book['id'].append(user)
            book['book_name'].append(generate_random_chinese(random.randint(5, 16)))
            book['book_type'].append(random.randint(0, 1))
            book['publish_type'].append(random.randint(1, 1))

            finish_order = random.randint(1, 20)
            book['finish_order'].append(finish_order)
            book['max_order'].append(random.randint(min(finish_order, min(finish_order+1, 20)), 20))

            book['finish_year'].append(year)

    book_df = pd.DataFrame(book)
    print(book_df)
    book_df.to_csv('../data/base_info/book_info.csv', index=False)


def generate_standard_info():
    '''
    人资编码	姓名	标准名称	级别(0国际标准、1国家标准、2行标、3团标、4企标、5地标)		完成人排序（从1开始计数）	完成年份
    :return:
    '''
    user_num = user_base_df['id']
    years = ['2019', '2020', '2021', '2022', '2023']
    standards = {'id': [], 'standard_name': [], 'standard_class': [], 'finish_order': [], 'max_order': [], 'finish_year': []}

    for year in years:
        for user in user_num:
            standards['id'].append(user)
            standards['standard_name'].append(generate_random_chinese(random.randint(8, 16)))
            standards['standard_class'].append(random.randint(0, 5))

            finish_order = random.randint(1, 8)
            standards['finish_order'].append(finish_order)
            standards['max_order'].append(random.randint(min(finish_order, min(finish_order+1, 8)), 8))

            standards['finish_year'].append(year)

    paper_df = pd.DataFrame(standards)
    print(paper_df)
    paper_df.to_csv('../data/base_info/standard_info.csv', index=False)


def generate_paper_info():
    # 姓名	论文名称	收录类型(0SCI、1EI、2其它)	完成人排序(从1开始）	完成年份
    user_num = user_base_df['id']
    years = ['2019', '2020', '2021', '2022', '2023']
    paper = {
        'id': [], 'paper_name': [], 'paper_type': [], 'finish_order': [], 'max_order': [], 'finish_year': []
    }
    for year in years:
        for user in user_num:
            paper['id'].append(user)
            paper['paper_name'].append(generate_random_chinese(random.randint(5, 16)))
            paper['paper_type'].append(random.randint(0, 2))

            finish_order = random.randint(1, 6)
            paper['finish_order'].append(finish_order)
            paper['max_order'].append(random.randint(min(finish_order, min(finish_order+1, 6)), 6))

            paper['finish_year'].append(year)

    paper_df = pd.DataFrame(paper)
    print(paper_df)
    paper_df.to_csv('../data/base_info/paper_info.csv', index=False)


if __name__ == '__main__':
    generate_reward_info()
    generate_patent_info()
    generate_book_info()
    generate_paper_info()
    generate_standard_info()




