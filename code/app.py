from flask import Flask, render_template
from flask_bootstrap import Bootstrap
from expert_score2 import _main, _get_author_info, _get_author_key_word
from flask import Flask, jsonify


app = Flask(__name__)
bootstrap = Bootstrap(app)

# 模拟一些作者数据
authors = [
    {'author_id': 1, 'author_name': 'Author 1', 'author_score_sum': 100, 'author_paper_cnt': 10},
    {'author_id': 2, 'author_name': 'Author 2', 'author_score_sum': 150, 'author_paper_cnt': 15},
    # 添加更多作者数据...
]


@app.route('/')
def index():
    res_df = _main()
    data = res_df.to_dict(orient='records')
    return render_template('index.html', data=data)


@app.route('/author_info/<author_id>', methods=['GET'])
def author_info(author_id):
    # 作者的论文、在其中的排名、论文的关键词
    res_df = _get_author_info(author_id)
    data = res_df.to_dict(orient='records')
    print(data)
    if data:
        return render_template('author_info.html', data=data)
    else:
        return 'Author not found', 404


@app.route('/author_profile/<author_id>', methods=['GET'])
def author_profile(author_id):
    print('hhhh1  ', author_id, type(author_id))
    key_word = _get_author_key_word(author_id)
    return render_template('author_knowledge_graph.html', author=key_word[0], keywords=key_word[1:])


if __name__ == '__main__':
    app.run(debug=True)
