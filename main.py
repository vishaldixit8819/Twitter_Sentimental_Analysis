from flask import request
from flask import jsonify
from flask import Flask, render_template
import index
import nltk
nltk.download('popular')

app = Flask(__name__)

@app.route('/')
def my_form():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def my_form_post():
    text = request.form['text']

    score = index.predict(text)
    if(score == 0):
        label = 'This tweet is positive'
    else:
        label = 'This tweet is negative'

    return(render_template('/index.html', variable=label))

if __name__ == "__main__":
    app.run(port='5000', threaded=False, debug=True)