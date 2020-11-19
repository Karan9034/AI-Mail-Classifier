from flask import Flask, render_template, url_for, flash,redirect
from form import TestForm, TrainForm

app=Flask(__name__)
app.config['SECRET_KEY'] = '18256fdc199f95f0cdac2b6ddbae9214'

@app.route('/', methods=['GET','POST'])
def index():
    return render_template('index.html')


@app.route('/home')
def home():
    return render_template('output.html')


if __name__=="__main__":
    app.run(debug=True)