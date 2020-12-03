import os
import secrets
from flask import Flask, request, render_template, url_for, flash, redirect, current_app
from form import TestForm, TrainForm
from werkzeug.utils import secure_filename


app=Flask(__name__)
app.config['SECRET_KEY'] = '18256fdc199f95f0cdac2b6ddbae9214'


@app.route('/', methods=['GET','POST'])
def index():
    testForm = TestForm()
    trainForm = TrainForm()
    if testForm.validate_on_submit():
        f = testForm.testData.data
        filename = secure_filename(f.filename)
        f.save(os.path.join(app.instance_path, 'static/data', filename))
        return redirect(url_for('home'))
    if trainForm.validate_on_submit():
        pass
    return render_template('index.html', testForm=testForm, trainForm=trainForm)


@app.route('/home')
def home():
    return render_template('output.html')


if __name__=="__main__":
    app.run(debug=True)