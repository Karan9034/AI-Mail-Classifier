import os
import secrets
from flask import Flask, request, render_template, url_for, flash, redirect, current_app
from form import TestForm, TrainForm
from werkzeug.utils import secure_filename


app=Flask(__name__)
app.config['SECRET_KEY'] = '18256fdc199f95f0cdac2b6ddbae9214'
app.config['UPLOAD_FOLDER'] = '.\\static\\data'

@app.route('/')
@app.route('/home')
def home():
    return render_template('index.html')

@app.route('/testntrain', methods=['GET','POST'])
def testntrain():
    testForm = TestForm()
    trainForm = TrainForm()
    if request.method == 'POST':
        if testForm.validate_on_submit():
            f = request.files
            f.save(secure_filename(f.filename))
            return redirect(url_for('home'))
        if trainForm.validate_on_submit():
            pass
    return render_template('testntrain.html', testForm=testForm, trainForm=trainForm)



if __name__=="__main__":
    app.run(debug=True)