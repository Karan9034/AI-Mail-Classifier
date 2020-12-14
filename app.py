import os, secrets, csv
from flask import Flask, request, render_template, url_for, flash, redirect, current_app, send_from_directory
from form import TestForm, TrainForm
from test import cleanForTest
from train import cleanForTrain
from werkzeug.utils import secure_filename
from models.final import Processing_Test, Training


app=Flask(__name__)
app.config['SECRET_KEY'] = '18256fdc199f95f0cdac2b6ddbae9214'

@app.route('/')
@app.route('/home')
def home(category="Transfers"):
    with open(os.path.join(os.getcwd(),'test-uploads', 'model-output', 'result.csv'),'r') as csvfile:
        reader = csv.DictReader(csvfile, fieldnames=["Label","Subject", "Date", "Sender", "Body", "Body_Unformatted"])
        return render_template('index.html', category=category, reader=reader)


@app.route('/testntrain', methods=['GET','POST'])
def testntrain():
    testForm = TestForm()
    trainForm = TrainForm()
    if request.method == 'POST':
        if testForm.testSubmit.data:
            f = request.files['testData']
            f.save(os.path.join(os.getcwd(), 'test-uploads', secure_filename(f.filename)))
            cleanForTest()
            if testForm.arch.data:
                transfers,retirements,mdu = Processing_Test(os.path.join(os.getcwd(), 'test-uploads', 'model-input', 'testing.csv'), testForm.arch.data)
            else:
                transfers,retirements,mdu = Processing_Test(os.path.join(os.getcwd(), 'test-uploads', 'model-input', 'testing.csv'))
            os.system('paste ./test-uploads/model-output/pred.csv ./test-uploads/model-input/testing.csv -d "," > ./test-uploads/model-output/result.csv')
            msg="Transfers: "+str(transfers)+" | Retirements: "+str(retirements)+" | MDU: "+str(mdu)
            flash(msg, "success")
            return redirect(url_for('results', category='Transfers'))

        if trainForm.trainSubmit.data:
            f = request.files['trainData']
            f.save(os.path.join(os.getcwd(), 'train-uploads', secure_filename(f.filename)))
            cleanForTrain()
            if trainForm.arch.data:
                score = Training(os.path.join(os.getcwd(), 'test-uploads', 'model-input', 'training.csv'), trainForm.arch.data)
            else:
                score = Training(os.path.join(os.getcwd(), 'test-uploads', 'model-input', 'training.csv'))
            flash('Model successfully retrained | Validation Accuracy: '+ str(score)[:3], 'success')
            return redirect(url_for('testntrain'))

    return render_template('testntrain.html', testForm=testForm, trainForm=trainForm)

@app.route('/<string:category>')
def results(category):
    with open(os.path.join(os.getcwd(),'test-uploads', 'model-output', 'result.csv'),'r') as csvfile:
        reader = csv.DictReader(csvfile, fieldnames=["Label","Subject", "Date", "Sender", "Body", "Body_Unformatted"])
        return render_template('results.html', category=category, reader=reader)

@app.route('/results/download')
def download():
    return send_from_directory('./test-uploads/model-output', 'result.csv')

if __name__=="__main__":
    app.run(debug=True)
