import os, secrets, csv
from flask import Flask, request, render_template, url_for, flash, redirect, current_app
from form import TestForm, TrainForm
from test import cleanForTest
from train import cleanForTrain
from werkzeug.utils import secure_filename
from models.final import Processing_Test, Training


app=Flask(__name__)
app.config['SECRET_KEY'] = '18256fdc199f95f0cdac2b6ddbae9214'

@app.route('/')
@app.route('/home')
def home():
    return render_template('index.html')

@app.route('/testntrain', methods=['GET','POST'])
def testntrain():
    testForm = TestForm()
    trainForm = TrainForm()
    if request.method == 'POST':
        if testForm.submit():
            f = request.files['testData']
            f.save(os.path.join(os.getcwd(), 'test-uploads', secure_filename(f.filename)))
            cleanForTest()
            if testForm.arch.data:
                transfers,retirements,mdu = Processing_Test(os.path.join(os.getcwd(), 'test-uploads', 'model-input', 'training.csv'), os.path.join(os.getcwd(), 'test-uploads', 'model-input', 'testing.csv'), testForm.arch.data)
            else:
                transfers,retirements,mdu = Processing_Test(os.path.join(os.getcwd(), 'test-uploads', 'model-input', 'training.csv'), os.path.join(os.getcwd(), 'test-uploads', 'model-input', 'testing.csv'))
            os.system('paste ./test-uploads/model-output/pred.csv ./test-uploads/model-input/testing.csv -d "," > ./test-uploads/model-output/result.csv')
            flash("Transfers: "+transfers+" | Retirements: "+retirements+" | MDU: "+mdu, "success")
            return redirect(url_for('results', category='Transfers'))
        if trainForm.submit():
            f = request.files['testData']
            f.save(os.path.join(os.getcwd(), 'train-uploads', secure_filename(f.filename)))
            cleanForTrain()
            return redirect(url_for('results', category='Transfers',transfers=transfers,retirements=retirements,mdu=mdu))
    return render_template('testntrain.html', testForm=testForm, trainForm=trainForm)

@app.route('/<string:category>', methods=['GET'])
def results(category,transfers,retirements,mdu):
    with open(os.path.join(os.getcwd(),'test-uploads', 'model-output', 'result.csv'),'r') as csvfile:
        reader = csv.DictReader(csvfile, fieldnames=["Label","Subject", "Date", "Sender", "Body", "Body_Unformatted"])
        return render_template('results.html', category=category, reader=reader, transfers=transfers,retirements=retirements,mdu=mdu)

if __name__=="__main__":
    app.run(debug=True)
