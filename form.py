from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed, FileRequired
from wtforms import SubmitField, SelectField
from wtforms.validators import DataRequired

class TestForm(FlaskForm):
    testData = FileField('Testing Data:', validators=[FileRequired(), FileAllowed(['zip', 'rar', '7z'], 'zip, rar and 7z files only!')])  
    arch = SelectField('Model:', choices=[('','--select--'),('Bagging','Bagging (Recommended)'),('LightGBM','LightGBM'),('RandomForestClassifier','Random Forest'),('XgBoost', 'XgBoost')])
    testSubmit = SubmitField('Test Your Model')

class TrainForm(FlaskForm):
    trainData = FileField('Training Data:', validators=[FileRequired(), FileAllowed(['zip', 'rar', '7z'], 'zip, rar and 7z files only!')])
    threads = SelectField('Number of Threads(n_jobs):', choices=[('all','--select--'), ('all','All'),('1','1'),('2','2'),('3','3'),('4', '4')])
    depth = SelectField('Max Depth:', choices=[('40','--select--'),('5','5'),('20','20'),('40','40'),('80', '80')])
    arch = SelectField('Model:', choices=[('Bagging','--select--'),('Bagging','Bagging (Recommended)'),('LightGBM','LightGBM'),('RandomForestClassifier','Random Forest'),('XgBoost', 'XgBoost')])
    trainSubmit = SubmitField('Retrain Your Model')
