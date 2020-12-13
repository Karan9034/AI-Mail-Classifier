from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed, FileRequired
from wtforms import SubmitField, SelectField
from wtforms.validators import DataRequired

class TestForm(FlaskForm):
    testData = FileField('Testing Data:', validators=[FileRequired(), FileAllowed(['zip', 'rar', '7z'], 'zip, rar and 7z files only!')])  
    arch = SelectField('Model:', choices=[('','--select--'),('Bagging','Bagging (Recommended)'),('LightGBM','LightGBM'),('RandomForestClassifier','Random Forest'),('XgBoost', 'XgBoost')])
    submit = SubmitField('Test Your Model')

class TrainForm(FlaskForm):
    trainData = FileField('Training Data:', validators=[FileRequired(), FileAllowed(['zip', 'rar', '7z'], 'zip, rar and 7z files only!')])
    threads = SelectField('Number of Threads(n_jobs):', choices=[('','--select--'), ('','All'),('','1'),('','2'),('','3'),('', '4')])
    depth = SelectField('Max Depth:', choices=[('','--select--'),('','5'),('','20'),('','69'),('', '80')])
    arch = SelectField('Model:', choices=[('','--select--'),('Bagging','Bagging (Recommended)'),('LightGBM','LightGBM'),('RandomForestClassifier','Random Forest'),('XgBoost', 'XgBoost')])
    submit = SubmitField('Retrain Your Model')