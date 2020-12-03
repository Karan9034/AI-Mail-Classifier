from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed, FileRequired
from wtforms import SubmitField, SelectField
from wtforms.validators import DataRequired

class TestForm(FlaskForm):
    testData = FileField('Testing Data:', validators=[FileRequired(), FileAllowed(['zip'])])
    submit = SubmitField('Test Your Model')

class TrainForm(FlaskForm):
    trainData = FileField('Training Data:', validators=[FileRequired(), FileAllowed(['zip'])])
    learnRate = SelectField('Learn Rate:', choices=[('','--select--'),('','1e-5'),('','1e-4'),('','0.5e-5'),('', 'Custom')])
    epochs = SelectField('Epochs:', choices=[('','--select--'),('','10'),('','5'),('','15'),('', 'Custom')])
    arch = SelectField('Model Archiecture:', choices=[('','--select--'),('','XLNet (Recommended)'),('','T5'),('','BERT'),('', 'GPT2')])
    submit = SubmitField('Retrain Your Model')