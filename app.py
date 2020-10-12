from flask import Flask, render_template, url_for, flash,redirect
from form import LoginForm

app=Flask(__name__)
app.config['SECRET_KEY'] = '18256fdc199f95f0cdac2b6ddbae9214'

@app.route('/', methods=['GET','POST'])
@app.route('/login', methods=['GET','POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        if form.email.data == 'admin@blog.com' and form.password.data == 'password':
            flash('You have been logged in!','success')
            return redirect(url_for('home'))
        else:
            flash('Login Unsuccessful. Please check username and password', 'danger')
    return render_template('login.html', form = form)

@app.route('/emails')
def emails():
    return render_template()

@app.route('/home')
def home():
    return "HOME"

if __name__=="__main__":
    app.run(debug=True)