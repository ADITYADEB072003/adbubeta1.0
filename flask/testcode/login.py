from flask import Flask, render_template, request, redirect, session

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Change this to a secure secret key

# Define a predefined username and password (for demonstration purposes)
USERNAME = 'admin'
PASSWORD = 'password'


@app.route('/')
def home():
    return redirect('/login')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if username == USERNAME and password == PASSWORD:
            session['logged_in'] = True
            return redirect('/index1')
        else:
            return render_template('login.html', message='Invalid credentials. Please try again.')

    return render_template('login.html')


@app.route('/index1')
def index():
    if session.get('logged_in'):
        return render_template('index1.html')
    else:
        return redirect('/login')


@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    return redirect('/login')


if __name__ == '__main__':
    app.run(debug=True)
