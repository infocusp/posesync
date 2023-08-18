import os

from flask import Flask

UPLOAD_FOLDER = 'static/uploads/'
os.makedirs('static/uploads/', exist_ok=True)

app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
