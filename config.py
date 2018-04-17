from flask import Flask
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.environ['UPLOAD_FOLDER']
app.config['ALLOWED_EXTENSIONS'] = set(['png', 'jpg', 'jpeg'])
app.config['DEBUG'] = os.environ['DEBUG']
