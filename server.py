import os
from flask import jsonify, request, redirect, url_for, Response, render_template
from werkzeug.utils import secure_filename
import traceback
from config import app
from exception import GenericException
import json
from functools import wraps
from image_classifier import ImageClassifierCNN


@app.context_processor
def override_url_for():
    return dict(url_for=dated_url_for)


def dated_url_for(endpoint, **values):
    if endpoint == 'static':
        filename = values.get('filename', None)
        if filename:
            file_path = os.path.join(app.root_path,
                                     endpoint, filename)
            values['q'] = int(os.stat(file_path).st_mtime)
    return url_for(endpoint, **values)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


@app.errorhandler(GenericException)
def error_handler(error):
    if app.config['DEBUG']:
        traceback.print_exc()
    response = {
        "status": error.status_code,
        "msg": error.message
    }
    return Response(json.dumps(response), status=error.status_code, mimetype='application/json')


def exception_helper(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            response = f(*args, **kwargs)
            return response
        except Exception as e:
            error = GenericException('See trace message for futher info')
            if(hasattr(e, 'status_code')):
                error = GenericException('See trace message for futher info', status_code=e.status_code)
            return error_handler(error)
    return decorated_function


@app.route('/', methods=['GET'], strict_slashes=False)
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'], strict_slashes=False)
@exception_helper
def predict():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            raise GenericException('No file part')
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            raise GenericException('No selected file')
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            if not os.path.exists(app.config['UPLOAD_FOLDER']):
                os.makedirs(app.config['UPLOAD_FOLDER'])
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            image_classifier = ImageClassifierCNN(classes=['dogs', 'cats'])
            result = image_classifier.predict(os.path.join(app.config['UPLOAD_FOLDER'] + "/" + str(filename)))
            response = {
                "status": 200,
                "payload": {
                    "predicted": result,
                    "path": app.config['UPLOAD_FOLDER'] + "/" + str(filename)
                }
            }
            return Response(json.dumps(response), status=200, mimetype='application/json')


@app.route('/predict', methods=['GET'], strict_slashes=False)
@exception_helper
def predict_():
    response = {
        "status": 200,
        "payload": {
            "method": "POST",
            "key": "file",
            "enctype": "multipart/form-data",
            "allowed_extensions": list(app.config["ALLOWED_EXTENSIONS"])
        }
    }
    return Response(json.dumps(response), status=200, mimetype='application/json')


if __name__ == '__main__':
    app.run(debug=app.config['DEBUG'])
