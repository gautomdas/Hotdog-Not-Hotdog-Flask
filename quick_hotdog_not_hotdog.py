from flask import Flask, render_template, request
from flask import Flask, request, redirect, url_for
from werkzeug.utils import secure_filename

import tensorflow as tf
import os

UPLOAD_FOLDER = 'images'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])



app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def hello_world():
    print(request.method)
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            print('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            print('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            hot = 0
            not_a = 0
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

            image = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            # ^ Path to the image from command line

            image_file = tf.gfile.FastGFile(image, 'rb')
            # ^ Image being read

            data = image_file.read()
            # ^ Data from image file

            # Loads label file, strips off carriage return
            classes = [line.rstrip() for line in tf.gfile.GFile("hot_dog_labels.txt")]

            # Unpersists graph from file
            with tf.gfile.FastGFile("hot_dog_graph.pb", 'rb') as inception_graph:
                definition = tf.GraphDef()
                definition.ParseFromString(inception_graph.read())
                _ = tf.import_graph_def(definition, name='')

            with tf.Session() as session:
                tensor = session.graph.get_tensor_by_name('final_result:0')
                # ^ Feeding data as input and find the first prediction
                result = session.run(tensor, {'DecodeJpeg/contents:0': data})

                top_results = result[0].argsort()[-len(result[0]):][::-1]

                hot = result[0][top_results[0]]*100
                not_a = result[0][top_results[1]]*100
                print(hot)
                if(float(hot)>0.90):
                    col = "#00FF00"
                    verd = "Hotdog"
                else:
                    col = "#FF0000"
                    verd = "Not Hotdog"

            return render_template('index.html', hot_dog = hot, not_hot_dog = not_a, verdict = verd, color=col)
            #return redirect(url_for('uploaded_file', filename=filename))

    return render_template('index.html')

@app.route('/', methods=['GET', 'POST'])
def run_model():

    return render_template('results.html')
from flask import send_from_directory

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

if __name__ == '__main__':
    app.run()
