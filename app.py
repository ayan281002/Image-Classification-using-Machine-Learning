import numpy as np
from flask import Flask, render_template, request, flash
import os
import pickle
import cv2
from werkzeug.utils import secure_filename
import shutil


# HOG parameters
winSize = 32
blockSize = 12
blockStride = 4
cellSize = 4
nbins = 9
derivAperture = 1
winSigma = -1.
histogramNormType = 0
L2HysThreshold = 0.2
gammaCorrection = True
nlevels = 64
signedGradient = True
hog = cv2.HOGDescriptor((winSize, winSize), (blockSize, blockSize), (blockStride, blockStride), (cellSize, cellSize),
                        nbins, derivAperture, winSigma, histogramNormType, L2HysThreshold, gammaCorrection, nlevels, signedGradient)

# Calculate Hog features


def calcHOG(image):
    hogDescriptor = hog.compute(image)
    hogDescriptor = np.squeeze(hogDescriptor)
    return hogDescriptor

# Load the models from pickle files


pca = pickle.load(open('pca_model.pkl', 'rb'))
svm = pickle.load(open('svm_model.pkl', 'rb'))

# Find image class


def classifyImage(testImage):

    # testImage = convertImage(testImage)
    testHogDescriptor = calcHOG(testImage)
    testHogProjected = pca.transform(testHogDescriptor.reshape(1, -1))
    testResponse = svm.predict(testHogProjected)

    return testResponse


label = {
    0: 'airplane',
    1: 'automobile',
    2: 'bird',
    3: 'cat',
    4: 'deer',
    5: 'dog',
    6: 'frog',
    7: 'horse',
    8: 'ship',
    9: 'truck'
}
img_extensions = ['tiff', 'bmp', 'pjp', 'apng', 'gif', 'svg', 'png', 'xbm',
                  'dib', 'jxl', 'jpeg', 'svgz', 'jpg', 'webp', 'ico', 'tif', 'pjpeg', 'avif']


# Loading App

app = Flask(__name__)
app.secret_key = '3d6f45a5fc12445dbac2f59c3b6c7cb1'

UPLOAD_FOLDER = os.path.join('static/tempFiles')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/',  methods=("POST", "GET"))
def uploadFile():
    if request.method == 'POST':
        if request.form['btn'] == 'doTest':
            return render_template('testPage.html')
        elif request.form['btn'] == 'showDetail':
            return render_template('projectDetails.html')
        elif request.form['btn'] == 'bck':
            return render_template('index.html')
        elif request.form['btn'] == 'submit':
            try:
                uploaded_img = request.files['uploaded-file']
                filename = secure_filename(uploaded_img.filename)
                extension = filename.split('.')[-1]
                if extension in img_extensions:
                    uploaded_img.save(os.path.join(
                        app.config['UPLOAD_FOLDER'], filename))
                    path = os.path.join('static/tempFiles', filename)
                    pth = 'tempFiles/'+filename
                    testImage = cv2.imread(path)
                    testImage = cv2.resize(testImage, (32, 32))
                    id = classifyImage(testImage)
                    id = id[0]
                    output = label.get(id)
                    render = render_template(
                        'prediction.html', class_image='Image classified as: '+output, p=pth)
                    return render
                elif extension == '':
                    flash('No file selected')
                    return render_template('testPage.html')
                else:
                    flash('Input file is not image')
                    return render_template('testPage.html')
            except:
                flash('No file uploaded')
                return render_template('testPage.html')
        elif request.form['btn'] == 'return':
            path = 'static/tempFiles'
            shutil.rmtree(path)
            os.mkdir(path=path)
            return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=80)
