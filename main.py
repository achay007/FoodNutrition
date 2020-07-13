import tensorflow
import cv2
from flask import Flask, render_template, request
from io import BytesIO
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array
from tensorflow.keras.models import load_model
import os
from PIL import Image
import numpy as np
from base64 import b64encode
from tensorflow.keras.applications.inception_v3 import preprocess_input

from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from tensorflow.python.keras.saving.model_config import model_from_json
from wtforms import SubmitField

# code which helps initialize our server
app = Flask(__name__)
app.config['SECRET_KEY'] = 'any secret key'

bootstrap = Bootstrap(app)

saved_model = load_model('models/model2.h5')

#json_file = open('models/food101_final_modelo.json', 'r')
#loaded_model_json = json_file.read()
#json_file.close()
#loaded_model = model_from_json(loaded_model_json)
# load weights into new model
#loaded_model.load_weights("models/food101_final_model.h5")
#print("Loaded model from disk")


# saved_model._make_predict_function()


class UploadForm(FlaskForm):
    photo = FileField('Upload an image', validators=[FileAllowed(['jpg', 'png', 'jpeg'], u'Image only!'),
                                                     FileRequired(u'File was empty!')])
    submit = SubmitField(u'Predict')


def preprocess(img):
    width, height = img.shape[0], img.shape[1]
    img = image.array_to_img(img, scale=False)

    desired_width, desired_height = 224, 224

    if width < desired_width:
        desired_width = width
    start_x = np.maximum(0, int((width - desired_width) / 2))

    img = img.crop((start_x, np.maximum(0, height - desired_height), start_x + desired_width, height))
    img = img.resize((224, 224))

    img = image.img_to_array(img)
    return img / 255.


@app.route('/', methods=['GET', 'POST'])
def predict():
    form = UploadForm()
    if form.validate_on_submit():
        print(form.photo.data)

        image_stream = form.photo.data.stream
        original_img = Image.open(image_stream)
        img = image.load_img(image_stream, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        images = np.vstack([x])
        #original_img = image.load_img(image_stream, target_size=(229, 229, 3))
        #img = image.img_to_array(original_img)
        #img = img.astype(np.float32)
        #img = image.load_img(img, target_size=(224, 244))
        #img = image.img_to_array(img)
        #img = np.expand_dims(img, axis=0)
        #img = preprocess_input(img)
        #img = np.expand_dims(img, axis=0)
        #img = cv2.resize(cv2.cvtColor(cv2.imread(original_img, 1), cv2.COLOR_BGR2RGB), (224, 224))
        #img = np.expand_dims(cv2.imread(original_img, 1), axis=0)
        ##for i in range(3):
          #img[:, :, :, i] = (img[:, :, :, i] - MEAN[i]) / STD[i]

        prediction = saved_model.predict(images)
        #prediction = np.round(loaded_model.predict(img))

        result = np.argmax(prediction)

        byteIO = BytesIO()
        original_img.save(byteIO, format=original_img.format)
        byteArr = byteIO.getvalue()
        encoded = b64encode(byteArr)

        return render_template('result.html', result=result, encoded_photo=encoded.decode('ascii'))

    return render_template('index.html', form=form)


if __name__ == '__main__':
    app.run(debug=True)
