from flask import Flask, render_template, request, url_for, redirect
from keras.saving.model_config import model_from_json
from keras.preprocessing.image import *
import os

for path in ['uploads', 'uploads/NORMAL', 'uploads/PNEUMONIA']:
    if not os.path.exists(path):
        os.mkdir(path)

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads/NORMAL'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = ['jpg', 'jpeg']

batch_size = 20
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model_best_weights.h5")
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
test_datagen = ImageDataGenerator(rescale=1. / 255)
test_data_dir = 'uploads'


@app.route('/', methods=['POST', 'GET'])
def home():
    if request.method == 'POST':
        dict = {}
        files = request.files.getlist('file')
        for file in files:
            folder = 'uploads/NORMAL/'
            for the_file in os.listdir(folder):
                file_path = os.path.join(folder, the_file)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    print(e)
            suf = file.filename.split('.')[-1]
            if suf in ALLOWED_EXTENSIONS:
                file.save(os.path.join(UPLOAD_FOLDER, file.filename))
                test_generator = test_datagen.flow_from_directory(test_data_dir, target_size=(150, 150),
                                                                  batch_size=batch_size,
                                                                  class_mode="binary")
                test_accuracy = loaded_model.evaluate_generator(test_generator)
                if test_accuracy[1] == 1.0:
                    dict[file.filename] = 'Здоров'
                else:
                    dict[file.filename] = 'Пневмония'
            else:
                return 'Файл должен иметь расширение jpg или jpeg'
        return render_template('result.html', dict=dict)
    return render_template('index.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False)