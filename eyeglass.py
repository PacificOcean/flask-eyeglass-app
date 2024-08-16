import os
from flask import Flask, request, redirect, render_template, flash, url_for
from werkzeug.utils import secure_filename
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing import image

import numpy as np

UPLOAD_FOLDER = "./static/uploads"
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

model = load_model('./eyeglass_model.h5') #学習済みモデルをロード
img_height, img_width = 128, 128

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('ファイルがありません')
            return redirect(request.url)

        file = request.files['file']

        if file.filename == '':
            flash('ファイルがありません')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(UPLOAD_FOLDER, filename))
            filepath = os.path.join(UPLOAD_FOLDER, filename)

            #受け取った画像を読み込み、np形式に変換
            img = image.load_img(filepath, target_size=(img_height, img_width))
            img = image.img_to_array(img)
            img = np.expand_dims(img, axis=0)
            
            #変換したデータをモデルに渡して予測する
            pred = model.predict(img)[0][0]
            if pred >= 0.5:
                pred_answer = "これは '眼鏡あり' です"
                score = f"{pred:.2f}"  # 予測スコアを小数点以下2桁で表示
            else:
                pred_answer = "これは '眼鏡なし' です"
                score = f"{1 - pred:.2f}"

            return render_template("index.html", answer=pred_answer, score=score, image_url=url_for('static', filename=f'uploads/{filename}'))
            # return render_template("index.html", answer=pred_answer, score=score, image_url=f'uploads/{filename}')

    return render_template("index.html",answer="")


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8080))
    app.run(host ='0.0.0.0',port = port)