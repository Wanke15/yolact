import os
import uuid

from flask import Flask, render_template, request, url_for, send_from_directory, flash
from flask_caching import Cache

from predict import MattingService

app = Flask(__name__)
cache = Cache(app, config={'CACHE_TYPE': 'simple'})
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'
app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'upload')
app.config['RESULTS_FOLDER'] = os.path.join(os.getcwd(), 'results')

service = MattingService()


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)


@app.route('/results/<filename>')
def results_file(filename):
    return send_from_directory(app.config['RESULTS_FOLDER'],
                               filename)


# 设置允许的文件格式
ALLOWED_EXTENSIONS = ['png', 'jpg', 'JPG', 'PNG', 'bmp']


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']

        if not file.filename:
            flash('未提交图片！请选择一张下列格式的图片提交：{}'.format(ALLOWED_EXTENSIONS))
            return render_template("index.html")
        elif not allowed_file(file.filename):
            flash('提交的图片不合法！支持的图片格式有：{}'.format(ALLOWED_EXTENSIONS))
            return render_template("index.html")
        threshold = request.form.get("threshold", "0.8")
        threshold = float(threshold) if threshold else 0.8
        top_k = request.form.get("top_k", "1")
        top_k = int(top_k) if top_k else 1

        # 获取缓存
        cached_res = cache.get("{}:{}:{}".format(top_k, threshold, file.filename))
        if cached_res:
            source_img_url, result_img_url = cached_res.split(':')
            cache.set(file.filename, "{}:{}".format(source_img_url, result_img_url))
            return render_template("index.html", source_img=source_img_url, res_img=result_img_url)

        _image_name = "{}.png".format(uuid.uuid4())
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], _image_name)
        file.save(upload_path)
        source_img_url = url_for('uploaded_file', filename=_image_name)
        result_image = service.process(upload_path, top_k, threshold)
        result_img_url = url_for('results_file', filename=result_image)

        # 设置缓存
        cache.set("{}:{}:{}".format(top_k, threshold, file.filename), "{}:{}".format(source_img_url, result_img_url))

        return render_template("index.html", source_img=source_img_url, res_img=result_img_url)
    else:
        return render_template("index.html")


if __name__ == "__main__":
    app.run(port=8888, host='0.0.0.0')
