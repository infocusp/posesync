import os

import cv2
from flask import render_template, send_from_directory
from werkzeug.utils import secure_filename

from src import run_chain
from src.app import app


@app.route('/')
def upload_form():
    return render_template('upload.html')


@app.route('/', methods=['POST'])
def upload_video():

    ref_file = request.files['ref_file']
    test_file = request.files['test_file']

    ref_filename = secure_filename(ref_file.filename)
    ref_file.save(os.path.join(app.config['UPLOAD_FOLDER'], ref_filename))

    test_filename = secure_filename(test_file.filename)
    test_file.save(os.path.join(app.config['UPLOAD_FOLDER'], test_filename))

    crop_method = request.form['crop_method']
    try:
        display_video_name = run_chain.main(
            os.path.join(app.config['UPLOAD_FOLDER'], ref_filename),
            os.path.join(app.config['UPLOAD_FOLDER'], test_filename),
            os.path.join(
                app.config['UPLOAD_FOLDER'],
                ref_filename.split('.mp4')[0] + '--' + test_filename
                ),
            crop_method=crop_method
            )

    except NameError:
        return render_template(
            'upload.html',
            error='Video file is not appropriate. Please try again.'
            )
    except ValueError:
        return render_template(
            'upload.html',
            error="YOLO couldn't detect bounding box for given video. Please "
                  "try "
                  "again."
            )
    except cv2.error:
        return render_template(
            'upload.html',
            error="Can not convert color from BGR to RGB. Please check the "
                  "input "
                  "frame and try again."
            )
    except:
        return render_template(
            'upload.html', error="Something went wrong. Please try again."
            )

    return render_template(
        'upload.html', filename=os.path.basename(display_video_name),
        crop_method=crop_method
        )


@app.route('/display/<filename>')
def display_video(filename):
    return send_from_directory(
        '../static', 'uploads/' + filename, conditional=True
        )


from flask import request

if __name__ == "__main__":

    app.run(debug=True, threaded=True, host='0.0.0.0')
