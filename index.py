# Using flask to make an api
# import necessary libraries and functions
from flask import Flask, jsonify, request


import tensorflow as tf
import numpy as np
import scipy.io
import os
import sys
import argparse
sys.path.append('../')
from model import Attention_mask, MTTS_CAN
import h5py
import matplotlib.pyplot as plt
from scipy.signal import butter
from inference_preprocess import preprocess_raw_video, detrend
from werkzeug.utils import secure_filename
import os


UPLOAD_FOLDER = 'uploads'

def predict_vitals(video_path):
    img_rows = 36
    img_cols = 36
    frame_depth = 10
    model_checkpoint = 'mtts_can.hdf5'
    batch_size = 100
    fs = 30
    sample_data_path = video_path

    dXsub = preprocess_raw_video(sample_data_path, dim=36)
    # print('dXsub shape', dXsub.shape)

    dXsub_len = (dXsub.shape[0] // frame_depth)  * frame_depth
    dXsub = dXsub[:dXsub_len, :, :, :]

    model = MTTS_CAN(frame_depth, 32, 64, (img_rows, img_cols, 3))
    model.load_weights(model_checkpoint)

    yptest = model.predict((dXsub[:, :, :, :3], dXsub[:, :, :, -3:]), batch_size=batch_size, verbose=1)

    pulse_pred = yptest[0]
    pulse_pred = detrend(np.cumsum(pulse_pred), 100)
    [b_pulse, a_pulse] = butter(1, [0.75 / fs * 2, 2.5 / fs * 2], btype='bandpass')
    pulse_pred = scipy.signal.filtfilt(b_pulse, a_pulse, np.double(pulse_pred))

    resp_pred = yptest[1]
    resp_pred = detrend(np.cumsum(resp_pred), 100)
    [b_resp, a_resp] = butter(1, [0.08 / fs * 2, 0.5 / fs * 2], btype='bandpass')
    resp_pred = scipy.signal.filtfilt(b_resp, a_resp, np.double(resp_pred))

    ########## Plot ##################
    # plt.subplot(211)
    # plt.plot(pulse_pred)
    # plt.title('Pulse Prediction')
    # plt.subplot(212)
    # plt.plot(resp_pred)
    # plt.title('Resp Prediction')
    # plt.show()
    
    return pulse_pred,resp_pred
  
# creating a Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
  
# on the terminal type: curl http://127.0.0.1:5000/
# returns hello world when we use GET.
# returns the data that we send when we use POST.
@app.route('/video_vital', methods=['GET','POST'])
def home():
    if(request.method == 'POST'):
        # print(request.files)
        if 'file' not in request.files:
        #     return "not"
            return jsonify({'pulse_pred': 'if'})
        else:
            file = request.files['file']
            filename = secure_filename(file.filename)
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(video_path)

            # return jsonify({'pulse_pred': 'else'})

        #     return "yes"
            pulse_pred,resp_pred = predict_vitals(video_path)
        # # print(pulse_pred)
            return jsonify({'pulse_pred': list(pulse_pred),'resp_pred':list(resp_pred)})

        # return str(pulse_pred[0])
  
  
# A simple function to calculate the square of a number
# the number to be squared is sent in the URL when we use GET
# on the terminal type: curl http://127.0.0.1:5000 / home / 10
# this returns 100 (square of 10)
@app.route('/home/<int:num>', methods = ['GET'])
def disp(num):
  
    return jsonify({'data': num**2})
  
  
# driver function
if __name__ == '__main__':
  
    # app.run(debug = True)
    app.run(host='0.0.0.0',debug = True)