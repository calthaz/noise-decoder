#!/home/yumon/PTCam/PTCam-virtualenv/bin/python3
from flask_socketio import SocketIO
from flask import Flask, render_template, request, Response
from flask import stream_with_context, session, abort, send_file, jsonify
import os
import json

import time
import argparse
import random
import string
from flask_cors import CORS, cross_origin

import logging

import eventlet
#eventlet.monkey_patch()

import predict


app = Flask(__name__)
cors = CORS(app, support_credentials=True)
app.config['CORS_HEADERS'] = 'Content-Type'

socketio = SocketIO(app)#, async_mode='threading')

socketio.init_app(app, cors_allowed_origins="*")

# Set the secret key to some random bytes. Keep this really secret!
app.secret_key = b'S^Sdjk}|_sfn3#@'

logging.getLogger('flask_cors').level = logging.DEBUG

@app.route('/')
@app.route('/index')
def indexPage():
    """Home page."""
    return render_template('index.html')

@socketio.on('connect', namespace='/web')
def connect_web():
    print('[INFO] Web client connected: {}'.format(request.sid))
    #socketio.emit('after connect',  {'data':'Lets dance'}, namespace='/web')


@socketio.on('disconnect', namespace='/web')
def disconnect_web():
    print('[INFO] Web client disconnected: {}'.format(request.sid))


@socketio.on('predict', namespace="/web")
def predict_sound(time):
    if time is None:
        time=1000
    for _ in range(time//2):
        socketio.emit('words', predict.predict_mfccs(1e-6, 2), namespace='/web') #
        eventlet.sleep(0.1)

if __name__ == "__main__":
    print('[INFO] Starting server at http://localhost:5001')
    socketio.run(app=app, host=os.getenv('IP', '0.0.0.0'), port=int(os.getenv('PORT', 5001)))