import numpy as np
import os
from keras.models import load_model
from keras.preprocessing import image
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from tensorflow.python.keras.backend import set_session
tf_config = tf.ConfigProto()
sess = tf.Session(config=tf_config)

#tf_config = tf.compat.v1.ConfigProto()
#sess = tf.compat.v1.Session(config=tf_config)

global graph




#graph = tf.compat.v1.get_default_graph()
graph = tf.get_default_graph() # To communicate with the model
from flask import Flask , request, render_template
from werkzeug.utils import secure_filename # To save the file whenever we upload an image
from gevent.pywsgi import WSGIServer # Gateway server between our python program and HTML.




# request -> To get the data from the HTML page
# render -> open the HTML file that is created



app = Flask(__name__)


set_session(sess)
model = load_model("mymodel.h5")

@app.route('/', methods=['GET'])
def index():
    return render_template('base.html')

@app.route('/predict',methods = ['GET','POST'])
def upload():
    if request.method == 'POST':
        # Get the file from POST request
        f = request.files['image']
        
        # print("current path")
        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)        
        print("current folder path", basepath)
        filepath = os.path.join(basepath,'uploads',secure_filename(f.filename))
        # print("upload folder is ", filepath)
        f.save(filepath)
        
        img = image.load_img(filepath,target_size = (64,64))
        x = image.img_to_array(img)
        x = np.expand_dims(x,axis =0)
        
        with graph.as_default():
            set_session(sess)
            preds = model.predict_classes(x)
            
            
            print("prediction",preds)
            
        index = ['chrysocolla','quartz']
        
        text = "the predicted rock is : " + str(index[preds[0]])
        
        return text
    
if __name__ == '__main__':
    app.run(debug = True, threaded = False)
        
        
    
    
    