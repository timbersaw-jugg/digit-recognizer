from flask import Flask,request,render_template
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os

app=Flask(__name__)
model=load_model("mnist_cnn_model.h5")
#model = load_model("mnist_cnn_model.keras")


def preprocess(img_path):
    img=cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img=cv2.resize(img,(28,28))
    img=255-img
    img=img/255.0
    return img.reshape(1,28,28,1)

@app.route('/',methods=['GET','POST'])
def predict():
    if request.method=="POST":
        file=request.files["digit"]
        path=os.path.join("uploads",file.filename)
        file.save(path)
        img=preprocess(path)
        pred=model.predict(img)
        digit=np.argmax(pred)
        return render_template("index.html",prediction=digit)

    return render_template("index.html")

if __name__=='__main__':
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
