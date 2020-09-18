from flask import Flask,render_template,request
import pickle
import numpy as np


app=Flask(__name__,template_folder='templates')
model=pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():
    int_feature=[float(x) for x in request.form.values()]
    f_features=np.array(int_feature)
    final_features=f_features.reshape(1,len(f_features))
    prediction=model.predict(final_features)[0]
        
    return render_template('predict.html', prediction_text=prediction)
   
if __name__=='__main__':
      app.run(debug=True)  