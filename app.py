from flask import Flask,jsonify,request
import numpy as np
import pickle
app=Flask(__name__)
model=pickle.load(open('model1.pkl','rb'))
@app.route('/predict',methods=['POST'])
def predict():
    total_sqft = request.form.get('total_sqft')
    bath = request.form.get('bath')
    bhk1 = request.form.get('bhk1')
    bhk2 = request.form.get('bhk2')
    bhk3 = request.form.get('bhk3')
    bhkother = request.form.get('bhkother')
    ec = request.form.get('ec')
    kr = request.form.get('kr')
    sr = request.form.get('sr')
    wf = request.form.get('wf')
    other_loc = request.form.get('other_loc')

    input_query = np.array([[total_sqft,bath,bhk1,bhk2,bhk3,bhkother,ec,kr,sr,wf]])
    input= np.array(input_query, dtype=float)

    result = model.predict(input)[0]

    return jsonify({'Price':str(result)})
if __name__ == '__main__':
    app.run(debug=False,host='0.0.0.0')