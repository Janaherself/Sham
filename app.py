from flask import Flask, request, render_template
#from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import catboost

app = Flask(__name__, '/static')

model = pickle.load(open('models/mylatestmodel.pkl', 'rb'))
vectorizer = pickle.load(open('models/vectorizer.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    email = request.form.values()

    alphabet = '''abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'''
    text = ""

    for element in email:
        if element in alphabet:
            text += element

    text = vectorizer.transform([text])

    prediction = model.predict(text)

    if prediction == 0:
        return render_template('index.html', prediction_text='This email is a HAM!')
    else:
        return render_template('index.html', prediction_text='This email is a SPAM!')


if __name__ == '__main__':
    app.debug = True
    app.run()

