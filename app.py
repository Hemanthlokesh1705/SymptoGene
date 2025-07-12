from flask import Flask, request, jsonify, render_template, redirect, url_for, session

from model.predictor import DiseasePredictor
from werkzeug.middleware.proxy_fix import ProxyFix

app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)
app.secret_key = '4hF@pZ9#sW!qK2mT$LxBv8NeYd^Ra6Ug' 


try:
    predictor = DiseasePredictor()
except Exception as e:
    print(f"Error initializing predictor: {str(e)}")
    predictor = None
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

       
        if username == 'admin' and password == 'symptogene':
            session['user'] = username
            return redirect(url_for('home'))
        else:
            return render_template('login.html', error='Invalid credentials')
    
    return render_template('login.html')

@app.route('/')
def home():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('index.html')
@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))

@app.route('/doctors')
def about():
    """Render the about page"""
    return render_template('doctors.html')
@app.route('/appointment')
def appoint():
    """Render the about page"""
    return render_template('appointment.html')

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for disease prediction"""
    if not predictor:
        return jsonify({'error': 'Predictor not initialized'}), 500

    try:
        data = request.get_json()
        if not data or 'symptoms' not in data:
            return jsonify({'error': 'No symptoms provided'}), 400

        symptoms = data['symptoms']
        if not isinstance(symptoms, list) or len(symptoms) == 0:
            return jsonify({'error': 'Symptoms should be a non-empty list'}), 400

        
        result = predictor.predict(symptoms)
   
        response = {
            'predictions': result['predictions'],
            'top_percentages': result['top_percentages']
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000, debug=True)