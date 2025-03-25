from flask import Flask, render_template
from routes.save_record import audio_bp
from routes.predict import predict_bp
from VER.module.model import EmotionTransformer

app = Flask(__name__)
app.register_blueprint(audio_bp)
app.register_blueprint(predict_bp)

@app.route('/')
def home():
    return render_template('main.html')

if __name__ == '__main__':
    app.run(debug=True)