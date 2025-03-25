from flask import Blueprint, request, jsonify
import os

UPLOAD_FOLDER = './dataset/record files'

audio_bp = Blueprint('audio', __name__)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@audio_bp.route('/save_record', methods=['POST'])
def save_record():
    audio = request.files['audio']
    
    if audio:
        audio.save(os.path.join(UPLOAD_FOLDER, audio.filename))
        return jsonify({'message': '업로드 성공'})
    
    return jsonify({'error': '파일 없음'}), 400