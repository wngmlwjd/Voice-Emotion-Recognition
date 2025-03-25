from flask import Blueprint, request, jsonify

from VER.module.recommendation import get_recommendations

recommend_bp = Blueprint('recommend', __name__)

@recommend_bp.route('/recommend', methods=['POST'])
def recommend():
    emotion = request.form.get('emotion')

    if emotion:
        try:
            results = get_recommendations(emotion)

            # 결과 반환
            return jsonify({
                "results": results,
            })

        except Exception as e:
            print(f"Error: {str(e)}") 
            return jsonify({'error': str(e)}), 500
    return jsonify({'error': '파일 없음'}), 400