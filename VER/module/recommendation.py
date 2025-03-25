import requests

TMDB_API_KEY = "df9a0caaf2a07ee6babd7024a6accaf8"

EMOTION_TO_GENRE = {
    '기쁨': 35,  # Comedy
    '슬픔': 18,  # Drama
    '분노': 53,  # Thriller
    '불안': 27,  # Horror
    '상처': 80,  # Crime
    '당황': 28,  # Action
    '중립': 10751,  # Family
}

def get_recommendations(emotion, result_num=10, api_key=TMDB_API_KEY):
    genre_id = EMOTION_TO_GENRE.get(emotion)
    if not genre_id:
        return f"'{emotion}'에 해당하는 추천 장르가 없습니다. 감정을 다시 입력해주세요."

    url = f"https://api.themoviedb.org/3/discover/movie"
    params = {
        "api_key": api_key,
        "include_video": True,
        "with_genres": genre_id,
        "sort_by": "popularity.desc",  # 인기 순으로 정렬
        "language": "ko-KR",          # 한국어 결과
        "vote_average.gte": 7.0,      # 평점 7 이상
    }

    response = requests.get(url, params=params)
    if response.status_code != 200:
        return f"TMDB API 호출 실패: {response.status_code}"

    data = response.json()
    results = data.get("results", [])

    if not results:
        return f"'{emotion}'에 맞는 추천 콘텐츠를 찾을 수 없습니다."

    recommendations = []
    for movie in results[:result_num]:  # 상위 N개만 추출
        recommendations.append({
            "title": movie.get("title"),
            "overview": movie.get("overview"),
            "vote_average": movie.get("vote_average"),
            "release_date": movie.get("release_date"),
        })

    return recommendations
