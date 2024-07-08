from flask import Flask, request, jsonify, Blueprint
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import load_npz
import pandas as pd
import numpy as np

recommendation_bp = Blueprint("recommendation", __name__)

# 전처리된 데이터 로딩
"""
    데이터 설명
    - books                       : 책 데이터(ISBN, description, category 등)
    - book_embedding_all          : 책 전체 메타데이터 임베딩 리스트
    - book_embedding_category     : category 임베딩 리스트
"""
books = pd.read_pickle("data/books.pkl")
book_embedding_all = load_npz("data/embedding_matrix_all.npz")
book_embedding_category = load_npz("data/embedding_matrix_category.npz")


def click_weight(click_count):
    """클릭 수 기반 가중치 함수
    - parameter
      - click_count : 클릭 수

    - return
      - 가중치
    """
    if click_count == 0:
        return 0.0
    elif click_count == 1:
        return 0.3
    elif click_count == 2:
        return 0.5
    else:
        return 0.7


# 비로그인 유저 대상 랜덤 책 추천 함수
def recommend_random_books():
    """랜덤한 책 10개 추천 함수
    - parameter

    - return
      - list : 랜덤 추천 책 10개
    """

    random_books = books.sample(n=10)[['isbn13','coverURL']].to_dict(orient='records')
    return random_books


# 로그인 유저 대상 사용자 맞춤 책 추천 함수
def recommend_for_user(user_preferences=None, user_clicks=None):
    """유저 선호 정보 기반 책 추천 함수
    - 추천 기준 : category
    - parameter
      - user_preferences : 사용자 선호 책 정보
      - user_clicks : 사용자 클릭 정보

    - return
      - list : 추천 책 10개
    """
    if user_preferences is None:
        user_preferences = []
    if user_clicks is None:
        user_clicks = {}

    # 사용자 정보가 없으면 랜덤 책 추천
    if not user_preferences and not user_clicks:
        return recommend_random_books()

    user_preferences = [str(isbn) for isbn in user_preferences]
    user_clicks = {
        str(isbn): click_weight(click_count)
        for isbn, click_count in user_clicks.items()
    }

    vectorizer = TfidfVectorizer()
    combined_text = books['category'].apply(str).tolist()
    tfidf_matrix = vectorizer.fit_transform(combined_text)

    # 선호책 벡터
    preference_vector = tfidf_matrix[books["isbn13"].isin(user_preferences)].mean(
        axis=0
    )
    preference_vector = np.asarray(preference_vector).ravel()

    # 클릭 수 가중치 벡터
    if user_clicks:
        click_vector = tfidf_matrix[books["isbn13"].isin(user_clicks.keys())].sum(
            axis=0
        )
        click_vector = click_vector.flatten()
    else:
        click_vector = np.zeros(preference_vector.shape)

    # 최종 사용자 프로필
    user_profile = (preference_vector + click_vector) / 2

    # 유사도 계산
    similarity_scores = cosine_similarity(
        user_profile.reshape(1, -1), tfidf_matrix
    ).flatten()

    # 유사도를 기준으로 추천 도서 정렬
    recommended_indices = similarity_scores.argsort()[::-1][:10]
    recommended_books = books.iloc[recommended_indices][['isbn13', 'coverURL']].to_dict(orient='records')

    return recommended_books


def recommend_by_category(preferred_category):
    """사용자 선호 카테고리 기반 책 추천 함수
    - 추천 기준 : category
    - parameter
      - preferred_categories : 사용자 선호 카테고리

    - return
      - list : 추천 책 10개
    """
    filtered_books = books[books["category"] == preferred_category]
    if filtered_books.empty:
        return recommend_random_books()
    else:
        recommendations = filtered_books.sample(n=10)[["isbn13", "coverURL"]].to_dict(orient="records")
        return recommendations


def recommend_similar_books(isbn):
    """이 책과 유사한 책 추천 함수
    - 추천 기준 : 책 메타데이터 전체
    - parameter
      - isbn : 책 ISBN

    - return
      - list : 추천 책 10개
    """
    # 입력 책의 인덱스
    book_index = books[books["isbn13"] == isbn].index[0]
    similarity_scores = cosine_similarity(
        book_embedding_all[book_index].reshape(1, -1), book_embedding_all
    ).flatten()

    # 유사도를 기준으로 추천 도서 정렬
    similar_indices = similarity_scores.argsort()[::-1][1:11]
    similar_books = books.iloc[similar_indices][["isbn13", "coverURL"]].to_dict(orient="records")

    return similar_books


@recommendation_bp.route("/recommendations", methods=["POST"])
def recommendation():
    """클릭 수 기반 가중치 함수
    - parameter
      - click_count : 클릭 수

    - return
      - 가중치
    """

    # 사용자 정보 추출
    user_preferences = request.json.get("user_preferences", [])
    user_clicks = request.json.get("bookClickDtos", {})

    if user_preferences or user_clicks:
        recommendations = recommend_for_user(user_preferences, user_clicks)
    else:
        recommendations = recommend_random_books()

    return jsonify(recommendations)


@recommendation_bp.route("/wishlist/count", methods=["GET"])
def category_recommendation():
    preferred_categories = request.args.getlist("preferred_categories")

    recommendations = {
        "first": recommend_by_category(str(preferred_categories[0])),
        "second": recommend_by_category(str(preferred_categories[1]))
    }

    return jsonify(recommendations)


@recommendation_bp.route("/similar/<isbn>", methods=["GET"])
def similar_recommendation(isbn):
    recommendations = recommend_similar_books(isbn)

    return jsonify(recommendations)
