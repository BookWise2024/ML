from flask import Flask, Blueprint, jsonify
import pandas as pd
import numpy as np
import random
import torch
import torch.nn.functional as F
from transformers import BertTokenizerFast, BertModel
from bs4 import BeautifulSoup
import requests
import logging
import re

# from flask_cors import CORS
# app = Flask(__name__)
# CORS(app, resources={r"/*": {"origins": "*"}})

sentiment_bp = Blueprint('sentiment', __name__)

# logging 설정
logging.basicConfig(level=logging.DEBUG)

def set_seed(SEED):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

set_seed(42) #random SEED 고정


class Predictor():
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.classifier = torch.nn.Linear(self.model.config.hidden_size, 2)  # 분류 레이어 추가
        self.classifier.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0.0)


    def predict(self, text):
        tokens = tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            padding='max_length',
            max_length=128,
            add_special_tokens=True
        )
        tokens = {k:v.to(device) for k,v in tokens.items()}

        with torch.no_grad():
            outputs = self.model(**tokens)
            
        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output)
        prediction = F.softmax(logits, dim=1)

        output = prediction.argmax(dim=1).item()
        prob = prediction.max(dim=1)[0].item()

        return output, prob


model_name = 'kykim/bert-kor-base'
state_dict_path = 'data/sentimentBert.pth'
tokenizer = BertTokenizerFast.from_pretrained(model_name)

bert = BertModel.from_pretrained(model_name)


state_dict = torch.load(state_dict_path, map_location=torch.device("cpu"))
new_state_dict = {}
for key, value in state_dict.items():
    if key.startswith("bert."):
        new_key = key[5:]
        new_state_dict[new_key] = value

bert.load_state_dict(new_state_dict)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert.to(device)
bert.eval()

def get_reviews(item_id):
    reviews = []
    try:
        url = f'https://www.aladin.co.kr/ucl/shop/product/ajax/GetCommunityListAjax.aspx?itemId={item_id}&IsAjax=true&pageType=1&sort=1&communitytype=CommentReview&IsOrderer=2&pageCount=500'
        res = requests.get(url)
        res.raise_for_status()
        soup = BeautifulSoup(res.text, 'lxml')
        
        review_blocks = soup.find_all("div", {"class": "np_40box_list_cont"})

        for block in review_blocks:
            review_content_divs = block.find_all("div", class_="np_40box_list_content")
            if len(review_content_divs) >= 3:
                review_content = review_content_divs[1].get_text(separator="\n", strip=True)
                review_content = re.sub(r"\n", " ", review_content)
                reviews.append(review_content)
        return reviews
    
    except Exception as e:
        logging.error(f"Error fetching reviews for {item_id} : {e}")
        return []


predictor = Predictor(bert, tokenizer)

def classify_sentiment(item_id):
    reviews = get_reviews(item_id)
    if not reviews:
        return [], []

    positive_reviews = []
    negative_reviews = []
    for review in reviews:
        sentiment, prob = predictor.predict(review)
        
        # if sentiment==1:
        if prob > 0.6:
            positive_reviews.append((review, prob))
        else:
            negative_reviews.append((review, prob))

    # 예측 확률 기준으로 상위 3개를 반환함
    positive_reviews = sorted(positive_reviews, key=lambda x: x[1], reverse=True)[:3]
    negative_reviews = sorted(negative_reviews, key=lambda x: x[1], reverse=True)[:3]

    return positive_reviews, negative_reviews


@sentiment_bp.route("/reviews/<item_id>", methods=["GET"])
# @app.route("/reviews/<item_id>", methods=["GET"])
def reviews(item_id):
    logging.debug(f'Recieving item_id: {item_id}')
    
    positive_reviews, negative_reviews = classify_sentiment(item_id)
    response = {
        "positive": [review for review, _ in positive_reviews],
        "negative": [review for review, _ in negative_reviews]
    }
    return jsonify(response)

# if __name__ == "__main__":
#     app.run(debug=True, host="0.0.0.0", port=5000)
