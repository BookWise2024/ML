import pandas as pd
import requests
from bs4 import BeautifulSoup
import json

# title, author, publication, pub_date, description, category, cover_url, review,


# item-> title, author, pubDate, description, itemId(저장X), cover, categoryName, publisher
# packing-> styleDesc (판형정보)
# 카테고리(총 30개) -> '건강/취미', '경제경영', '고등학교참고서', '공무원 수험서', '과학', '달력/기타', '대학교재', '만화', '사회과학', '소설/시/희곡', '수험서/자격증', '어린이', '에세이', '여행', '역사', '예술/대중문화', '외국어', '요리/살림', '유아', '인문학', '자기계발', '잡지', '장르소설', '전집/중고전집', '종교/역학', '좋은부모', '청소년', '컴퓨터/모바일', '초등학교참고서', '중학교참고서'

# itemID 얻어와서 review 가져오기

def get_reviews(item_id):
    reviews = []
    try:
        url2 = f"https://www.aladin.co.kr/ucl/shop/product/ajax/GetCommunityListAjax.aspx?itemId={item_id}&IsAjax=true&pageType=1&sort=1&communitytype=CommentReview&IsOrderer=2&pageCount=500"
        res = requests.get(url2)
        res.raise_for_status()
        soup = BeautifulSoup(res.text, "lxml")

        review_elements = soup.find_all("div", {"class": "np_40box_list_cont"})
        for element in review_elements:
            divs = element.find_all('div')
            if len(divs) > 1:
                review = divs[1].get_text(separator="\n", strip=True)
                reviews.append(review)

        return reviews if reviews else None
        
    except Exception as e:
        print(f"No Reviews for itemID {item_id}: {e}")
        return None

# 알라딘 api로 가져옴
def get_book_info(ttb_key, isbn13):

    book_info = []

    url = f"http://www.aladin.co.kr/ttb/api/ItemLookUp.aspx?ttbkey={ttb_key}&itemIdType=ISBN&ItemId={isbn13}&output=js&Version=20131101&OptResult=reviewList,packing,ratingInfo"

    res = requests.get(url)
    if res.status_code == 200:
        items = json.loads(res.text).get('item', [])
        print(f"{isbn13} 메타 정보 추출중!")

        for item in items:
            # 저자 파싱
            author_full = item['author']
            author_name = author_full.split('(지은이)')[0].strip()

            # 카테고리 파싱
            category_full = item['categoryName']
            categories = [cat.strip() for cat in category_full.split('>') if cat.strip()]
            categories = categories[1:]
            categories_len = len(categories)

            book_dict = {
                'title' : item['title'] if item['title'] else None,
                'author' : author_name,
                'pubDate' : item['pubDate'],
                'description' : item['description'],
                'isbn13' : item['isbn13'],
                'itemId' : item['itemId'],
                'cover' : item['cover'],
                'category' : categories[0] if categories_len>0 else None,
                'category2' : categories[1] if categories_len>1 else None,
                'category3' : categories[2] if categories_len>2 else None,
                'publisher' : item['publisher'],
                'styleDesc' : item.get('subInfo', {}).get('packing', {}).get('styleDesc'),
                'bestSellerRank' : item.get('subInfo', {}).get('bestSellerRank'),
                'reviews' : get_reviews(item['itemId']) or None
            }
            book_info.append(book_dict)
            print(f"{isbn13} 정상 추출 완료!-{book_info}")

    else:
        print(f"{isbn13}에서 정보를 가져오지 못했습니다")

    return book_info

import os
def main():

    isbn_list = pd.read_csv('webisbn.csv')
    isbn_list = isbn_list['isbn13'].astype(str)
    book_metadata = []

    # or_review_metadata = pd.read_csv('review_metadata.csv')


    start_index = 0
    # last_processed = "9788961711784"
    # if last_processed in isbn_list.values: 
    #     start_index = isbn_list[isbn_list.values==last_processed].index[0] + 1

    try:
        for isbn13 in isbn_list[start_index:]:
            book_info = get_book_info(ttb_key, isbn13)
            if book_info:
                book_metadata.extend(book_info)

    except Exception as e:
        print(e)
        result = pd.DataFrame(book_metadata)

        result.to_csv("review_metadata.csv", index=False)

    result = pd.DataFrame(book_metadata)

    # 초기 데이터 생성 코드
    result.to_csv('review_metadata.csv', index=False)

    # 초기 데이터 생성 후 합치기
    # new_review_metadata = pd.concat([or_review_metadata, result], ignore_index=True)
    # new_review_metadata.to_csv('review_metadata.csv', index=False)


if __name__ == "__main__":
    main()
