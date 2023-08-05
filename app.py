from flask import Flask, request, jsonify
import requests
import cv2
import numpy as np

app = Flask(__name__)

@app.route('/calculate', methods=['POST'])
def calculate_similarity():
    try:
        # リクエストからJSONデータを取得
        data = request.get_json()

        # 画像のURLを取得
        image_url1 = data['image_url1']
        image_url2 = data['image_url2']

        # 画像をダウンロード
        response1 = requests.get(image_url1)
        response2 = requests.get(image_url2)

        # 画像をOpenCVの形式に変換
        image1 = cv2.imdecode(np.frombuffer(response1.content, np.uint8), cv2.IMREAD_COLOR)
        image2 = cv2.imdecode(np.frombuffer(response2.content, np.uint8), cv2.IMREAD_COLOR)

        # 画像の類似度を計算
        similarity = calculate_image_similarity(image1, image2)

        return jsonify({'similarity': similarity})

    except Exception as e:
        return jsonify({'error': str(e)})

def calculate_image_similarity(image1, image2):
    # 画像の類似度を計算するロジックを実装
    # ここでは単純にヒストグラムの類似度を計算する例を示します
    hist1 = cv2.calcHist([image1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([image2], [0], None, [256], [0, 256])
    similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

    return similarity

if __name__ == '__main__':
    app.run()
