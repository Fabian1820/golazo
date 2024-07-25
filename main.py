import os
from flask import Flask, send_file, jsonify, request
from src.predictor import get_predictions, get_player_prediction

app = Flask(__name__)

@app.route("/")
def index():
    return send_file('src/index.html')

@app.route("/api/predict", methods=['POST'])
def predict():
    data = request.json
    home_team = data['home_team']
    away_team = data['away_team']
    predictions = get_predictions(home_team, away_team)
    return jsonify(predictions)

@app.route("/api/predict_player", methods=['POST'])
def predict_player():
    data = request.json
    player_name = data['player_name']
    home_team = data['home_team']
    away_team = data['away_team']
    prediction = get_player_prediction(player_name, home_team, away_team)
    return jsonify(prediction)

def main():
    app.run(port=int(os.environ.get('PORT', 80)))

if __name__ == "__main__":
    main()