import os
import pandas as pd
from flask import Flask, send_file, jsonify, request
from src.predictor import get_predictions, get_player_prediction

app = Flask(__name__)

# Cargar los datos
matches_data = pd.read_csv('/home/user/fabian-python/src/soccer/Matches.csv')
players_data = pd.read_csv('/home/user/fabian-python/src/soccer/PlayersData.csv')

@app.route("/")
def index():
    return send_file('src/index.html')

@app.route("/api/leagues")
def get_leagues():
    leagues = matches_data['league'].unique().tolist()
    return jsonify(leagues)

@app.route("/api/teams/<league>")
def get_teams(league):
    teams = matches_data[matches_data['league'] == league]['team_h'].unique().tolist()
    return jsonify(teams)

@app.route("/api/players/<team>")
def get_players(team):
    players = players_data[players_data['team_title'] == team]['player_name'].unique().tolist()
    return jsonify(players)

@app.route("/api/predict", methods=['POST'])
def predict():
    data = request.json
    home_team = data['team_h']  
    away_team = data['team_a'] 
    predictions = get_predictions(home_team, away_team)
    return jsonify(predictions)

@app.route("/api/predict_player", methods=['POST'])
def predict_player():
    data = request.json
    player_name = data['player_name']
    home_team = data['team_h']  
    away_team = data['team_a']  
    prediction = get_player_prediction(player_name, home_team, away_team)
    return jsonify(prediction)

def main():
    app.run(host="0.0.0.0", port=8080)

if __name__ == "__main__":
    main()