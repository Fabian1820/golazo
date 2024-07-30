import pandas as pd
import numpy as np
from scipy.stats import poisson
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
import matplotlib.pyplot as plt

def preparar_datos(matches, rosters):
    # Agregar información de tarjetas al DataFrame de partidos
    matches['yellow_cards'] = matches.apply(lambda row: rosters[(rosters['match_id'] == row['id']) & (rosters['yellow_card'] == 1)].shape[0], axis=1)
    matches['red_cards'] = matches.apply(lambda row: rosters[(rosters['match_id'] == row['id']) & (rosters['red_card'] == 1)].shape[0], axis=1)

    # Categorizar las tarjetas amarillas
    matches['yellow_card_category'] = pd.cut(matches['yellow_cards'], bins=[-1, 3, 6, 9, np.inf], labels=[1, 2, 3, 4])

    # Categorizar los disparos a puerta
    matches['h_shotOnTarget_category'] = pd.cut(matches['h_shotOnTarget'], bins=[-1, 2, 5, np.inf], labels=[0, 1, 2])
    matches['a_shotOnTarget_category'] = pd.cut(matches['a_shotOnTarget'], bins=[-1, 2, 5, np.inf], labels=[0, 1, 2])

    # Seleccionar características relevantes
    features = ['h_shot', 'a_shot', 'h_shotOnTarget', 'a_shotOnTarget', 'h_deep', 'a_deep', 'h_ppda', 'a_ppda']
    X = matches[features]
    y_home = matches['h_goals']
    y_away = matches['a_goals']
    y_yellow = matches['yellow_card_category']
    y_red = (matches['red_cards'] > 0).astype(int)
    y_h_shots = matches['h_shotOnTarget_category']
    y_a_shots = matches['a_shotOnTarget_category']

    return train_test_split(X, y_home, y_away, y_yellow, y_red, y_h_shots, y_a_shots, test_size=0.2, random_state=42)

def entrenar_modelos(X_train, y_home_train, y_away_train, y_yellow_train, y_red_train, y_h_shots_train, y_a_shots_train):
    rf_model_home = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model_away = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model_yellow = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model_red = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model_h_shots = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model_a_shots = RandomForestClassifier(n_estimators=100, random_state=42)

    rf_model_home.fit(X_train, y_home_train)
    rf_model_away.fit(X_train, y_away_train)
    rf_model_yellow.fit(X_train, y_yellow_train)
    rf_model_red.fit(X_train, y_red_train)
    rf_model_h_shots.fit(X_train, y_h_shots_train)
    rf_model_a_shots.fit(X_train, y_a_shots_train)

    return rf_model_home, rf_model_away, rf_model_yellow, rf_model_red, rf_model_h_shots, rf_model_a_shots

def calcular_probabilidades(goles_esperados):
    prob_0 = poisson.pmf(0, goles_esperados)
    prob_1 = poisson.pmf(1, goles_esperados)
    prob_2 = poisson.pmf(2, goles_esperados)
    prob_3_plus = 1 - poisson.cdf(2, goles_esperados)
    return prob_0, prob_1, prob_2, prob_3_plus

def predecir_goles_y_probabilidades(home_team, away_team, matches, rf_model_home, rf_model_away, rf_model_yellow, rf_model_red, rf_model_h_shots, rf_model_a_shots):
    home_data = matches[matches['team_h'] == home_team].sort_values('date').iloc[-1]
    away_data = matches[matches['team_a'] == away_team].sort_values('date').iloc[-1]

    input_data = pd.DataFrame({
        'h_shot': [home_data['h_shot']],
        'a_shot': [away_data['a_shot']],
        'h_shotOnTarget': [home_data['h_shotOnTarget']],
        'a_shotOnTarget': [away_data['a_shotOnTarget']],
        'h_deep': [home_data['h_deep']],
        'a_deep': [away_data['a_deep']],
        'h_ppda': [home_data['h_ppda']],
        'a_ppda': [away_data['a_ppda']]
    })

    home_goals = rf_model_home.predict(input_data)[0]
    away_goals = rf_model_away.predict(input_data)[0]
    total_goals = home_goals + away_goals

    yellow_card_probs = rf_model_yellow.predict_proba(input_data)[0]
    red_card_prob = rf_model_red.predict_proba(input_data)[0][1]
    h_shots_probs = rf_model_h_shots.predict_proba(input_data)[0]
    a_shots_probs = rf_model_a_shots.predict_proba(input_data)[0]

    home_probs = calcular_probabilidades(home_goals)
    away_probs = calcular_probabilidades(away_goals)
    total_probs = calcular_probabilidades(total_goals)

    return home_goals, away_goals, total_goals, home_probs, away_probs, total_probs, yellow_card_probs, red_card_prob, h_shots_probs, a_shots_probs

def predecir_gol_jugador(player_name, home_team, away_team, matches, rosters, rf_model_home, rf_model_away):
    try:
        # Filtrar los partidos relevantes
        relevant_matches = matches[(matches['team_h'] == home_team) | (matches['team_a'] == away_team) |
                                   (matches['team_h'] == away_team) | (matches['team_a'] == home_team)]
        if relevant_matches.empty:
            print(f"No se encontraron partidos para {home_team} o {away_team}")
            return 0.05  # Valor base más realista

        # Obtener el último partido de cada equipo
        last_home_match = relevant_matches[relevant_matches['team_h'] == home_team].sort_values('date').iloc[-1]
        last_away_match = relevant_matches[relevant_matches['team_a'] == away_team].sort_values('date').iloc[-1]

        # Filtrar los datos del jugador
        player_matches = rosters[rosters['player'] == player_name]
        if player_matches.empty:
            print(f"No se encontraron datos para el jugador {player_name}")
            return 0.05  # Valor base más realista

        # Calcular estadísticas del jugador
        goals_per_match = player_matches['goals'].mean()
        xg_per_match = player_matches['xG'].mean()
        minutes_per_match = player_matches['time'].mean()

        # Si el jugador no tiene estadísticas, usar promedios de la liga
        if pd.isna(goals_per_match) or goals_per_match == 0:
            goals_per_match = rosters['goals'].mean()
        if pd.isna(xg_per_match) or xg_per_match == 0:
            xg_per_match = rosters['xG'].mean()
        if pd.isna(minutes_per_match) or minutes_per_match == 0:
            minutes_per_match = rosters['time'].mean()

        # Determinar si el jugador es local o visitante
        is_home = player_matches.iloc[-1]['team'] == home_team if not player_matches.empty else True

        # Preparar datos para el modelo
        input_data = pd.DataFrame({
            'h_shot': [max(last_home_match['h_shot'], 1)],
            'a_shot': [max(last_away_match['a_shot'], 1)],
            'h_shotOnTarget': [max(last_home_match['h_shotOnTarget'], 1)],
            'a_shotOnTarget': [max(last_away_match['a_shotOnTarget'], 1)],
            'h_deep': [max(last_home_match['h_deep'], 1)],
            'a_deep': [max(last_away_match['a_deep'], 1)],
            'h_ppda': [max(last_home_match['h_ppda'], 0.1)],
            'a_ppda': [max(last_away_match['a_ppda'], 0.1)]
        })

        # Predecir goles del equipo
        team_goals = max(rf_model_home.predict(input_data)[0] if is_home else rf_model_away.predict(input_data)[0], 0.1)

        # Calcular el ratio de goles del equipo
        team_avg_goals = max(relevant_matches['h_goals' if is_home else 'a_goals'].mean(), 0.1)
        team_goal_ratio = team_goals / team_avg_goals

        # Calcular la probabilidad base
        expected_minutes = min(minutes_per_match, 90)
        minutes_factor = expected_minutes / 90
        base_prob = 1 - np.exp(-(goals_per_match * team_goal_ratio * minutes_factor))

        # Ajustar por xG
        xg_factor = min(max(xg_per_match / max(goals_per_match, 0.01), 0.5), 2)

        # Ajustar por posición
        position = player_matches.iloc[-1]['position'] if not player_matches.empty else 'M'
        position_factor = 1.5 if position in ['FW', 'F'] else 1.0 if position in ['M', 'MC'] else 0.5

        # Calcular la probabilidad ajustada final
        adjusted_goal_prob = base_prob * xg_factor * position_factor

        return min(max(adjusted_goal_prob, 0.01), 0.99)

    except Exception as e:
        print(f"Error en predecir_gol_jugador: {e}")
        return 0.05  # Valor base más realista en caso de error

# Cargar datos (deberás implementar esto según cómo almacenes tus datos)
matches = pd.read_csv('/home/user/fabian-python/src/soccer/Matches.csv')
rosters = pd.read_csv('/home/user/fabian-python/src/soccer/Rosters.csv')

# Preparar datos y entrenar modelos
X_train, X_test, y_home_train, y_home_test, y_away_train, y_away_test, y_yellow_train, y_yellow_test, y_red_train, y_red_test, y_h_shots_train, y_h_shots_test, y_a_shots_train, y_a_shots_test = preparar_datos(matches, rosters)
rf_model_home, rf_model_away, rf_model_yellow, rf_model_red, rf_model_h_shots, rf_model_a_shots = entrenar_modelos(X_train, y_home_train, y_away_train, y_yellow_train, y_red_train, y_h_shots_train, y_a_shots_train)

def get_predictions(home_team, away_team):
    results = predecir_goles_y_probabilidades(home_team, away_team, matches, rf_model_home, rf_model_away, rf_model_yellow, rf_model_red, rf_model_h_shots, rf_model_a_shots)
    home_goals, away_goals, total_goals, home_probs, away_probs, total_probs, yellow_card_probs, red_card_prob, h_shots_probs, a_shots_probs = results
    
    return {
        "home_goals": float(home_goals),
        "away_goals": float(away_goals),
        "total_goals": float(total_goals),
        "home_probs": list(home_probs),  # Cambiado de .tolist() a list()
        "away_probs": list(away_probs),  # Cambiado de .tolist() a list()
        "total_probs": list(total_probs),  # Cambiado de .tolist() a list()
        "yellow_card_probs": list(yellow_card_probs),  # Cambiado de .tolist() a list()
        "red_card_prob": float(red_card_prob),
        "h_shots_probs": list(h_shots_probs),  # Cambiado de .tolist() a list()
        "a_shots_probs": list(a_shots_probs)  # Cambiado de .tolist() a list()
    }

def get_player_prediction(player_name, home_team, away_team):
    goal_prob = predecir_gol_jugador(player_name, home_team, away_team, matches, rosters, rf_model_home, rf_model_away)
    return {"goal_probability": float(goal_prob)}