import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import json

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():
    try:
        unit_size = eval(request.form.get('unit_size', 0))
        unit_name = request.form.get('unit_name')
        unit_type = request.form.get('unit_type')
        experience_level = request.form.get('experience_level')
        equipment = request.form.get('equipment')
        command_structure = request.form.get('command_structure')
        logistics = request.form.get('logistics')
        enemy_size = request.form.get('enemy_size')
        enemy_unit_type = request.form.get('enemy_unit_type')
        enemy_capabilities = request.form.get('enemy_capabilities')
        morale = request.form.get('morale')
        leadership = request.form.get('leadership')
        mission_objective = request.form.get('mission_objective')
        terrain_type = request.form.get('terrain_type')
        weather_condition = request.form.get('weather_condition')
        time_day = request.form.get('time_day')
        civilian_presence = request.form.get('civilian_presence')
        supporting_arms = request.form.get('supporting_arms')
        enemy_movement = request.form.get('enemy_movement')
        rules_engagement = request.form.get('rules_engagement')
        algorithm = request.form.get('algorithm')
        
        if not all([unit_size, unit_name, unit_type, experience_level, equipment, command_structure, logistics, enemy_size, enemy_unit_type, enemy_capabilities, morale, leadership, mission_objective, terrain_type, weather_condition, time_day, civilian_presence, supporting_arms, enemy_movement, rules_engagement, algorithm]):
            return render_template("index.html", action_taken="Please fill in all fields")

        prediction = model.predict_decision(unit_size, unit_name, unit_type, experience_level, equipment, command_structure, logistics, enemy_size, enemy_unit_type, enemy_capabilities, morale, leadership, mission_objective, terrain_type, weather_condition, time_day, civilian_presence, supporting_arms, enemy_movement, rules_engagement, algorithm)
        return render_template("index.html", action_taken = prediction)
    except Exception as e:
        return render_template("index.html", action_taken=f"An error occurred: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)