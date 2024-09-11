import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, PolynomialFeatures
from sklearn.linear_model import RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import uniform
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

class DefenseModel:
    def __init__(self):
        df = pd.read_csv('military.csv')
        df_shuffled = df.sample(frac=1, random_state=42)

        split_ratio = 0.7
        split_index = int(len(df_shuffled) * split_ratio)

        train_data = df_shuffled[:split_index]
        test_data = df_shuffled[split_index:]

        self.train_Y = train_data['Action Taken']
        self.train_X = train_data.drop('Action Taken', axis=1)

        self.actual_Y = test_data['Action Taken']
        self.test_X = test_data.drop('Action Taken', axis=1)

        numerical_cols = self.train_X.columns[self.train_X.dtypes != 'object']

        numerical_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
        ])

        categorical_cols = self.train_X.columns[self.train_X.dtypes == 'object']

        ordinal_cols = ['Experience Level', 'Equipment', 'Logistics', 'Known Enemy Capabilities', 'Morale', 'Leadership', 'Mission Objective',
                        'Terrain Type', 'Weather Conditions', 'Civilian Presence', 'Enemy Movement', 'Rules of Engagement']

        ordinal_categories = [
            ['Rookie', 'Mixed', 'Veteran', 'Elite'],
            ['Medical Supplies', 'Basic (Limited Ammo)', 'Standard Issue', 'Riot Gear', 'Specialized Gear', 'Advanced Weaponry'],
            ['Minimal', 'Basic', 'Adequate', 'Well-Supplied', 'Overstocked'],
            ['Standard Equipment', 'Improvised Explosive Devices (IEDs)', 'Sniper Teams', 'Advanced Communication'],
            ['Low', 'Medium', 'High', 'Fanatical'],
            ['Incompetent', 'Weak', 'Competent', 'Strong', 'Tactical Genius', 'Charismatic Leader'],
            ['Peacekeeping', 'Cyber Defense', 'Infrastructure Protection', 'Reconnaissance', 'Secure Objective', 'Eliminate Enemy Force', 'Rescue/Evacuation', 'Counter-insurgency'],
            ['Island', 'Coastal', 'Desert', 'Rural', 'Suburban', 'Arctic', 'Forest', 'Mountain', 'Jungle', 'Urban'],
            ['Clear', 'Light Rain', 'Cloudy', 'Snow', 'Sandstorm', 'Heavy Rain', 'Fog', 'Thunderstorm', 'Blizzard', 'Extreme Heat', 'Extreme Cold'],
            ['None', 'Low', 'Medium', 'High'],
            ['Static Defense', 'Patrolling', 'Infiltration', 'Encirclement', 'Aggressive Push', 'Tactical Retreat', 'Guerrilla Tactics'],
            ['Observe and Report', 'Non-Lethal Force Only', 'Minimize Collateral Damage', 'Protect Civilians at All Costs', 'Return Fire Only', 'Weapons Free']
        ]
        ordinal_pipeline = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),
                                           ('ordinal_encoder', OrdinalEncoder(categories=ordinal_categories))
                                           ])

        nominal_cols = list(set(categorical_cols) - set(ordinal_cols))

        nominal_pipeline = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),
                                           ('onehot_encoder', OneHotEncoder(handle_unknown='ignore'))
                                           ])

        self.column_transformer = ColumnTransformer(
            transformers=[
                ('numerical', numerical_pipeline, numerical_cols),
                ('ordinal', ordinal_pipeline, ordinal_cols),
                ('nominal', nominal_pipeline, nominal_cols)
            ],
            remainder='passthrough'
        )

    def generate_decision(self, algorithm, test_X):
        predicted_test_Y = None
        action_taken = ''
        if algorithm == 'k-NN':
            pipe = Pipeline(steps=[('preprocessor', self.column_transformer), ('scaler', RobustScaler(with_centering=False)), ('knn', KNeighborsClassifier())])
            param_grid = {
                'knn__n_neighbors': [5, 10], 'knn__p': [1, 2], 'knn__weights': ['uniform', 'distance']
            }
            random_search_cv = GridSearchCV(pipe, param_grid=param_grid, n_jobs=30, scoring='neg_mean_squared_log_error', refit=True, cv=5)
            random_search_cv.fit(self.train_X, self.train_Y)
            best_model = random_search_cv.best_estimator_
            predicted_test_Y = best_model.predict(test_X)

        elif algorithm == 'Ridge Classifier':
            pipe = Pipeline(steps=[('preprocessor', self.column_transformer), ('scaler', RobustScaler(with_centering=False)), ('ridge', RidgeClassifier())])
            param_distribution = {'ridge__alpha': uniform(loc=0, scale=600).rvs(100)}
            random_search_cv = RandomizedSearchCV(pipe, param_distributions=param_distribution, n_iter=30, scoring='neg_mean_squared_log_error', refit=True, cv=5)
            random_search_cv.fit(self.train_X, self.train_Y)
            best_model = random_search_cv.best_estimator_
            predicted_test_Y = best_model.predict(test_X)

        elif algorithm == 'Polynomial Regression':
            pipe = Pipeline(steps=[('preprocessor', self.column_transformer), ('scaler', RobustScaler(with_centering=False)), ('poly', PolynomialFeatures(degree=2, include_bias=False)), ('ridge', RidgeClassifier())])
            param_distribution = {
                'ridge__alpha': uniform(loc=0, scale=600)
            }
            random_search_cv = RandomizedSearchCV(pipe, param_distributions=param_distribution, n_iter=10, scoring='neg_mean_squared_log_error', refit=True, cv=5)
            random_search_cv.fit(self.train_X, self.train_Y)
            best_model = random_search_cv.best_estimator_
            predicted_test_Y = best_model.predict(test_X)

        elif algorithm == 'Logistic Regression':
            pipe = Pipeline(steps=[('preprocessor', self.column_transformer), ('scaler', RobustScaler(with_centering=False)), ('log', LogisticRegression(max_iter=4000, solver='saga'))])
            param_distribution = {
                'log__C': [0.1, 1, 10, 100],
                'log__penalty': ['l1', 'l2']
            }

            random_search_cv = RandomizedSearchCV(pipe, param_distributions=param_distribution, scoring='neg_mean_squared_log_error', refit=True, cv=5)
            random_search_cv.fit(self.train_X, self.train_Y)
            best_model = random_search_cv.best_estimator_
            predicted_test_Y = best_model.predict(test_X)

        if predicted_test_Y[0] == 1:
            action_taken = 'Observe'
        elif predicted_test_Y[0] == 2:
            action_taken = 'Hold Position'
        elif predicted_test_Y[0] == 3:
            action_taken = 'Withdrawn'
        elif predicted_test_Y[0] == 4:
            action_taken = 'Negotiate'
        elif predicted_test_Y[0] == 5:
            action_taken = 'Maneuver'
        elif predicted_test_Y[0] == 6:
            action_taken = 'Engage'
        elif predicted_test_Y[0] == 7:
            action_taken = 'Evacuate Civilians'
        elif predicted_test_Y[0] == 8:
            action_taken = 'Secure Critical Objectives'
        elif predicted_test_Y[0] == 9:
            action_taken = 'Deliver Medical Supplies'

        return action_taken

    def predict_decision(self, unit_size, unit_name, unit_type, experience_level, equipment, command_structure, logistics, enemy_size, enemy_unit_type, enemy_capabilities, morale, leadership, mission_objective, terrain_type, weather_condition, time_day, civilian_presence, supporting_arms, enemy_movement, rules_engagement, algorithm):
        algorithm = algorithm
        test_X = pd.DataFrame([[unit_size, unit_name, unit_type, experience_level, equipment, command_structure, logistics, enemy_size, enemy_unit_type, enemy_capabilities, morale, leadership, mission_objective, terrain_type, weather_condition, time_day, civilian_presence, supporting_arms, enemy_movement, rules_engagement]],
                              columns=['Unit Size', 'Unit Name', 'Unit Type', 'Experience Level', 'Equipment',
                                       'Command Structure', 'Logistics', 'Estimated Enemy Size',
                                       'Enemy Unit Type Composition', 'Known Enemy Capabilities', 'Morale',
                                       'Leadership', 'Mission Objective', 'Terrain Type', 'Weather Conditions',
                                       'Time of Day', 'Civilian Presence', 'Supporting Arms', 'Enemy Movement',
                                       'Rules of Engagement'])

        return self.generate_decision(algorithm, test_X)

#an = model.predict_decision(220, 'Platoon', 'Reconnaissance', 'Elite', 'Medical Supplies', 'Decentralized', 'Basic', 123, 'Irregular Militia', 'Sniper Teams', 'Fanatical', 'Incompetent', 'Counter-insurgency', 'Urban', 'Extreme Cold', 'Night', 'Medium', 'Yes (Air & Artillery)', 'Static Defense', 'Weapons Free', 'k-NN')
