from combine_military import MilitaryDecisionModel
import pickle

with open('model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

# Example usage
predicted_price = loaded_model.predict_decision(220, 'Platoon', 'Reconnaissance', 'Elite', 'Medical Supplies', 'Decentralized', 'Basic', 123, 'Irregular Militia', 'Sniper Teams', 'Fanatical', 'Incompetent', 'Counter-insurgency', 'Urban', 'Extreme Cold', 'Night', 'Medium', 'Yes (Air & Artillery)', 'Static Defense', 'Weapons Free', 'k-NN')

print(predicted_price)
