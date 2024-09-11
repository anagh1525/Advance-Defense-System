from combine_military import MilitaryDecisionModel
import pickle

# Create an instance of the model
model = MilitaryDecisionModel()

# Dump the model to a pickle file
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
