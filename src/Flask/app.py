from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

with open('decision_tree_model_new.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

def prepare_input_data(request_form):
    #dictionary to map feature names to feature numbers
    feature_mapping = {
        'feature4': ['BLOWING SAND, SOIL, DIRT', 'BLOWING SNOW', 'CLEAR', 'CLOUDY', 'FOGGY', 'OTHER', 'RAINING', 'SEVERE WINDS', 'SLEET', 'SNOW', 'UNKNOWN', 'WINTRY MIX'],
        'feature5': ['DRY', 'ICE', 'MUD, DIRT, GRAVEL', 'OIL', 'OTHER', 'OIL', 'SAND', 'SLUSH', 'SNOW', 'UNKNOWN', 'WATER(STANDING/MOVING)', 'WET'],
        'feature6': ['ALCOHOL CONTRIBUTED', 'ALCOHOL PRESENT', 'COMBINATION CONTRIBUTED','COMBINED SUBSTANCE PRESENT', 'ILLEGAL DRUG CONTRIBUTED', 'ILLEGAL DRUG PRESENT', 'MEDICATION CONTRIBUTED', 'MEDICATION PRESENT', 'NONE DETECTED', 'OTHER', 'UNKNOWN'],
        'feature7': ['ADJUSTING AUDIO AND OR CLIMATE CONTROLS', 'BY MOVING OBJECT IN VEHICLE', 'BY OTHER OCCUPANTS', 'DIALING CELLULAR PHONE', 'OUTSIDE PERSON OBJECT OR EVENT', 'EATING OR DRINKING', 'INATTENTIVE OR LOST IN THOUGHT', 'LOOKED BUT DID NOT SEE', 'NO DRIVER PRESENT', 'NOT DISTRACTED', 'OTHER CELLULAR PHONE RELATED', 'OTHER DISTRACTION', 'OTHER ELECTRONIC DEVICE (NAVIGATIONAL PALM PILOT)', 'SMOKING RELATED', 'TALKING OR LISTENING TO CELLULAR PHONE', 'TEXTING FROM A CELLULAR PHONE']
    }

    #initialize the input data to zeros
    input_data = [0] * 53

    #iterate through each feature and its possible values
    for feature_name, feature_values in feature_mapping.items():
        feature_value = request.form.get(feature_name)
        if feature_value:
            feature_index = feature_values.index(feature_value)
            input_data[feature_index] = 1
    
    return input_data


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        input_data = prepare_input_data(request.form)
        prediction = loaded_model.predict([input_data])

        #print the values
        print(input_data)

        return render_template('index.html', prediction_result=prediction[0])


if __name__ == '__main__':
    app.run(debug=True)
