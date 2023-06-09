import requests
import csv
import json
from flask import Flask, jsonify, request
from flask_cors import CORS
import re
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.models import load_model


app = Flask(__name__)
CORS(app)
# api = Api(app)


@app.route('/predictions/contract_address=<contr_add>', methods=['POST'])
def return_predictions(contr_add):
    # Validate the contract address
    pattern = r'^0x[0-9a-fA-F]{40}$'
    if not re.match(pattern, contr_add):
        return jsonify({'error': 'Invalid contract address'}),400
  
    last4char = contr_add[-4:]    
    
    input_message = request.json.get('message')

    prompt = "i am giving you message. you need to tell me which feature out of these - ( number of sales, sales in usd, average usd, active market wallets, secondary sales, secondary sales usd, unique buyer, unique seller ) and time of prediction in talking about. if no time of prediction then return 0. do not give and explaination and greeting, just only feature and month number in json0.   message =  "
    # input_message= "what can be the number of active users after three months"
    user_message = prompt+input_message
        
    url = "https://chatgpt-api8.p.rapidapi.com/"

    payload = [
        {
            "content": user_message,
            "role": "user"
        }
    ]
    headers = {
        "content-type": "application/json",
        "X-RapidAPI-Key": "27af42f834msh74733b5484d6911p1e5002jsn1d23362f1c67",
        "X-RapidAPI-Host": "chatgpt-api8.p.rapidapi.com"
    }
#     b27a6c1e94mshd45ec690537c794p1ff7b3jsn444433f6000c
#     49fba4d0demshb6a2281b6adae66p13613ajsnedf2f167f7ac
#     52fb48fd7dmsh0b71e6ec5aa9659p195552jsn129611c1574a
# 03299c5613mshdb18cb35d75382ap1c703cjsn5087e3843a36
    response = requests.post(url, json=payload, headers=headers)

    response_data = response.json()
    print(response_data)
    print(type(response_data))

    text_data = json.loads(response_data['text'])

    for key, value in text_data.items():
        if 'month' in key.lower():
            months = value
        elif key == 'feature':
            feature = value

    poss_responses = ['active market wallets', 'number of sales', 'unique buyer']
    models = [f'{last4char}_active_market_wallet.h5', f'{last4char}_no_of_sales.h5', f'{last4char}_unique_buyers.h5']
    csvfiles = ['active_market_wallet', 'no_of_sales', 'unique_buyers']

    for respons,modl,csvfile in zip(poss_responses,models,csvfiles):
        if (feature == respons):
            reqd_model = modl
            reqd_csvfile = csvfile
            break
        else:
            return jsonify({'error': 'Model not yet trained for this feature'}), 404

    if os.path.isfile(reqd_model):
        # Model exists, load it
        model = load_model(reqd_model)
        print("Model loaded successfully.")
    else:
        return jsonify({'error': 'Model not yet trained for given parameters'}), 404
     
    # Load the time series data
    data = pd.read_csv(f'{reqd_csvfile}.csv')

    # Select only the 'DateTime' and 'Sales USD' columns
    data = data[['DateTime', 'Sales USD']]

    # Convert 'DateTime' column to datetime format
    data['DateTime'] = pd.to_datetime(data['DateTime'])

    # Set 'DateTime' column as the index
    data.set_index('DateTime', inplace=True)

    # Normalize the values using Min-Max scaling
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # Number of previous days to consider for prediction
    sequence_length = 30

    # Prepare the input data for prediction
    last_sequence = scaled_data[-sequence_length:]
    X_pred = np.array([last_sequence])
    X_pred = np.reshape(X_pred, (X_pred.shape[0], X_pred.shape[1], 1))

    no_of_days = float(months) * 30
    days = int(no_of_days)
    # Make predictions for the next n months
    predictions = []
    for _ in range(days):
        prediction = model.predict(X_pred)
        predictions.append(prediction[0, 0])
        X_pred = np.append(X_pred, [[[prediction[0, 0]]]], axis=1)

    # Inverse transform the predictions
    predictions = scaler.inverse_transform([predictions])[0]

    np.round(predictions,2)

    # Print the predicted sales USD for the next n days
    print(type(predictions[0]))
    print(predictions)

    # Convert the ndarray to a list
    predictions_list = predictions.tolist()

    response_data = {
        "values": predictions_list
    }

    return jsonify(response_data)


@app.route('/train_model/contract_address=<contr_add>', methods=['GET'])
def train_save(contr_add):
    # Validate the contract address
    pattern = r'^0x[0-9a-fA-F]{40}$'
    if not re.match(pattern, contr_add):
        return jsonify({'error': 'Invalid contract address'})   
     
    csvfiles = ['active_market_wallet', 'no_of_sales', 'unique_buyers']
    last4char = contr_add[-4:]  

    for csvfile in csvfiles:
        print(f'Building model for {csvfile}')
        
        # Load the time series data
        data = pd.read_csv(f'{csvfile}.csv')  

        # Select only the 'DateTime' and 'Sales USD' columns
        data = data[['DateTime', 'Sales USD']]

        # Convert 'DateTime' column to datetime format
        data['DateTime'] = pd.to_datetime(data['DateTime'])

        # Set 'DateTime' column as the index
        data.set_index('DateTime', inplace=True)

        # Normalize the 'Sales USD' values using Min-Max scaling
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)

        # Prepare the training data
        sequence_length = 30  # Number of previous days to consider for prediction
        X_train, y_train = [], []

        for i in range(sequence_length, len(data)):
            X_train.append(scaled_data[i-sequence_length:i, 0])
            y_train.append(scaled_data[i, 0])

        X_train, y_train = np.array(X_train), np.array(y_train)

        # Reshape the input data to match the expected input shape of the LSTM model
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

        # Build the LSTM model
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
        model.add(LSTM(units=50))
        model.add(Dense(units=1))

        # Compile the model
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Train the model
        model.fit(X_train, y_train, epochs=50, batch_size=32)

        #save model as h5 file
        model.save(f'{last4char}_{csvfile}.h5')    

        print(f'Model for {csvfile} successfully trained & saved')

    return jsonify({'message': f'Model for {contr_add} successfully trained & saved'})
    

# api.add_resource(TestClass, "/next")    

if __name__ == "__main__":
    app.run(debug=True,host='0.0.0.0')
