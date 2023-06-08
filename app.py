# import openai
# # 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
# openai.api_key = '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'

# def generate_response(input_message):
#     prompt = "i am giving you message. you need to tell me which feature  [out of these - ( number of sales, sales in usd , average usd , active market wallets, secondary sales , secondary sales usd , unique buyer , unique seller ) and time of prediction in talking about . if no time of prediction then return 0 . do not give and explaination and greeting ,just only feature and month number in json .   message =  "
#     user_message = input_message

#     messages = [
        
#         {"role": "user", "content": prompt + user_message}
#     ]

#     chat = openai.ChatCompletion.create(
#         model="gpt-3.5-turbo",
#         messages=messages,
#         max_tokens=50,  # Adjust the response length as desired
#         n=1,  # Set the number of responses to generate
#         stop=None,  # Set a custom stop sequence if needed
#         temperature=0.7,  # Adjust the temperature to control randomness
#     )

#     response = chat.choices[0].message.content
#     return response

# # Example usage
# input_message = "what can be number of active user after to month"
# response = generate_response(input_message)
# print(response)




import requests
from flask import Flask, jsonify, request
from flask_cors import CORS


app = Flask(__name__)
# CORS(app)
# api = Api(app)


@app.route('/prompts', methods=['POST'])
def return_prompt_response():
    input_message = request.json.get('message')

    prompt = "i am giving you message. you need to tell me which feature out of these - ( number of sales, sales in usd, average usd, active market wallets, secondary sales, secondary sales usd, unique buyer, unique seller ) and time of prediction in talking about. if no time of prediction then return 0. do not give and explaination and greeting, just only feature and month number in json0.   message =  "
    # input_message= "what can be number of active user after thre month"
    user_message = prompt+input_message
        
    url = "https://chatgpt-api8.p.rapidapi.com/"

    payload = [
        {
            "content": user_message ,
            "role": "user"
        }
    ]
    headers = {
        "content-type": "application/json",
        "X-RapidAPI-Key": "03299c5613mshdb18cb35d75382ap1c703cjsn5087e3843a36",
        "X-RapidAPI-Host": "chatgpt-api8.p.rapidapi.com"
    }

    response = requests.post(url, json=payload, headers=headers)

    print(response.json())

    return response.json()
    

# api.add_resource(TestClass, "/next")    

if __name__ == "__main__":
    app.run(debug=True,host='0.0.0.0')