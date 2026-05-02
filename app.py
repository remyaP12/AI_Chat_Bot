from flask import Flask, render_template, request, jsonify
from chatbot import get_response

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')  # ✅ Must match filename exactly

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')
    if not user_message:
        return jsonify({'error': 'No message provided'}), 400
    bot_response = get_response(user_message)
    return jsonify({
        'user': user_message,
        'bot': bot_response
    })

if __name__ == '__main__':
    app.run(debug=True)



#run
#python model\train_model.py
# python app.py    