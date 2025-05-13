from flask import Flask, render_template, request, jsonify, session
import random
import json
import os
import traceback

app = Flask(__name__)
app.secret_key = 'ai_learning_application_key'  # Required for session

# Create global AI assistant with better error handling
ai_assistant = None

def initialize_ai():
    """Initialize AI with proper error handling"""
    global ai_assistant
    
    try:
        # Try the new ConversationAI class first
        from ai_model import ConversationAI
        print("Initializing ConversationAI...")
        ai_assistant = ConversationAI()
        return True
    except Exception as e:
        print(f"Error initializing ConversationAI: {e}")
        print(traceback.format_exc())
        
        try:
            # Fall back to SimpleAI if it exists
            from ai_model import SimpleAI
            print("Falling back to SimpleAI...")
            ai_assistant = SimpleAI()
            return True
        except Exception as e2:
            print(f"Error initializing SimpleAI: {e2}")
            print(traceback.format_exc())
            
            # Last resort - create a minimal responder
            print("Creating minimal responder...")
            ai_assistant = MinimalResponder()
            return False

class MinimalResponder:
    """Fallback class if AI initialization fails"""
    def __init__(self):
        self.responses = {
            "hello": "Hello! I'm running in minimal mode due to initialization errors.",
            "hi": "Hi there! The system is currently in fallback mode.",
            "help": "I'm a minimal responder. The AI system couldn't be initialized properly."
        }
    
    def generate_response(self, user_input):
        """Provide basic responses when AI is unavailable"""
        user_input = user_input.lower().strip()
        
        # Check for exact matches
        if user_input in self.responses:
            return self.responses[user_input]
            
        # Check for partial matches
        for key in self.responses:
            if key in user_input:
                return self.responses[key]
        
        return "I'm sorry, the AI system is currently in minimal mode. Please try again later."

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"response": "No data received"}), 400
            
        user_message = data.get('message', '')
        if not user_message:
            return jsonify({"response": "Please send a message."}), 400
        
        # Extra protection around the AI response generation
        try:
            ai_response = ai_assistant.generate_response(user_message)
            return jsonify({"response": ai_response})
        except Exception as e:
            print(f"AI response generation error: {str(e)}")
            # Return a friendlier message for AI processing errors
            return jsonify({"response": "I'm not sure how to respond to that. Could you try asking something else?"}), 200
            
    except Exception as e:
        print(f"Request processing error: {str(e)}")
        print(traceback.format_exc())
        return jsonify({"response": "Sorry, I'm having trouble processing requests right now."}), 500

@app.route('/feedback', methods=['POST'])
def feedback():
    """Endpoint to receive feedback on AI responses"""
    try:
        data = request.get_json()
        question = data.get('question')
        helpful = data.get('helpful', False)
        
        if question:
            ai_assistant.rate_response(question, helpful)
            return jsonify({"status": "Feedback received, thank you!"})
        return jsonify({"status": "Missing question"})
    except Exception as e:
        return jsonify({"status": f"Error: {str(e)}"})

@app.route('/add-to-corpus', methods=['POST'])
def add_to_corpus():
    """Endpoint to manually add content to the corpus"""
    try:
        data = request.get_json()
        text = data.get('text')
        category = data.get('category', 'general')
        filename = data.get('filename')
        
        if not text:
            return jsonify({"status": "No text provided"}), 400
            
        success = ai_assistant.add_to_corpus(text, category, filename)
        
        if success:
            return jsonify({"status": "Content added to corpus successfully"})
        else:
            return jsonify({"status": "Failed to add content to corpus"}), 500
    except Exception as e:
        return jsonify({"status": f"Error: {str(e)}"}), 500

@app.route('/reload-corpus', methods=['POST'])
def reload_corpus():
    """Endpoint to reload the corpus"""
    try:
        ai_assistant.load_corpus()
        return jsonify({"status": "Corpus reloaded successfully"})
    except Exception as e:
        return jsonify({"status": f"Error: {str(e)}"}), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    global ai_assistant
    
    # Make sure AI is initialized
    if ai_assistant is None:
        initialize_ai()
    
    try:
        user_message = request.json.get('message', '')
        if not user_message:
            return jsonify({"response": "I didn't catch that. Could you please say something?"}), 200
        
        # Get response from AI model with error handling
        response = ai_assistant.generate_response(user_message)
        
        return jsonify({
            "response": response
        })
    except Exception as e:
        print(f"Error in /api/chat endpoint: {str(e)}")
        print(traceback.format_exc())
        
        # Try to reinitialize AI if it failed
        success = initialize_ai()
        
        # Return a user-friendly error message
        if success:
            return jsonify({
                "response": "I had a temporary glitch but I'm ready now. Could you please repeat your question?"
            })
        else:
            return jsonify({
                "response": "I'm having technical difficulties at the moment. Please try again later."
            })

if __name__ == '__main__':
    # Initialize AI before starting the server
    initialize_ai()
    app.run(debug=True, host='127.0.0.1', port=5000)
