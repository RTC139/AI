import random
import re
import json
import os
import nltk                                            # Natural Language Toolkit for text processing
import numpy as np                                     # For numerical operations with vectors
from datetime import datetime
from collections import deque
from sklearn.feature_extraction.text import TfidfVectorizer  # Converts text to numeric vectors
from sklearn.metrics.pairwise import cosine_similarity       # Measures similarity between texts
import requests  # For making API calls to a dictionary service
import tensorflow as tf  # TensorFlow for deep learning capabilities
from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
from tensorflow.keras.utils import to_categorical
import csv

# Download nltk data if not already downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# New Conversation Dataset class to handle conversational data
class ConversationDataset:
    def __init__(self, dataset_path=None):
        self.conversations = []
        self.dataset_path = dataset_path or "datasets"
        self.ensure_dataset_directory()
        self.load_datasets()
        
        # TensorFlow model for conversation response selection
        self.conversation_model = None
        self.conversation_tokenizer = None
        self.max_sequence_length = 50
        self.model_file = "conversation_model.h5"
        self.tokenizer_file = "conversation_tokenizer.json"
        
        # Initialize TensorFlow models
        self.load_or_create_conversation_model()
        
    def ensure_dataset_directory(self):
        """Create dataset directory if it doesn't exist"""
        if not os.path.exists(self.dataset_path):
            os.makedirs(self.dataset_path)
            
        # Create sample dataset if none exists
        self.create_sample_dataset()
    
    def create_sample_dataset(self):
        """Create sample conversation dataset if none exists"""
        sample_dataset_path = os.path.join(self.dataset_path, "sample_conversations.csv")
        
        if os.path.exists(sample_dataset_path):
            return
            
        # Create a sample dataset with conversation pairs
        sample_conversations = [
            ["Hello", "Hi there! How can I help you today?"],
            ["How are you?", "I'm functioning well, thank you for asking. How can I assist you?"],
            ["What can you do?", "I can answer questions, have conversations, provide information on various topics, and learn new things."],
            ["Tell me about artificial intelligence", "AI refers to systems designed to perform tasks that typically require human intelligence. These include learning, reasoning, problem-solving, and understanding language."],
            ["What's machine learning?", "Machine learning is a subset of AI where systems learn from data without being explicitly programmed. They improve automatically through experience."],
            ["How does deep learning work?", "Deep learning uses neural networks with many layers to process data, learn from it, and make decisions. It's inspired by the human brain's structure."],
            ["What's your favorite color?", "As an AI, I don't have preferences or favorites, but I'm programmed to help answer your questions about many topics!"],
            ["Thanks for your help", "You're welcome! Feel free to ask if you have more questions."],
            ["Goodbye", "Goodbye! Have a great day!"],
            ["I need help with programming", "I'd be happy to help with programming. What language or concept are you working with?"],
        ]
        
        # Write to CSV
        with open(sample_dataset_path, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(["input", "response"])  # Header
            writer.writerows(sample_conversations)
            
        print(f"Created sample conversation dataset at {sample_dataset_path}")
    
    def load_datasets(self):
        """Load conversation datasets from files"""
        self.conversations = []
        
        # Check for datasets in the directory
        if os.path.exists(self.dataset_path):
            for filename in os.listdir(self.dataset_path):
                file_path = os.path.join(self.dataset_path, filename)
                
                # Handle CSV files
                if filename.endswith('.csv'):
                    try:
                        df = pd.read_csv(file_path)
                        if 'input' in df.columns and 'response' in df.columns:
                            for _, row in df.iterrows():
                                self.conversations.append({
                                    'input': row['input'],
                                    'response': row['response']
                                })
                    except Exception as e:
                        print(f"Error loading dataset {filename}: {e}")
                
                # Handle JSON files
                elif filename.endswith('.json'):
                    try:
                        with open(file_path, 'r', encoding='utf-8') as file:
                            data = json.load(file)
                            
                            # Handle various JSON formats
                            if isinstance(data, list):
                                for item in data:
                                    if isinstance(item, dict) and 'input' in item and 'response' in item:
                                        self.conversations.append(item)
                            elif isinstance(data, dict) and 'conversations' in data:
                                for conv in data['conversations']:
                                    if isinstance(conv, dict) and 'input' in conv and 'response' in conv:
                                        self.conversations.append(conv)
                    except Exception as e:
                        print(f"Error loading dataset {filename}: {e}")
        
        print(f"Loaded {len(self.conversations)} conversation pairs from datasets")
    
    def get_response(self, user_input, use_tf=True):
        """Get a response to user input from the dataset"""
        if not self.conversations:
            return "I don't have any conversation data to respond with."
        
        # Use TensorFlow model if available and requested
        if use_tf and self.conversation_model and self.conversation_tokenizer:
            return self.get_tf_response(user_input)
        
        # Otherwise use cosine similarity to find the best match
        vectorizer = TfidfVectorizer(min_df=1, stop_words='english')
        inputs = [conv['input'] for conv in self.conversations]
        
        # Handle empty corpus
        if not inputs:
            return "I don't have any conversation data to respond with."
        
        # Create TF-IDF matrix
        tfidf_matrix = vectorizer.fit_transform(inputs)
        
        # Transform user input
        user_vector = vectorizer.transform([user_input])
        
        # Calculate similarity
        similarities = cosine_similarity(user_vector, tfidf_matrix).flatten()
        
        # Find the most similar
        best_match_idx = similarities.argmax()
        
        # Return if similarity is above threshold
        if similarities[best_match_idx] > 0.2:
            return self.conversations[best_match_idx]['response']
        
        # If no good match found
        return None
    
    def add_conversation(self, user_input, response):
        """Add a new conversation pair to the dataset"""
        if not user_input or not response:
            return False
            
        self.conversations.append({
            'input': user_input,
            'response': response,
            'timestamp': datetime.now().isoformat()
        })
        
        # Save to a CSV file periodically
        if random.random() < 0.2:  # 20% chance to save
            self.save_conversations()
            
        return True
    
    def save_conversations(self):
        """Save conversations to a CSV file"""
        if not self.conversations:
            return
            
        dataset_file = os.path.join(self.dataset_path, f"learned_conversations_{datetime.now().strftime('%Y%m%d')}.csv")
        
        # Write to CSV
        try:
            with open(dataset_file, 'w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(["input", "response", "timestamp"])  # Header
                
                for conv in self.conversations:
                    timestamp = conv.get('timestamp', datetime.now().isoformat())
                    writer.writerow([conv['input'], conv['response'], timestamp])
                    
            print(f"Saved {len(self.conversations)} conversation pairs to {dataset_file}")
            
            # Retrain model occasionally
            if random.random() < 0.3 and len(self.conversations) > 20:  # 30% chance if we have enough data
                self.create_and_train_conversation_model()
                
        except Exception as e:
            print(f"Error saving conversations: {e}")
    
    def load_or_create_conversation_model(self):
        """Load existing conversation model or create a new one"""
        try:
            # Try to load existing model
            if os.path.exists(self.model_file) and os.path.exists(self.tokenizer_file):
                print("Loading existing conversation TensorFlow model...")
                self.conversation_model = load_model(self.model_file)
                
                with open(self.tokenizer_file, 'r') as f:
                    tokenizer_json = json.load(f)
                    self.conversation_tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(json.dumps(tokenizer_json))
                
                print("Conversation model and tokenizer loaded successfully.")
            elif len(self.conversations) > 10:  # Only create if we have enough data
                self.create_and_train_conversation_model()
        except Exception as e:
            print(f"Error loading conversation model: {e}")
    
    def create_and_train_conversation_model(self):
        """Create and train a TensorFlow model for conversation response selection"""
        if len(self.conversations) < 10:
            print("Not enough conversation data to train a model")
            return
            
        try:
            # Prepare training data
            inputs = [conv['input'] for conv in self.conversations]
            responses = [conv['response'] for conv in self.conversations]
            
            # Create response categories (each response is a category)
            response_to_idx = {resp: idx for idx, resp in enumerate(set(responses))}
            idx_to_response = {idx: resp for resp, idx in response_to_idx.items()}
            
            # Convert responses to categorical labels
            labels = [response_to_idx[resp] for resp in responses]
            labels = to_categorical(labels, num_classes=len(response_to_idx))
            
            # Create tokenizer
            self.conversation_tokenizer = Tokenizer(num_words=5000)
            self.conversation_tokenizer.fit_on_texts(inputs)
            
            # Convert texts to sequences
            sequences = self.conversation_tokenizer.texts_to_sequences(inputs)
            padded_sequences = pad_sequences(sequences, maxlen=self.max_sequence_length)
            
            # Build model
            vocab_size = len(self.conversation_tokenizer.word_index) + 1
            embedding_dim = 128
            
            model = Sequential([
                Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=self.max_sequence_length),
                Bidirectional(LSTM(64)),
                Dense(128, activation='relu'),
                Dropout(0.5),
                Dense(len(response_to_idx), activation='softmax')
            ])
            
            # Compile and train
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            
            model.fit(
                padded_sequences, labels,
                epochs=20,
                batch_size=8,
                validation_split=0.2,
                verbose=1
            )
            
            # Save model and tokenizer
            model.save(self.model_file)
            
            tokenizer_json = self.conversation_tokenizer.to_json()
            with open(self.tokenizer_file, 'w') as f:
                json.dump(json.loads(tokenizer_json), f)
                
            # Save response mappings
            with open('response_mappings.json', 'w') as f:
                json.dump({str(k): v for k, v in idx_to_response.items()}, f)
            
            self.conversation_model = model
            self.idx_to_response = idx_to_response
            
            print("Conversation model trained and saved successfully")
            
        except Exception as e:
            print(f"Error creating conversation model: {e}")
    
    def get_tf_response(self, user_input):
        """Get response using the TensorFlow model"""
        try:
            # Load response mappings if not available
            if not hasattr(self, 'idx_to_response'):
                try:
                    with open('response_mappings.json', 'r') as f:
                        self.idx_to_response = {int(k): v for k, v in json.load(f).items()}
                except:
                    # Fallback to recreating the mapping
                    responses = [conv['response'] for conv in self.conversations]
                    self.idx_to_response = {idx: resp for idx, resp in enumerate(set(responses))}
            
            # Prepare input
            sequence = self.conversation_tokenizer.texts_to_sequences([user_input])
            padded_seq = pad_sequences(sequence, maxlen=self.max_sequence_length)
            
            # Get prediction
            prediction = self.conversation_model.predict(padded_seq)[0]
            
            # Get top response
            top_idx = np.argmax(prediction)
            confidence = prediction[top_idx]
            
            # Only use if confidence is high enough
            if confidence > 0.4:
                return self.idx_to_response[top_idx]
            else:
                return None
                
        except Exception as e:
            print(f"Error getting TF response: {e}")
            return None

# AI class that uses the dataset-based approach for conversations
class ConversationAI:
    def __init__(self):
        # Initialize datasets
        self.conversation_dataset = ConversationDataset()
        
        # TensorFlow text classifier (from SimpleAI)
        self.categories = ["ai", "programming", "science", "history", "technology", "general"]
        self.tf_model = None
        self.tokenizer = None
        self.max_sequence_length = 100
        self.tf_model_file = "tf_text_classifier.h5"
        self.tokenizer_file = "tokenizer.json"
        
        # Load existing TensorFlow model if available
        self.load_tf_model()
        
        # Knowledge base and corpus handling 
        self.corpus_dir = "corpus"
        self.ensure_corpus_structure()
        self.documents = []
        self.document_sources = []
        self.load_corpus()
        
        # TF-IDF for document similarity
        self.vectorizer = TfidfVectorizer(stop_words='english')
        if self.documents:
            self.tfidf_matrix = self.vectorizer.fit_transform(self.documents)
        else:
            self.tfidf_matrix = None
            
        # Other utilities
        self.conversation_history = deque(maxlen=10)
        self.learned_knowledge = self.load_learned_knowledge()
        
        # Dictionary API
        self.dictionary_api_url = "https://api.dictionaryapi.dev/api/v2/entries/en/"
        self.dictionary_enabled = True
        
        # Add word knowledge structures that were missing
        self.dictionary_cache = {}
        self.word_associations = {}
        self.vocabulary_focus = set()
        self.interesting_words = set()
        self.word_usage_examples = {}
        self.max_interesting_words = 100
        
        # Load word knowledge data
        self.load_word_knowledge()
    
    def load_tf_model(self):
        """Load TensorFlow text classification model"""
        try:
            if os.path.exists(self.tf_model_file) and os.path.exists(self.tokenizer_file):
                print("Loading existing TensorFlow classification model...")
                self.tf_model = load_model(self.tf_model_file)
                
                with open(self.tokenizer_file, 'r') as f:
                    tokenizer_json = json.load(f)
                    self.tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(json.dumps(tokenizer_json))
                
                print("TensorFlow classification model loaded successfully")
        except Exception as e:
            print(f"Error loading TensorFlow model: {e}")
    
    def ensure_corpus_structure(self):
        """Create corpus directory structure if it doesn't exist"""
        if not os.path.exists(self.corpus_dir):
            os.makedirs(self.corpus_dir)
            
        for category in self.categories:
            category_dir = os.path.join(self.corpus_dir, category)
            if not os.path.exists(category_dir):
                os.makedirs(category_dir)
    
    def load_corpus(self):
        """Load all text documents from the corpus directories"""
        self.documents = []
        self.document_sources = []
        
        for category in self.categories:
            category_dir = os.path.join(self.corpus_dir, category)
            
            if os.path.exists(category_dir):
                for filename in os.listdir(category_dir):
                    if filename.endswith('.txt'):
                        try:
                            file_path = os.path.join(category_dir, filename)
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                                self.documents.append(content)
                                self.document_sources.append({
                                    'category': category,
                                    'filename': filename,
                                    'path': file_path
                                })
                        except Exception as e:
                            print(f"Error loading {filename}: {e}")
    
    def load_learned_knowledge(self):
        """Load previously learned knowledge"""
        learned_data_file = "learned_knowledge.json"
        try:
            if os.path.exists(learned_data_file):
                with open(learned_data_file, 'r') as file:
                    return json.load(file)
            return {}
        except Exception as e:
            print(f"Error loading learned data: {e}")
            return {}
    
    def load_word_knowledge(self):
        """Load word associations and dictionary cache from files"""
        try:
            # Load word associations
            if os.path.exists("word_associations.json"):
                with open("word_associations.json", "r") as f:
                    loaded_associations = json.load(f)
                    
                    # Convert lists back to sets
                    for word, data in loaded_associations.items():
                        self.word_associations[word] = {
                            "related_words": set(data["related_words"]),
                            "contexts": data["contexts"],
                            "importance": data["importance"]
                        }
            
            # Load dictionary cache
            if os.path.exists("dictionary_cache.json"):
                with open("dictionary_cache.json", "r") as f:
                    self.dictionary_cache = json.load(f)
                    
            # Load interesting words
            if os.path.exists("interesting_words.json"):
                with open("interesting_words.json", "r") as f:
                    self.interesting_words = set(json.load(f))
                    
        except Exception as e:
            print(f"Error loading word knowledge: {e}")
    
    def save_word_knowledge(self):
        """Save word associations and dictionary cache to files"""
        try:
            # Convert sets to lists for JSON serialization
            serializable_associations = {}
            for word, data in self.word_associations.items():
                serializable_associations[word] = {
                    "related_words": list(data["related_words"]),
                    "contexts": data["contexts"],
                    "importance": data["importance"]
                }
                
            # Save word associations
            with open("word_associations.json", "w") as f:
                json.dump(serializable_associations, f)
                
            # Save dictionary cache periodically (not every time to avoid excessive writes)
            if random.random() < 0.1:  # 10% chance to save
                with open("dictionary_cache.json", "w") as f:
                    json.dump(self.dictionary_cache, f)
                    
            # Save interesting words
            with open("interesting_words.json", "w") as f:
                json.dump(list(self.interesting_words), f)
                
        except Exception as e:
            print(f"Error saving word knowledge: {e}")
    
    def add_interesting_word(self, word):
        """Add a word to the set of interesting words"""
        word = word.lower()
        if len(word) > 3:  # Only track non-trivial words
            self.interesting_words.add(word)
            # If we exceed our limit, remove least important word
            if len(self.interesting_words) > self.max_interesting_words:
                # Find least important word
                least_important = None
                min_importance = float('inf')
                for w in self.interesting_words:
                    importance = self.word_associations.get(w, {}).get("importance", 0)
                    if importance < min_importance:
                        min_importance = importance
                        least_important = w
                
                if least_important:
                    self.interesting_words.remove(least_important)
    
    def classify_text_with_tf(self, text):
        """Use TensorFlow to classify text into one of the categories"""
        if not self.tf_model or not self.tokenizer:
            return "general"
        
        try:
            # Convert text to sequence
            sequences = self.tokenizer.texts_to_sequences([text])
            padded_seq = pad_sequences(sequences, maxlen=self.max_sequence_length)
            
            # Get prediction probabilities
            predictions = self.tf_model.predict(padded_seq)[0]
            
            # Get the category with highest probability
            category_index = np.argmax(predictions)
            confidence = predictions[category_index]
            
            # Return the category only if confidence is high enough
            if confidence > 0.4:
                return self.categories[category_index]
            else:
                return "general"
                
        except Exception as e:
            print(f"Error classifying text with TensorFlow: {e}")
            return "general"
    
    def search_corpus(self, query, top_n=3):
        """Search the corpus for documents relevant to the query"""
        if not self.documents or len(self.documents) == 0:
            return []
            
        try:
            query_vector = self.vectorizer.transform([query])
            similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
            
            top_indices = similarities.argsort()[-top_n:][::-1]
            
            results = []
            for i in top_indices:
                if similarities[i] > 0.05:
                    results.append({
                        'text': self.documents[i],
                        'source': self.document_sources[i],
                        'similarity': similarities[i]
                    })
            
            return results
        except Exception as e:
            print(f"Error searching corpus: {e}")
            return []
    
    def generate_response(self, user_input):
        """Generate a response to user input using datasets and TensorFlow"""
        if not user_input or user_input.strip() == "":
            return "I didn't catch that. Could you please say something?"
            
        # Add to conversation history
        self.conversation_history.append(("user", user_input))
        
        try:
            # First try to get a response from the conversation dataset
            dataset_response = self.conversation_dataset.get_response(user_input)
            
            if dataset_response:
                self.conversation_history.append(("ai", dataset_response))
                return dataset_response
            
            # Check learned knowledge
            user_input_lower = user_input.lower().strip()
            for question, data in self.learned_knowledge.items():
                if question in user_input_lower:
                    response = data["answer"]
                    self.conversation_history.append(("ai", response))
                    return f"{response} (I learned this previously)"
            
            # Search the corpus
            search_results = self.search_corpus(user_input)
            if search_results and len(search_results) > 0:
                best_result = search_results[0]
                
                # Format the response
                if "Question:" in best_result['text'] and "Answer:" in best_result['text']:
                    answer_part = best_result['text'].split("Answer:")[1].strip()
                    response = answer_part
                else:
                    response = best_result['text']
                
                self.conversation_history.append(("ai", response))
                # Add this successful response to the conversation dataset
                self.conversation_dataset.add_conversation(user_input, response)
                return response
            
            # If we couldn't generate a response, say so
            no_answer_response = "I don't have enough information to answer that question yet. Would you like to teach me about it?"
            self.conversation_history.append(("ai", no_answer_response))
            return no_answer_response
            
        except Exception as e:
            print(f"Error generating response: {e}")
            fallback = "I'm having trouble processing that. Let's try something else."
            self.conversation_history.append(("ai", fallback))
            return fallback

# Legacy SimpleAI class retained for backward compatibility
class SimpleAI:
    def __init__(self):
        # Create a ConversationAI instance for actual functionality
        self._conversation_ai = ConversationAI()
        # Forward attributes
        self.greetings = ["hello", "hi", "hey", "greetings", "what's up", "howdy", "hola", "good morning", "good afternoon", "good evening"]
        self.farewells = ["bye", "goodbye", "see you", "cya", "farewell", "later", "take care", "adios"]
        self.corpus_dir = self._conversation_ai.corpus_dir
        self.categories = self._conversation_ai.categories
        
    # Forward method calls to ConversationAI
    def generate_response(self, user_input):
        return self._conversation_ai.generate_response(user_input)
    
    def classify_text_with_tf(self, text):
        return self._conversation_ai.classify_text_with_tf(text)
