"""
Script to enhance the AI corpus with additional knowledge
"""
import os
import sys

def ensure_dir(directory):
    """Make sure directory exists"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def write_text_file(filepath, content):
    """Write content to file, creating directories if needed"""
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Created: {filepath}")
    except Exception as e:
        print(f"Error creating {filepath}: {e}")

def create_enhanced_corpus():
    """Create an enhanced corpus with more useful knowledge"""
    print("Creating enhanced AI corpus...")
    
    # Base directory
    corpus_dir = "corpus"
    ensure_dir(corpus_dir)
    
    # Create directories for each category
    categories = ["ai", "programming", "science", "history", "technology", "general"]
    for category in categories:
        ensure_dir(os.path.join(corpus_dir, category))
    
    # AI self-knowledge
    ai_knowledge = {
        "about_this_ai.txt": """
This is a simple AI assistant built with Python and Flask. It uses a text corpus and TF-IDF vectorization
to find relevant information to answer questions. The AI can learn from conversations, perform calculations,
and store new information in its knowledge base.

The AI was created as a demonstration project to show how to build a basic AI assistant using Python.
It doesn't use complex deep learning models but instead relies on information retrieval and simple
pattern matching.
""",
        "what_are_you.txt": """
I am a simple AI assistant. I'm designed to answer questions, learn from conversations,
and provide helpful information. Unlike more complex AI systems, I work by searching through 
a collection of text documents to find relevant information.
""",
        "how_do_you_work.txt": """
I work by converting questions into mathematical vectors using TF-IDF (Term Frequency-Inverse Document Frequency).
Then I compare these vectors to my corpus of documents using cosine similarity to find the most relevant information.
I can also learn new information when users teach me, which gets saved to my corpus for future use.
"""
    }
    
    # Add more categories and files
    programming_knowledge = {
        "python_detailed.txt": """
Python is a high-level, interpreted programming language created by Guido van Rossum and first released in 1991.
Python's design philosophy emphasizes code readability with its notable use of significant whitespace.
Its language constructs and object-oriented approach aim to help programmers write clear, logical code for small and large-scale projects.

Python features:
- Easy to learn and use syntax
- Dynamic typing and memory management
- Extensive standard library
- Support for multiple programming paradigms
- Large community and ecosystem of packages
"""
    }
    
    # Write all files
    for filename, content in ai_knowledge.items():
        write_text_file(os.path.join(corpus_dir, "ai", filename), content.strip())
    
    for filename, content in programming_knowledge.items():
        write_text_file(os.path.join(corpus_dir, "programming", filename), content.strip())
    
    print("Enhanced corpus creation complete!")

if __name__ == "__main__":
    create_enhanced_corpus()
    print("\nRun the following to add this content to your AI:")
    print("python create_enhanced_corpus.py")
    print("Then restart your Flask server")
