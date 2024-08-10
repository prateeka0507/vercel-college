import os
from flask import Flask, render_template_string, request, jsonify
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
import tiktoken
from tiktoken import get_encoding
import uuid
import time
import random

app = Flask(__name__)

# Access your API keys (set these in Vercel environment variables)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
INDEX_NAME_CONTENT = "college"
INDEX_NAME_METADATA = "college-buddy-metadata"

# Initialize OpenAI
client = OpenAI(api_key=OPENAI_API_KEY)

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Create or connect to the Pinecone indexes
for index_name in [INDEX_NAME_CONTENT, INDEX_NAME_METADATA]:
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric='cosine',
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )

index_content = pc.Index(INDEX_NAME_CONTENT)
index_metadata = pc.Index(INDEX_NAME_METADATA)

# Helper functions
def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response.data[0].embedding

def query_pinecone(query, index, top_k=5):
    query_embedding = get_embedding(query)
    results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
    contexts = []
    for match in results['matches']:
        if 'chunk_text' in match['metadata']:
            contexts.append(match['metadata']['chunk_text'])
        else:
            contexts.append(f"Content from {match['metadata'].get('file_name', 'unknown file')}")
    return " ".join(contexts)

def identify_intents(query):
    intent_prompt = f"Identify the main intent or question within this query. Provide only one primary intent: {query}"
    intent_response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an intent identification assistant. Identify and provide only the primary intent or question within the given query."},
            {"role": "user", "content": intent_prompt}
        ]
    )
    intent = intent_response.choices[0].message.content.strip()
    return [intent] if intent else []

def generate_keywords_per_intent(intents):
    intent_keywords = {}
    for intent in intents:
        keyword_prompt = f"Generate 5-10 relevant keywords or phrases for this intent, separated by commas: {intent}"
        keyword_response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a keyword extraction assistant. Generate relevant keywords or phrases for the given intent."},
                {"role": "user", "content": keyword_prompt}
            ]
        )
        keywords = keyword_response.choices[0].message.content.strip().split(',')
        intent_keywords[intent] = [keyword.strip() for keyword in keywords]
    return intent_keywords

def query_for_multiple_intents(intent_keywords):
    intent_data = {}
    all_metadata_results = []
    for intent, keywords in intent_keywords.items():
        metadata_results = index_metadata.query(vector=get_embedding(" ".join(keywords)), top_k=5, include_metadata=True)
        new_metadata_results = [match for match in metadata_results['matches'] if match['id'] not in [r['id'] for r in all_metadata_results]]
        all_metadata_results.extend(new_metadata_results)
        pinecone_context = query_pinecone(" ".join(keywords), index_content)
        intent_data[intent] = {
            'metadata_results': new_metadata_results,
            'pinecone_context': pinecone_context
        }
    return intent_data

def generate_multi_intent_answer(query, intent_data):
    context = "\n".join([f"Intent: {intent}\nMetadata Results: {data['metadata_results']}\nPinecone Context: {data['pinecone_context']}" for intent, data in intent_data.items()])
    max_context_tokens = 4000
    tokenizer = get_encoding("cl100k_base")
    truncated_context = tokenizer.decode(tokenizer.encode(context)[:max_context_tokens])
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are College Buddy, an AI assistant designed to help students with their academic queries..."},
            {"role": "user", "content": f"Query: {query}\n\nContext: {truncated_context}"}
        ]
    )
   
    return response.choices[0].message.content.strip()

def extract_keywords_from_response(response):
    keyword_prompt = f"Extract 5-10 key terms or phrases from this text, separated by commas: {response}"
    keyword_response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a keyword extraction assistant. Extract key terms or phrases from the given text."},
            {"role": "user", "content": keyword_prompt}
        ]
    )
    keywords = keyword_response.choices[0].message.content.strip().split(',')
    return [keyword.strip() for keyword in keywords]

def get_answer(query):
    intents = identify_intents(query)
    intent_keywords = generate_keywords_per_intent(intents)
    intent_data = query_for_multiple_intents(intent_keywords)
    initial_answer = generate_multi_intent_answer(query, intent_data)
    response_keywords = extract_keywords_from_response(initial_answer)
    all_keywords = list(set(intent_keywords[intents[0]] + response_keywords))
    expanded_intent_data = query_for_multiple_intents({query: all_keywords})
    final_answer = generate_multi_intent_answer(query, expanded_intent_data)
    return final_answer, expanded_intent_data, all_keywords

# Flask routes
@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/chat', methods=['POST'])
def chat():
    user_query = request.json['message']
    final_answer, intent_data, all_keywords = get_answer(user_query)
    return jsonify({
        'answer': final_answer,
        'keywords': all_keywords,
        'related_documents': [
            {
                'title': match['metadata']['title'],
                'tags': match['metadata']['tags'],
                'links': match['metadata']['links']
            }
            for intent in intent_data.values()
            for match in intent['metadata_results']
        ]
    })

# HTML template
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>College Buddy Assistant</title>
    <style>
        body { font-family: Arial, sans-serif; line-height: 1.6; padding: 20px; }
        h1 { color: #333; }
        #chat-container { margin-top: 20px; }
        #user-input { width: 80%; padding: 10px; }
        button { padding: 10px; background-color: #007bff; color: white; border: none; cursor: pointer; }
        .message { margin: 10px 0; padding: 10px; border-radius: 5px; }
        .user-message { background-color: #f0f0f0; }
        .bot-message { background-color: #e6f2ff; }
    </style>
</head>
<body>
    <h1>College Buddy Assistant</h1>
    <p>Welcome to College Buddy! I am here to help you stay organized, find information fast and provide assistance. Feel free to ask me a question below.</p>
    
    <div id="chat-container"></div>
    
    <input type="text" id="user-input" placeholder="Ask your question...">
    <button onclick="sendMessage()">Send</button>

    <script>
        function sendMessage() {
            const userInput = document.getElementById('user-input');
            const message = userInput.value;
            if (message.trim() === '') return;

            addMessageToChat(message, 'user-message');
            
            fetch('/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({message: message}),
            })
            .then(response => response.json())
            .then(data => {
                addMessageToChat(data.answer, 'bot-message');
                addKeywordsToChat(data.keywords);
                addRelatedDocumentsToChat(data.related_documents);
                userInput.value = '';
            });
        }

        function addMessageToChat(message, className) {
            const chatContainer = document.getElementById('chat-container');
            const messageElement = document.createElement('div');
            messageElement.classList.add('message', className);
            messageElement.textContent = message;
            chatContainer.appendChild(messageElement);
        }

        function addKeywordsToChat(keywords) {
            const chatContainer = document.getElementById('chat-container');
            const keywordsElement = document.createElement('div');
            keywordsElement.classList.add('message', 'bot-message');
            keywordsElement.innerHTML = '<strong>Related Keywords:</strong> ' + keywords.join(', ');
            chatContainer.appendChild(keywordsElement);
        }

        function addRelatedDocumentsToChat(documents) {
            const chatContainer = document.getElementById('chat-container');
            const documentsElement = document.createElement('div');
            documentsElement.classList.add('message', 'bot-message');
            documentsElement.innerHTML = '<strong>Related Documents:</strong><ul>' + 
                documents.map(doc => `<li>${doc.title} (Tags: ${doc.tags})</li>`).join('') + 
                '</ul>';
            chatContainer.appendChild(documentsElement);
        }
    </script>
</body>
</html>
'''

if __name__ == '__main__':
    app.run(debug=True)
