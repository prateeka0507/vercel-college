import os
from flask import Flask, render_template_string, request, jsonify, session
from werkzeug.utils import secure_filename
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
import tiktoken
from tiktoken import get_encoding
import uuid
import time
import random
from docx import Document

app = Flask(__name__)
app.secret_key = os.urandom(24)  # For session management

# Access your API keys (set these in Vercel environment variables)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
INDEX_NAME_CONTENT = "college"
INDEX_NAME_METADATA = "college-buddy-metadata"

# Initialize OpenAI and Pinecone clients
client = OpenAI(api_key=OPENAI_API_KEY)
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

# List of example questions
EXAMPLE_QUESTIONS = [
    "What are the steps to declare a major at Texas Tech University",
    "What are the GPA and course requirements for declaring a major in the Rawls College of Business?",
    "How can new students register for the Red Raider Orientation (RRO)",
    "What are the key components of the Texas Tech University Code of Student Conduct",
    "What resources are available for students reporting incidents of misconduct at Texas Tech University",
    "What are the guidelines for amnesty provisions under the Texas Tech University Code of Student Conduct",
    "How does Texas Tech University handle academic misconduct, including plagiarism and cheating",
    "What are the procedures for resolving student misconduct through voluntary resolution or formal hearings",
    "What are the rights and responsibilities of students during the investigative process for misconduct at Texas Tech University",
    "How can students maintain a healthy lifestyle, including nutrition and fitness, while attending Texas Tech University"
]

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
        model="gpt-3.5-turbo",
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
            model="gpt-3.5-turbo",
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
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": """You are College Buddy, an AI assistant designed to help students with their academic queries. Your primary function is to analyze and provide insights based on the context of uploaded documents. Please adhere to the following guidelines:
1. Focus on addressing the primary intent of the query.
2. Provide accurate, relevant information derived from the provided context.
3. If the context doesn't contain sufficient information to answer the query, state this clearly.
4. Maintain a friendly, supportive tone appropriate for assisting students.
5. Provide concise yet comprehensive answers, breaking down complex concepts when necessary.
6. If asked about topics beyond the scope of the provided context, politely state that you don't have that information.
7. Encourage critical thinking by guiding students towards understanding rather than simply providing direct answers.
8. Respect academic integrity by not writing essays or completing assignments on behalf of students.
9. Suggest additional resources only if directly relevant to the primary query.
"""},
            {"role": "user", "content": f"Query: {query}\n\nContext: {truncated_context}"}
        ]
    )
   
    return response.choices[0].message.content.strip()

def get_answer(query):
    try:
        intents = identify_intents(query)
        intent_keywords = generate_keywords_per_intent(intents)
        intent_data = query_for_multiple_intents(intent_keywords)
        final_answer = generate_multi_intent_answer(query, intent_data)
        
        # Ensure intent_data is JSON serializable
        serializable_intent_data = {}
        for intent, data in intent_data.items():
            serializable_intent_data[intent] = {
                'metadata_results': [
                    {
                        'id': result.get('id', ''),
                        'metadata': {
                            'title': result.get('metadata', {}).get('title', ''),
                            'tags': result.get('metadata', {}).get('tags', ''),
                            'links': result.get('metadata', {}).get('links', '')
                        }
                    } for result in data['metadata_results']
                ],
                'pinecone_context': data['pinecone_context']
            }
        
        return final_answer, serializable_intent_data
    except Exception as e:
        print(f"Error in get_answer: {str(e)}")
        return "I'm sorry, I encountered an error while processing your query.", {}

def get_all_metadata():
    results = index_metadata.query(vector=[0]*1536, top_k=10000, include_metadata=True)
    return [
        {
            'id': match['id'],
            'title': match['metadata'].get('title', ''),
            'tags': match['metadata'].get('tags', ''),
            'links': match['metadata'].get('links', '')
        } for match in results['matches']
    ]

def insert_metadata(title, tags, links):
    id = str(uuid.uuid4())
    metadata = {
        "title": title,
        "tags": tags,
        "links": links
    }
    embedding = get_embedding(f"{title} {tags} {links}")
    index_metadata.upsert(vectors=[(id, embedding, metadata)])
    return True

def delete_metadata(id):
    index_metadata.delete(ids=[id])

# Flask routes
@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE, example_questions=EXAMPLE_QUESTIONS)

@app.route('/chat', methods=['POST'])
def chat():
    user_query = request.json['message']
    final_answer, intent_data = get_answer(user_query)
    return jsonify({
        'response': final_answer,
        'intent_data': intent_data
    })

@app.route('/database')
def database():
    metadata = get_all_metadata()
    return render_template_string(DATABASE_TEMPLATE, metadata=metadata)

@app.route('/add_metadata', methods=['POST'])
def add_metadata():
    data = request.json
    success = insert_metadata(data['title'], data['tags'], data['links'])
    return jsonify({'success': success})

@app.route('/delete_metadata/<id>', methods=['DELETE'])
def delete_metadata_route(id):
    delete_metadata(id)
    return jsonify({'success': True})

# HTML Templates
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>College Buddy Assistant</title>
    <script src="https://cdn.jsdelivr.net/npm/axios/dist/axios.min.js"></script>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f0f0f0;
        }
        .container {
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }
        h1, h2 {
            color: #0066cc;
        }
        #chat-container {
            height: 300px;
            overflow-y: auto;
            border: 1px solid #ddd;
            padding: 10px;
            margin-bottom: 20px;
            background-color: #fff;
        }
        #user-input {
            width: 70%;
            padding: 10px;
            margin-right: 10px;
        }
        button {
            padding: 10px 20px;
            background-color: #0066cc;
            color: white;
            border: none;
            cursor: pointer;
            transition: background-color 0.3s;
        }
        button:hover {
            background-color: #0056b3;
        }
        .message {
            margin-bottom: 10px;
            padding: 10px;
            border-radius: 5px;
        }
        .user-message {
            background-color: #e1f5fe;
            text-align: right;
        }
        .bot-message {
            background-color: #f0f0f0;
        }
        .popular-questions {
            margin-top: 20px;
        }
        .popular-question {
            background-color: #e1f5fe;
            padding: 10px;
            margin-bottom: 10px;
            border-radius: 5px;
            cursor: pointer;
        }
        .popular-question:hover {
            background-color: #b3e5fc;
        }
        .sidebar {
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            margin-top: 20px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>College Buddy Assistant</h1>
        <p>Welcome to College Buddy! I am here to help you stay organized, find information fast and provide assistance. Feel free to ask me a question below.</p>
        
        <div id="chat-container"></div>
        
        <input type="text" id="user-input" placeholder="Ask your question...">
        <button onclick="sendMessage()">Send</button>
        
        <div class="popular-questions">
            <h2>Popular Questions</h2>
            <div id="popular-questions-container"></div>
        </div>
        
        <div class="sidebar">
            <h2>Manage Database</h2>
            <button onclick="window.location.href='/database'">View/Manage Database</button>
        </div>
    </div>

    <script>
        // Function to get random items from an array
        function getRandomItems(arr, count) {
            const shuffled = arr.sort(() => 0.5 - Math.random());
            return shuffled.slice(0, count);
        }

        // Populate popular questions with 3 random questions
        const popularQuestionsContainer = document.getElementById('popular-questions-container');
        const randomQuestions = getRandomItems({{ example_questions|tojson }}, 3);
        
        randomQuestions.forEach(question => {
            const div = document.createElement('div');
            div.className = 'popular-question';
            div.textContent = question;
            div.onclick = () => {
                document.getElementById('user-input').value = question;
                sendMessage();
            };
            popularQuestionsContainer.appendChild(div);
        });

        function sendMessage() {
            const userInput = document.getElementById('user-input');
            const message = userInput.value;
            if (message.trim() === '') return;

            addMessageToChat('You', message, 'user-message');
            
            axios.post('/chat', { message: message })
                .then(response => {
                    addMessageToChat('College Buddy', response.data.response, 'bot-message');
                    userInput.value = '';
                })
                .catch(error => {
                    console.error('Error:', error);
                    addMessageToChat('College Buddy', 'Sorry, I encountered an error. Please try again.', 'bot-message');
                });
        }

        function addMessageToChat(sender, message, className) {
            const chatContainer = document.getElementById('chat-container');
            const messageElement = document.createElement('div');
            messageElement.className = `message ${className}`;
            messageElement.innerHTML = `<strong>${sender}:</strong> ${message}`;
            chatContainer.appendChild(messageElement);
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }

        // Allow sending message with Enter key
        document.getElementById('user-input').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                sendMessage();
            }
        });
    </script>
</body>
</html>
'''

DATABASE_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Database Management - College Buddy</title>
    <script src="https://cdn.jsdelivr.net/npm/axios/dist/axios.min.js"></script>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f0f0f0;
        }
        .container {
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }
        h1, h2 {
            color: #0066cc;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }
        th {
            background-color: #f2f2f2;
        }
        input[type="text"] {
            width: 100%;
            padding: 8px;
            margin: 5px 0;
        }
        button {
            padding: 10px 20px;
            background-color: #0066cc;
            color: white;
            border: none;
            cursor: pointer;
            transition: background-color 0.3s;
        }
        button:hover {
            background-color: #0056b3;
        }
        .form-group {
            margin-bottom: 15px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Database Management</h1>
        <button onclick="window.location.href='/'">Back to Chat</button>

        <h2>Add New Document</h2>
        <div class="form-group">
            <input type="text" id="new-title" placeholder="Title">
        </div>
        <div class="form-group">
            <input type="text" id="new-tags" placeholder="Tags (comma-separated)">
        </div>
        <div class="form-group">
            <input type="text" id="new-links" placeholder="Links">
        </div>
        <button onclick="addMetadata()">Add Document</button>

        <h2>Existing Documents</h2>
        <table id="metadata-table">
            <thead>
                <tr>
                    <th>Title</th>
                    <th>Tags</th>
                    <th>Links</th>
                    <th>Action</th>
                </tr>
            </thead>
            <tbody>
                {% for item in metadata %}
                <tr>
                    <td>{{ item.title }}</td>
                    <td>{{ item.tags }}</td>
                    <td>{{ item.links }}</td>
                    <td><button onclick="deleteMetadata('{{ item.id }}')">Delete</button></td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
    </div>

    <script>
        function addMetadata() {
            const title = document.getElementById('new-title').value;
            const tags = document.getElementById('new-tags').value;
            const links = document.getElementById('new-links').value;
            
            axios.post('/add_metadata', { title, tags, links })
                .then(function (response) {
                    if (response.data.success) {
                        alert('Document added successfully');
                        location.reload();
                    } else {
                        alert('Failed to add document');
                    }
                })
                .catch(function (error) {
                    console.error('Error:', error);
                    alert('Error adding document');
                });
        }

        function deleteMetadata(id) {
            if (confirm('Are you sure you want to delete this document?')) {
                axios.delete('/delete_metadata/' + id)
                    .then(function (response) {
                        if (response.data.success) {
                            alert('Document deleted successfully');
                            location.reload();
                        } else {
                            alert('Failed to delete document');
                        }
                    })
                    .catch(function (error) {
                        console.error('Error:', error);
                        alert('Error deleting document');
                    });
            }
        }
    </script>
</body>
</html>
'''

if __name__ == '__main__':
    app.run(debug=True)
