import os
from flask import Flask, render_template_string, request, jsonify, session
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
import tiktoken
from tiktoken import get_encoding
import uuid
import time
import random
import sqlite3
from difflib import SequenceMatcher
from werkzeug.utils import secure_filename
from docx import Document

app = Flask(__name__)
app.secret_key = os.urandom(24)  # For session management

# Access your API keys (set these in Vercel environment variables)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
INDEX_NAME = "college-buddy"

# Initialize OpenAI
client = OpenAI(api_key=OPENAI_API_KEY)

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Create or connect to the Pinecone index
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=1536,
        metric='cosine',
        spec=ServerlessSpec(cloud='aws', region='us-east-1')
    )
index = pc.Index(INDEX_NAME)

# Database functions
def get_database_connection():
    db_path = os.path.join(os.getcwd(), 'college_buddy.db')
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    return conn

def init_db(conn):
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS documents
                 (id INTEGER PRIMARY KEY, title TEXT, tags TEXT, links TEXT)''')
    conn.commit()

def load_initial_data():
    conn = get_database_connection()
    data = [
        # ... (keep your existing initial data)
    ]
    c = conn.cursor()
    c.executemany("INSERT OR REPLACE INTO documents (id, title, tags, links) VALUES (?, ?, ?, ?)", data)
    conn.commit()

def insert_document(title, tags, links):
    if tags.strip() and links.strip():
        conn = get_database_connection()
        c = conn.cursor()
        c.execute("INSERT INTO documents (title, tags, links) VALUES (?, ?, ?)",
                  (title, tags, links))
        conn.commit()
        return True
    return False

def get_all_documents():
    conn = get_database_connection()
    c = conn.cursor()
    c.execute("SELECT id, title, tags, links FROM documents WHERE tags != '' AND links != ''")
    return c.fetchall()

# NLP functions
def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response.data[0].embedding

def query_pinecone(query, top_k=5):
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
        model="gpt-4o",
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
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a keyword extraction assistant. Generate relevant keywords or phrases for the given intent."},
                {"role": "user", "content": keyword_prompt}
            ]
        )
        keywords = keyword_response.choices[0].message.content.strip().split(',')
        intent_keywords[intent] = [keyword.strip() for keyword in keywords]
    return intent_keywords

def query_db_for_keywords(keywords):
    conn = get_database_connection()
    c = conn.cursor()
    query = """
    SELECT DISTINCT id, title, tags, links 
    FROM documents 
    WHERE tags LIKE ?
    """
    results = []
    for keyword in keywords:
        c.execute(query, (f'%{keyword}%',))
        for row in c.fetchall():
            score = sum(SequenceMatcher(None, keyword.lower(), tag.lower()).ratio() for tag in row[2].split(','))
            results.append((score, row))
    
    results.sort(reverse=True, key=lambda x: x[0])
    return results[:3]

def query_for_multiple_intents(intent_keywords):
    intent_data = {}
    all_db_results = set()
    for intent, keywords in intent_keywords.items():
        db_results = query_db_for_keywords(keywords)
        new_db_results = [result for result in db_results if result[1][0] not in [r[1][0] for r in all_db_results]]
        all_db_results.update(new_db_results)
        pinecone_context = query_pinecone(" ".join(keywords))
        intent_data[intent] = {
            'db_results': new_db_results,
            'pinecone_context': pinecone_context
        }
    return intent_data

def generate_multi_intent_answer(query, intent_data):
    context = "\n".join([f"Intent: {intent}\nDB Results: {data['db_results']}\nPinecone Context: {data['pinecone_context']}" for intent, data in intent_data.items()])
    max_context_tokens = 4000
    tokenizer = get_encoding("cl100k_base")
    truncated_context = tokenizer.decode(tokenizer.encode(context)[:max_context_tokens])
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are College Buddy, an AI assistant designed to help students with their academic queries..."},
            {"role": "user", "content": f"Query: {query}\n\nContext: {truncated_context}"}
        ]
    )
   
    return response.choices[0].message.content.strip()

def get_answer(query):
    intents = identify_intents(query)
    intent_keywords = generate_keywords_per_intent(intents)
    intent_data = query_for_multiple_intents(intent_keywords)
    answer = generate_multi_intent_answer(query, intent_data)
    return answer, intent_data

# New function to handle file upload and processing
def process_uploaded_file(file):
    if file and file.filename.endswith('.docx'):
        filename = secure_filename(file.filename)
        file_path = os.path.join('/tmp', filename)
        file.save(file_path)
        
        doc = Document(file_path)
        text = "\n".join([para.text for para in doc.paragraphs])
        
        file_id = str(uuid.uuid4())
        chunks = [text[i:i+8000] for i in range(0, len(text), 8000)]
        
        for i, chunk in enumerate(chunks):
            embedding = get_embedding(chunk)
            metadata = {
                "file_name": filename,
                "file_id": file_id,
                "chunk_id": i,
                "chunk_text": chunk
            }
            index.upsert(vectors=[(f"{file_id}_{i}", embedding, metadata)])
            time.sleep(1)  # To avoid rate limiting
        
        os.remove(file_path)
        return True, file_id
    return False, None

# Flask routes
@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/chat', methods=['POST'])
def chat():
    user_query = request.json['message']
    answer, intent_data = get_answer(user_query)
    
    # Add to chat history
    if 'chat_history' not in session:
        session['chat_history'] = []
    session['chat_history'].append((user_query, answer))
    session['chat_history'] = session['chat_history'][-5:]  # Keep only the last 5 conversations
    session.modified = True
    
    return jsonify({'response': answer, 'intent_data': intent_data})

@app.route('/upload_document', methods=['POST'])
def upload_document():
    if 'file' not in request.files:
        return jsonify({'success': False, 'message': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'message': 'No selected file'})
    
    success, file_id = process_uploaded_file(file)
    if success:
        return jsonify({'success': True, 'message': 'File uploaded and processed successfully', 'file_id': file_id})
    else:
        return jsonify({'success': False, 'message': 'Invalid file format'})

@app.route('/get_chat_history')
def get_chat_history():
    return jsonify(session.get('chat_history', []))

@app.route('/database')
def database_management():
    return render_template_string(DATABASE_TEMPLATE)

@app.route('/documents')
def get_documents():
    documents = get_all_documents()
    return jsonify(documents)

@app.route('/add_document', methods=['POST'])
def add_document():
    data = request.json
    success = insert_document(data['title'], data['tags'], data['links'])
    return jsonify({'success': success})

@app.route('/delete_document', methods=['POST'])
def delete_document():
    data = request.json
    conn = get_database_connection()
    c = conn.cursor()
    c.execute("DELETE FROM documents WHERE id = ?", (data['id'],))
    success = c.rowcount > 0
    conn.commit()
    return jsonify({'success': success})

# HTML Templates
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>College Buddy Assistant</title>
    <style>
        body { font-family: Arial, sans-serif; line-height: 1.6; padding: 20px; }
        h1, h2 { color: #333; }
        #chat-container { margin-top: 20px; }
        #user-input { width: 80%; padding: 10px; }
        button { padding: 10px; background-color: #007bff; color: white; border: none; cursor: pointer; }
        .message { margin: 10px 0; padding: 10px; border-radius: 5px; }
        .user-message { background-color: #f0f0f0; }
        .bot-message { background-color: #e6f2ff; }
        #file-upload { margin-top: 20px; }
        #chat-history { margin-top: 20px; border-top: 1px solid #ddd; padding-top: 10px; }
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
        
        <div id="file-upload">
            <h2>Upload Document</h2>
            <input type="file" id="document-file" accept=".docx">
            <button onclick="uploadDocument()">Upload</button>
        </div>
        
        <div id="chat-history">
            <h2>Recent Conversations</h2>
            <ul id="history-list"></ul>
        </div>
        
        <div class="sidebar">
            <h2>Manage Database</h2>
            <button onclick="window.location.href='/database'">View/Manage Database</button>
        </div>
    </div>

    <script>
        const popularQuestions = [
            "What are the steps to declare a major at Texas Tech University",
            "What are the GPA and course requirements for declaring a major in the Rawls College of Business?",
            "How can new students register for the Red Raider Orientation (RRO)",
            "What are the key components of the Texas Tech University Code of Student Conduct",
            "What resources are available for students reporting incidents of misconduct at Texas Tech University"
        ];

        function loadPopularQuestions() {
            const container = document.getElementById('popular-questions-container');
            popularQuestions.forEach(question => {
                const button = document.createElement('button');
                button.textContent = question;
                button.onclick = () => sendMessage(question);
                container.appendChild(button);
            });
        }

        function sendMessage(question = null) {
            const userInput = document.getElementById('user-input');
            const message = question || userInput.value;
            if (message.trim() === '') return;

            addMessageToChat('You: ' + message, 'user-message');
            
            fetch('/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({message: message}),
            })
            .then(response => response.json())
            .then(data => {
                addMessageToChat('College Buddy: ' + data.response, 'bot-message');
                userInput.value = '';
                loadChatHistory();
            });
        }

        function addMessageToChat(message, className) {
            const chatContainer = document.getElementById('chat-container');
            const messageElement = document.createElement('div');
            messageElement.classList.add('message', className);
            messageElement.textContent = message;
            chatContainer.appendChild(messageElement);
        }

        function uploadDocument() {
            const fileInput = document.getElementById('document-file');
            const file = fileInput.files[0];
            if (!file) {
                alert('Please select a file to upload');
                return;
            }

            const formData = new FormData();
            formData.append('file', file);

            fetch('/upload_document', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    alert(data.message);
                    fileInput.value = '';
                } else {
                    alert('Upload failed: ' + data.message);
                }
            });
        }

        function loadChatHistory() {
            fetch('/get_chat_history')
            .then(response => response.json())
            .then(history => {
                const historyList = document.getElementById('history-list');
                historyList.innerHTML = '';
                history.forEach(([question, answer]) => {
                    const li = document.createElement('li');
                    li.innerHTML = `<strong>Q:</strong> ${question}<br><strong>A:</strong> ${answer}`;
                    historyList.appendChild(li);
                });
            });
        }

        // Load popular questions and chat history when the page loads
        window.onload = function() {
            loadPopularQuestions();
            loadChatHistory();
        };
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
    <title>College Buddy Database Management</title>
    <style>
        body { font-family: Arial, sans-serif; line-height: 1.6; padding: 20px; }
        h1, h2 { color: #333; }
        table { width: 100%; border-collapse: collapse; margin-top: 20px; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        input[type="text"] { width: 100%; padding: 5px; }
        button { padding: 10px; background-color: #007bff; color: white; border: none; cursor: pointer; margin-top: 10px; }
    </style>
</head>
<body>
    <h1>College Buddy Database Management</h1>
    
    <h2>Document Metadata</h2>
    <table id="documents-table">
        <thead>
            <tr>
                <th>ID</th>
                <th>Title</th>
                <th>Tags</th>
                <th>Links</th>
                <th>Actions</th>
            </tr>
        </thead>
        <tbody></tbody>
    </table>

    <h2>Add New Document</h2>
    <input type="text" id="new-title" placeholder="Title">
    <input type="text" id="new-tags" placeholder="Tags (comma-separated)">
    <input type="text" id="new-links" placeholder="Links">
    <button onclick="addDocument()">Add Document</button>

    <script>
        function loadDocuments() {
            fetch('/documents')
                .then(response => response.json())
                .then(documents => {
                    const tbody = document.querySelector('#documents-table tbody');
                    tbody.innerHTML = '';
                    documents.forEach(doc => {
                        const row = `
                            <tr>
                                <td>${doc[0]}</td>
                                <td>${doc[1]}</td>
                                <td>${doc[2]}</td>
                                <td>${doc[3]}</td>
                                <td><button onclick="deleteDocument(${doc[0]})">Delete</button></td>
                            </tr>
                        `;
                        tbody.innerHTML += row;
                    });
                });
        }

        function addDocument() {
            const title = document.getElementById('new-title').value;
            const tags = document.getElementById('new-tags').value;
            const links = document.getElementById('new-links').value;

            fetch('/add_document', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ title, tags, links })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    alert('Document added successfully');
                    loadDocuments();
                    document.getElementById('new-title').value = '';
                    document.getElementById('new-tags').value = '';
                    document.getElementById('new-links').value = '';
                } else {
                    alert('Failed to add document');
                }
            });
        }

        function deleteDocument(id) {
            if (confirm('Are you sure you want to delete this document?')) {
                fetch('/delete_document', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ id })
                })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        alert('Document deleted successfully');
                        loadDocuments();
                    } else {
                        alert('Failed to delete document');
                    }
                });
            }
        }

        // Load documents when the page loads
        window.onload = loadDocuments;
    </script>
</body>
</html>
'''

if __name__ == '__main__':
    # Initialize database and load initial data
    conn = get_database_connection()
    init_db(conn)
    load_initial_data()
    app.run(debug=True)
    
