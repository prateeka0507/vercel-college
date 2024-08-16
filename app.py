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
import base64

app = Flask(__name__)
app.secret_key = os.urandom(24)  # For session management
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max-limit

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

def get_background_image():
    image_path = "texas tech image 1.jpg"
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
        return f"data:image/jpeg;base64,{encoded_string}"
    except FileNotFoundError:
        print(f"Background image not found at {image_path}")
        return ""

def get_logo_image():
    logo_path = "Texas_Tech logo 2.png"
    try:
        with open(logo_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
        return f"data:image/png;base64,{encoded_string}"
    except FileNotFoundError:
        print(f"Logo image not found at {logo_path}")
        return ""

# Flask routes
@app.route('/')
def home():
    background_image = get_background_image()
    logo_image = get_logo_image()
    return render_template_string(HTML_TEMPLATE, example_questions=EXAMPLE_QUESTIONS, background_image=background_image, logo_image=logo_image)

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
    background_image = get_background_image()
    logo_image = get_logo_image()
    return render_template_string(DATABASE_TEMPLATE, metadata=metadata, background_image=background_image, logo_image=logo_image)

@app.route('/add_metadata', methods=['POST'])
def add_metadata():
    data = request.json
    success = insert_metadata(data['title'], data['tags'], data['links'])
    return jsonify({'success': success})

@app.route('/delete_metadata/<id>', methods=['DELETE'])
def delete_metadata_route(id):
    delete_metadata(id)
    return jsonify({'success': True})

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        file.save(file_path)
        # Here you can add logic to process the file, e.g., extract text and add to Pinecone
        return jsonify({'success': True, 'filename': filename})
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>College Buddy Assistant</title>
    <script src="https://cdn.jsdelivr.net/npm/axios/dist/axios.min.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600&display=swap" rel="stylesheet">
    <style>
        body {
            font-family: 'Poppins', sans-serif;
            line-height: 1.6;
            color: #5d5d5d;
            margin: 0;
            padding: 0;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            min-height: 100vh;
            justify-content: center;
            align-items: center;
            background-image: url('{{ background_image }}');
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }
        .container {
            max-width: 1200px;
            margin: 20px;
            background-color: rgba(255, 245, 238, 0.9); /* Pastel peach */
            border-radius: 20px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
            overflow: hidden;
            display: grid;
            grid-template-columns: 3fr 1fr;
            flex: 1;
        }
        .main-content, .sidebar {
            padding: 30px;
        }
        h1, h2, h3 {
            color: #7b6079; /* Pastel purple */
        }
        h1 {
            font-size: 2.5em;
            margin-bottom: 20px;
        }
        .header {
            display: flex;
            align-items: center;
            margin-bottom: 20px;
        }
        .logo {
            width: 100px;
            height: auto;
            margin-right: 20px;
        }
        #chat-container {
            height: 400px;
            overflow-y: auto;
            border: 2px solid #b2d8d8; /* Pastel teal */
            padding: 15px;
            margin-bottom: 20px;
            background-color: rgba(255, 255, 255, 0.8);
            border-radius: 15px;
            display: flex;
            flex-direction: column; /* Change to column for top-to-bottom scrolling */
        }
        #user-input {
            width: calc(100% - 120px);
            padding: 12px;
            border: 2px solid #b2d8d8; /* Pastel teal */
            border-radius: 25px;
            font-size: 16px;
            background-color: rgba(255, 255, 255, 0.8);
        }
        button {
            padding: 12px 25px;
            background-color: #aec6cf; /* Pastel blue */
            color: #5d5d5d;
            border: none;
            cursor: pointer;
            transition: all 0.3s ease;
            border-radius: 25px;
            font-size: 16px;
            font-weight: 500;
        }
        button:hover {
            background-color: #f7cac9; /* Pastel pink */
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
        }
        .message {
            margin-bottom: 15px;
            padding: 12px;
            border-radius: 15px;
            max-width: 80%;
            animation: fadeIn 0.5s;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .user-message {
            background-color: #dcd0ff; /* Pastel lavender */
            color: #5d5d5d;
            align-self: flex-start; /* Align to the left */
            margin-right: auto; /* Push to the left */
        }
        .bot-message {
            background-color: #e0f0e3; /* Pastel mint */
            align-self: flex-start;
        }
        .popular-questions {
            margin-top: 30px;
        }
        .popular-question {
            background-color: rgba(255, 223, 211, 0.5); /* Light pastel orange */
            padding: 15px;
            margin-bottom: 15px;
            border-radius: 15px;
            cursor: pointer;
            transition: all 0.3s ease;
            color: #5d5d5d;
        }
        .popular-question:hover {
            background-color: rgba(255, 223, 211, 0.8); /* Darker pastel orange */
            transform: translateY(-3px);
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
        }
        .related-info {
            margin-top: 20px;
            padding: 15px;
            background-color: rgba(255, 255, 255, 0.8);
            border-radius: 15px;
        }
        .sidebar {
            background-color: rgba(255, 245, 238, 0.8); /* Lighter pastel peach */
            border-left: 1px solid #f7cac9; /* Pastel pink */
        }
        .file-upload {
            margin-top: 20px;
        }
        .file-upload input[type="file"] {
            display: none;
        }
        .file-upload label {
            display: inline-block;
            padding: 10px 20px;
            background-color: #aec6cf; /* Pastel blue */
            color: #5d5d5d;
            border-radius: 25px;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        .file-upload label:hover {
            background-color: #f7cac9; /* Pastel pink */
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
        }
        #upload-status {
            margin-top: 10px;
            font-style: italic;
        }
        .admin-controls {
            margin-top: 20px;
        }
        .admin-controls button {
            width: 100%;
            margin-bottom: 10px;
        }
        .hidden {
            display: none;
        }
        @media (max-width: 768px) {
            .container {
                grid-template-columns: 1fr;
            }
            .sidebar {
                border-left: none;
                border-top: 1px solid #f7cac9; /* Pastel pink */
            }

        }
        .copyright {
            text-align: center;
            padding: 15px;
            background-color: rgba(255, 245, 238, 0.9);
            color: #7b6079;
            font-size: 14px;
            border-top: 1px solid #f7cac9;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="main-content">
            <div class="header">
                <img src="{{ logo_image }}" alt="Texas Tech University Logo" class="logo">
                <h1>College Buddy Assistant</h1>
            </div>
            <p>Welcome to College Buddy! I'm here to help you stay organized, find information fast, and provide assistance. Feel free to ask me a question below.</p>
            
            <div id="chat-container"></div>
            
            <input type="text" id="user-input" placeholder="Ask your question...">
            <button onclick="sendMessage()">Send</button>
            
            <div class="related-info" id="related-info"></div>
        </div>
        
        <div class="sidebar">
            <div class="popular-questions">
                <h3>Popular Questions</h3>
                <div id="popular-questions-container"></div>
            </div>

            <div class="admin-controls">
                <button onclick="toggleAdminControls()">Admin Controls</button>
                <div id="admin-buttons" class="hidden">
                    <button onclick="window.location.href='/database'" style="width: 100%; margin-bottom: 10px;">Manage Database</button>
                    <div class="file-upload">
                        <label for="file-input">Upload Document</label>
                        <input type="file" id="file-input" onchange="uploadFile()">
                    </div>
                </div>
            </div>
            <div id="upload-status"></div>
        </div>
        <div class="copyright">
            &copy; 2024 KLM Solutions. All rights reserved.
        </div>
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

        function displayRelatedInfo(intentData) {
            const relatedInfo = document.getElementById('related-info');
            relatedInfo.innerHTML = '<h3>Related Information:</h3>';
            
            for (const [intent, data] of Object.entries(intentData)) {
                if (data.related_documents.length > 0 || data.related_links.length > 0) {
                    const intentInfo = document.createElement('div');
                    intentInfo.innerHTML = `<h4>${intent}</h4>`;
                    
                    if (data.related_documents.length > 0) {
                        intentInfo.innerHTML += '<p><strong>Related Documents:</strong> ' + data.related_documents.join(', ') + '</p>';
                    }
                    
                    if (data.related_links.length > 0) {
                        intentInfo.innerHTML += '<p><strong>Related Links:</strong> ' + data.related_links.map(link => `<a href="${link}" target="_blank">${link}</a>`).join(', ') + '</p>';
                    }
                    
                    relatedInfo.appendChild(intentInfo);
                }
            }
        }

        function sendMessage() {
            const userInput = document.getElementById('user-input');
            const message = userInput.value;
            if (message.trim() === '') return;

            addMessageToChat('You', message, 'user-message');
            
            // Create a placeholder for the bot's response
            const botMessageElement = document.createElement('div');
            botMessageElement.className = 'message bot-message';
            botMessageElement.innerHTML = '<strong>College Buddy:</strong> <span id="bot-response"></span>';
            document.getElementById('chat-container').appendChild(botMessageElement);

            axios.post('/chat', { message: message })
                .then(response => {
                    simulateStreaming(response.data.response, 'bot-response');
                    displayRelatedInfo(response.data.intent_data);
                    userInput.value = '';
                })
                .catch(error => {
                    console.error('Error:', error);
                    document.getElementById('bot-response').textContent = 'Sorry, I encountered an error. Please try again.';
                });
        }

        function simulateStreaming(text, elementId) {
            const element = document.getElementById(elementId);
            let index = 0;

            function addNextChar() {
                if (index < text.length) {
                    element.textContent += text.charAt(index);
                    index++;
                    setTimeout(addNextChar, 20); // Adjust the delay as needed
                }
            }

            addNextChar();
        }

        function addMessageToChat(sender, message, className) {
            const chatContainer = document.getElementById('chat-container');
            const messageElement = document.createElement('div');
            messageElement.className = `message ${className}`;
            messageElement.innerHTML = `<strong>${sender}:</strong> ${message}`;
            chatContainer.appendChild(messageElement);
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }

        function uploadFile() {
            const fileInput = document.getElementById('file-input');
            const file = fileInput.files[0];
            if (!file) {
                return;
            }

            const formData = new FormData();
            formData.append('file', file);

            const statusElement = document.getElementById('upload-status');
            statusElement.textContent = 'Uploading...';

            axios.post('/upload', formData, {
                headers: {
                    'Content-Type': 'multipart/form-data'
                }
            })
            .then(function (response) {
                if (response.data.success) {
                    statusElement.textContent = 'File uploaded successfully: ' + response.data.filename;
                } else {
                    statusElement.textContent = 'Upload failed: ' + response.data.error;
                }
            })
            .catch(function (error) {
                console.error('Error:', error);
                statusElement.textContent = 'An error occurred during upload';
            });

            fileInput.value = ''; // Reset file input
        }

        function toggleAdminControls() {
            const adminButtons = document.getElementById('admin-buttons');
            adminButtons.classList.toggle('hidden');
        }

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
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600&display=swap" rel="stylesheet">
    <style>
        body {
            font-family: 'Poppins', sans-serif;
            line-height: 1.6;
            color: #5d5d5d;
            margin: 0;
            padding: 20px;
            display: flex;
            flex-direction: column;
            min-height: 100vh;
            justify-content: center;
            align-items: flex-start;
            background-image: url('{{ background_image }}');
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }
        .container {
            width: 100%;
            max-width: 1000px;
            background-color: rgba(255, 245, 238, 0.95); /* Pastel peach */
            border-radius: 20px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
            overflow: hidden;
            padding: 30px;
            flex: 1;
        }
        .card {
            background-color: rgba(255, 255, 255, 0.9);
            border-radius: 15px;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.05);
            padding: 25px;
            margin-bottom: 30px;
        }
        h1, h2 {
            color: #7b6079; /* Pastel purple */
            margin-top: 0;
        }
        h1 {
            font-size: 2.5em;
            margin-bottom: 20px;
        }
        .header {
            display: flex;
            align-items: center;
            margin-bottom: 20px;
        }
                .logo {
            width: 100px;
            height: auto;
            margin-right: 20px;
        }
        .table-container {
            overflow-x: auto;
            margin-top: 20px;
            border-radius: 15px;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.05);
        }
        table {
            width: 100%;
            border-collapse: separate;
            border-spacing: 0;
            background-color: rgba(255, 255, 255, 0.9);
            table-layout: fixed; /* Added to ensure consistent column widths */
        }
        th, td {
            padding: 15px;
            text-align: left;
            border-bottom: 1px solid #e0f0e3; /* Pastel mint */
            word-wrap: break-word; /* Allow long words to break and wrap */
            overflow-wrap: break-word; /* Alternative for word-wrap */
        }
        th {
            background-color: #aec6cf; /* Pastel blue */
            color: #5d5d5d;
            font-weight: 500;
            position: sticky;
            top: 0;
        }
        tr:last-child td {
            border-bottom: none;
        }
        tr:hover {
            background-color: rgba(255, 255, 255, 0.95);
        }
        input[type="text"] {
            width: 100%;
            padding: 12px;
            margin: 8px 0;
            border: 2px solid #b2d8d8; /* Pastel teal */
            border-radius: 25px;
            box-sizing: border-box;
            font-size: 16px;
            background-color: rgba(255, 255, 255, 0.8);
        }
        button {
            padding: 12px 25px;
            background-color: #aec6cf; /* Pastel blue */
            color: #5d5d5d;
            border: none;
            cursor: pointer;
            transition: all 0.3s ease;
            border-radius: 25px;
            font-size: 16px;
            font-weight: 500;
        }
        button:hover {
            background-color: #f7cac9; /* Pastel pink */
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
        }
        .form-group {
            margin-bottom: 20px;
        }
        .action-buttons {
            display: flex;
            justify-content: flex-start;
            margin-top: 20px;
            margin-bottom: 20px;
        }
        .delete-btn {
            background-color: #ffd1dc; /* Light pastel pink */
            padding: 8px 15px;
            font-size: 14px;
        }
        .delete-btn:hover {
            background-color: #ffb3ba; /* Darker pastel pink */
        }
        @media (max-width: 768px) {
            .container {
                padding: 20px;
            }
            .card {
                padding: 20px;
            }
        }
        .copyright {
            text-align: center;
            padding: 15px;
            background-color: rgba(255, 245, 238, 0.9);
            color: #7b6079;
            font-size: 14px;
            border-top: 1px solid #f7cac9;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <img src="{{ logo_image }}" alt="Texas Tech University Logo" class="logo">
            <h1>Database Management</h1>
        </div>
        <div class="action-buttons">
            <button onclick="window.location.href='/'">Back to Chat</button>
        </div>

        <div class="card">
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
        </div>

        <div class="card">
            <h2>Existing Documents</h2>
            <div class="table-container">
                <table id="metadata-table">
                    <thead>
                        <tr>
                            <th style="width: 25%;">Title</th>
                            <th style="width: 35%;">Tags</th>
                            <th style="width: 30%;">Links</th>
                            <th style="width: 10%;">Action</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for item in metadata %}
                        <tr>
                            <td>{{ item.title }}</td>
                            <td>{{ item.tags }}</td>
                            <td>{{ item.links }}</td>
                            <td><button class="delete-btn" onclick="deleteMetadata('{{ item.id }}')">Delete</button></td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </div>
        </div>
        <div class="copyright">
            &copy; 2024 KLM Solutions. All rights reserved.
        </div>
    </div>
    </div>

    <script>
        function addMetadata() {
            const title = document.getElementById('new-title').value;
            const tags = document.getElementById('new-tags').value;
            const links = document.getElementById('new-links').value;
            
            if (!title || !tags || !links) {
                alert('Please fill in all fields');
                return;
            }
            
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
            'pinecone_context': pinecone_context,
            'related_documents': [result['metadata'].get('title', '') for result in new_metadata_results],
            'related_links': [result['metadata'].get('links', '') for result in new_metadata_results]
        }
    return intent_data

def generate_multi_intent_answer(query, intent_data):
    context = "\n".join([
        f"Intent: {intent}\n"
        f"Metadata Results: {data['metadata_results']}\n"
        f"Pinecone Context: {data['pinecone_context']}\n"
        f"Related Documents: {', '.join(data['related_documents'])}\n"
        f"Related Links: {', '.join(data['related_links'])}"
        for intent, data in intent_data.items()
    ])
    max_context_tokens = 4000
    tokenizer = get_encoding("cl100k_base")
    truncated_context = tokenizer.decode(tokenizer.encode(context)[:max_context_tokens])
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": """You are College Buddy, an AI assistant designed to help students with their academic queries. Your primary function is to analyze and provide insights based on the context of uploaded documents. Please adhere to the following guidelines:
1. Focus on addressing all the intents of the query and give related documents for all the intents.
2. Provide accurate, relevant information derived from the provided context.
3. If the context doesn't contain sufficient information to answer the query, state this clearly.
4. Maintain a friendly, supportive tone appropriate for assisting students.
5. Provide concise yet comprehensive answers, breaking down complex concepts when necessary.
6. If asked about topics beyond the scope of the provided context, politely state that you don't have that information.
7. Encourage critical thinking by guiding students towards understanding rather than simply providing direct answers.
8. Respect academic integrity by not writing essays or completing assignments on behalf of students.
9. Suggest additional resources or related documents when relevant to the query.
10. Include related links in your response when they provide valuable additional information."""},
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
                'pinecone_context': data['pinecone_context'],
                'related_documents': data['related_documents'],
                'related_links': data['related_links']
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

if __name__ == '__main__':
    app.run(debug=True)
