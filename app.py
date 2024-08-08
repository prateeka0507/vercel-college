import os
import time
from flask import Flask, render_template_string, request, jsonify, current_app
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
import tiktoken
from tiktoken import get_encoding
from difflib import SequenceMatcher

app = Flask(__name__)

# Access your API keys from environment variables
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
INDEX_NAME = "college-buddy"

# HTML Templates (keep your existing HTML_TEMPLATE and DATABASE_TEMPLATE here)
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>College Buddy Assistant</title>
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
        // Full list of example questions
        const allPopularQuestions = [
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
        ];

        // Function to get random items from an array
        function getRandomItems(arr, count) {
            const shuffled = arr.sort(() => 0.5 - Math.random());
            return shuffled.slice(0, count);
        }

        // Populate popular questions with 3 random questions
        const popularQuestionsContainer = document.getElementById('popular-questions-container');
        const randomQuestions = getRandomItems(allPopularQuestions, 3);
        
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
            
            fetch('/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ message: message }),
            })
            .then(response => response.json())
            .then(data => {
                addMessageToChat('College Buddy', data.response, 'bot-message');
                userInput.value = '';
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
        form {
            margin-top: 20px;
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
    </style>
</head>
<body>
    <div class="container">
        <h1>Database Management</h1>
        <button onclick="window.location.href='/'">Back to Chat</button>

        <h2>Add New Document</h2>
        <form id="add-document-form">
            <input type="text" id="doc-title" placeholder="Document Title" required>
            <input type="text" id="doc-tags" placeholder="Tags (comma-separated)" required>
            <input type="text" id="doc-links" placeholder="Links" required>
            <button type="submit">Add Document</button>
        </form>

        <h2>Existing Documents</h2>
        <table id="documents-table">
            <thead>
                <tr>
                    <th>ID</th>
                    <th>Title</th>
                    <th>Tags</th>
                    <th>Links</th>
                    <th>Action</th>
                </tr>
            </thead>
            <tbody>
                <!-- Documents will be populated here -->
            </tbody>
        </table>
    </div>

    <script>
        // Function to load and display documents
        function loadDocuments() {
            fetch('/documents')
                .then(response => response.json())
                .then(documents => {
                    const tableBody = document.querySelector('#documents-table tbody');
                    tableBody.innerHTML = '';
                    documents.forEach(doc => {
                        const row = `<tr>
                            <td>${doc[0]}</td>
                            <td>${doc[1]}</td>
                            <td>${doc[2]}</td>
                            <td>${doc[3]}</td>
                            <td><button onclick="deleteDocument(${doc[0]})">Delete</button></td>
                        </tr>`;
                        tableBody.innerHTML += row;
                    });
                });
        }

        // Function to add a new document
        document.getElementById('add-document-form').onsubmit = function(e) {
            e.preventDefault();
            const title = document.getElementById('doc-title').value;
            const tags = document.getElementById('doc-tags').value;
            const links = document.getElementById('doc-links').value;

            fetch('/add_document', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ title, tags, links }),
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    alert('Document added successfully');
                    loadDocuments();
                    // Clear form fields
                    document.getElementById('doc-title').value = '';
                    document.getElementById('doc-tags').value = '';
                    document.getElementById('doc-links').value = '';
                } else {
                    alert('Failed to add document');
                }
            });
        };
        // Function to delete a document
        function deleteDocument(id) {
            if (confirm('Are you sure you want to delete this document?')) {
                fetch('/delete_document', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ id }),
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
        loadDocuments();
    </script>
</body>
</html>
'''

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/chat', methods=['POST'])
def chat():
    start_time = time.time()
    try:
        user_query = request.json['message']
        current_app.logger.info(f"Received query: {user_query}")

        # Initialize API clients
        client = OpenAI(api_key=OPENAI_API_KEY)
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(INDEX_NAME)
        
        current_app.logger.info("API clients initialized")

        # Simplified processing for testing
        simple_response = simplified_chat(user_query, client)
        current_app.logger.info(f"Simple response generated: {simple_response[:100]}...")

        # If simple chat works, try the full process
        if simple_response:
            answer, intent_data = get_answer(user_query, client, pc, index)
            current_app.logger.info("Full answer generated")
        else:
            answer = "Failed to generate a simple response."
            intent_data = {}

        end_time = time.time()
        current_app.logger.info(f"Total processing time: {end_time - start_time:.2f} seconds")

        return jsonify({'response': answer, 'intent_data': intent_data})
    except Exception as e:
        current_app.logger.error(f"Error in chat route: {str(e)}")
        return jsonify({'error': str(e)}), 500

def simplified_chat(query, client):
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": query}
            ],
            timeout=5  # 5-second timeout
        )
        return response.choices[0].message.content
    except Exception as e:
        current_app.logger.error(f"Error in simplified_chat: {str(e)}")
        return None

def get_answer(query, client, pc, index):
    try:
        current_app.logger.info("Starting get_answer function")
        
        intents = identify_intents(query, client)
        current_app.logger.info(f"Identified intents: {intents}")
        
        intent_keywords = generate_keywords_per_intent(intents, client)
        current_app.logger.info(f"Generated keywords: {intent_keywords}")
        
        intent_data = query_for_multiple_intents(intent_keywords, pc, index)
        current_app.logger.info("Query for multiple intents completed")
        
        answer = generate_multi_intent_answer(query, intent_data, client)
        current_app.logger.info("Multi-intent answer generated")
        
        return answer, intent_data
    except Exception as e:
        current_app.logger.error(f"Error in get_answer: {str(e)}")
        raise

def identify_intents(query, client):
    intent_prompt = f"Identify the main intent or question within this query. Provide only one primary intent: {query}"
    intent_response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an intent identification assistant. Identify and provide only the primary intent or question within the given query."},
            {"role": "user", "content": intent_prompt}
        ],
        timeout=5
    )
    intent = intent_response.choices[0].message.content.strip()
    return [intent] if intent else []

def generate_keywords_per_intent(intents, client):
    intent_keywords = {}
    for intent in intents:
        keyword_prompt = f"Generate 5-10 relevant keywords or phrases for this intent, separated by commas: {intent}"
        keyword_response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a keyword extraction assistant. Generate relevant keywords or phrases for the given intent."},
                {"role": "user", "content": keyword_prompt}
            ],
            timeout=5
        )
        keywords = keyword_response.choices[0].message.content.strip().split(',')
        intent_keywords[intent] = [keyword.strip() for keyword in keywords]
    return intent_keywords

def query_for_multiple_intents(intent_keywords, pc, index):
    intent_data = {}
    for intent, keywords in intent_keywords.items():
        pinecone_context = query_pinecone(" ".join(keywords), index)
        intent_data[intent] = {
            'pinecone_context': pinecone_context
        }
    return intent_data

def query_pinecone(query, index):
    query_embedding = get_embedding(query)
    results = index.query(vector=query_embedding, top_k=5, include_metadata=True)
    contexts = []
    for match in results['matches']:
        if 'chunk_text' in match['metadata']:
            contexts.append(match['metadata']['chunk_text'])
        else:
            contexts.append(f"Content from {match['metadata'].get('file_name', 'unknown file')}")
    return " ".join(contexts)

def get_embedding(text):
    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response.data[0].embedding

def generate_multi_intent_answer(query, intent_data, client):
    context = "\n".join([f"Intent: {intent}\nPinecone Context: {data['pinecone_context']}" for intent, data in intent_data.items()])
    max_context_tokens = 4000
    tokenizer = get_encoding("cl100k_base")
    truncated_context = tokenizer.decode(tokenizer.encode(context)[:max_context_tokens])
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are College Buddy, an AI assistant designed to help students with their academic queries..."},
            {"role": "user", "content": f"Query: {query}\n\nContext: {truncated_context}"}
        ],
        timeout=15
    )
   
    return response.choices[0].message.content.strip()

@app.route('/database')
def database_management():
    return render_template_string(DATABASE_TEMPLATE)

@app.route('/documents')
def get_documents():
    # This would typically query your database. For now, return an empty list.
    return jsonify([])

@app.route('/add_document', methods=['POST'])
def add_document():
    # This would typically add a document to your database. For now, just return success.
    return jsonify({'success': True})

@app.route('/delete_document', methods=['POST'])
def delete_document():
    # This would typically delete a document from your database. For now, just return success.
    return jsonify({'success': True})

if __name__ == '__main__':
    app.run(debug=True)
