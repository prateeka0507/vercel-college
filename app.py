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
def extract_text_from_docx(file):
    doc = Document(file)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text

def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response.data[0].embedding

def upsert_to_pinecone(text, file_name, file_id, index):
    chunks = [text[i:i+8000] for i in range(0, len(text), 8000)]
    for i, chunk in enumerate(chunks):
        embedding = get_embedding(chunk)
        metadata = {
            "file_name": file_name,
            "file_id": file_id,
            "chunk_id": i,
            "chunk_text": chunk
        }
        index.upsert(vectors=[(f"{file_id}_{i}", embedding, metadata)])
        time.sleep(1)  # To avoid rate limiting

def insert_metadata(title, tags, links):
    if tags.strip() and links.strip():
        id = str(uuid.uuid4())
        metadata = {
            "title": title,
            "tags": tags,
            "links": links
        }
        embedding = get_embedding(f"{title} {tags} {links}")
        index_metadata.upsert(vectors=[(id, embedding, metadata)])
        return True
    return False

def get_all_metadata():
    results = index_metadata.query(vector=[0]*1536, top_k=10000, include_metadata=True)
    return [(match['id'], match['metadata']['title'], match['metadata']['tags'], match['metadata']['links']) for match in results['matches']]

def update_metadata(id, title, tags, links):
    metadata = {
        "title": title,
        "tags": tags,
        "links": links
    }
    embedding = get_embedding(f"{title} {tags} {links}")
    index_metadata.upsert(vectors=[(id, embedding, metadata)])

def delete_metadata(id):
    index_metadata.delete(ids=[id])

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

def truncate_text(text, max_tokens):
    tokenizer = get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return tokenizer.decode(tokens[:max_tokens])

def generate_multi_intent_answer(query, intent_data):
    context = "\n".join([f"Intent: {intent}\nMetadata Results: {data['metadata_results']}\nPinecone Context: {data['pinecone_context']}" for intent, data in intent_data.items()])
    max_context_tokens = 4000
    truncated_context = truncate_text(context, max_context_tokens)
    
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

def extract_keywords_from_response(response):
    keyword_prompt = f"Extract 5-10 key terms or phrases from this text, separated by commas: {response}"
    keyword_response = client.chat.completions.create(
        model="gpt-3.5-turbo",
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
    return render_template_string(HTML_TEMPLATE, example_questions=EXAMPLE_QUESTIONS)

@app.route('/chat', methods=['POST'])
def chat():
    user_query = request.json['message']
    final_answer, intent_data, all_keywords = get_answer(user_query)
    return jsonify({
        'response': final_answer,
        'keywords': all_keywords,
        'intent_data': intent_data
    })

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    if file and file.filename.endswith('.docx'):
        filename = secure_filename(file.filename)
        file_id = str(uuid.uuid4())
        text = extract_text_from_docx(file)
        token_count = num_tokens_from_string(text)
        upsert_to_pinecone(text, filename, file_id, index_content)
        return jsonify({
            'message': 'File uploaded successfully',
            'filename': filename,
            'file_id': file_id,
            'token_count': token_count
        })
    return jsonify({'error': 'Invalid file format'})

@app.route('/database')
def database():
    return render_template_string(DATABASE_TEMPLATE)

@app.route('/metadata', methods=['GET'])
def get_metadata():
    metadata = get_all_metadata()
    return jsonify(metadata)

@app.route('/metadata', methods=['POST'])
def add_metadata():
    data = request.json
    success = insert_metadata(data['title'], data['tags'], data['links'])
    return jsonify({'success': success})

@app.route('/metadata/<id>', methods=['PUT'])
def update_metadata_route(id):
    data = request.json
    update_metadata(id, data['title'], data['tags'], data['links'])
    return jsonify({'success': True})

@app.route('/metadata/<id>', methods=['DELETE'])
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
            margin: 0;
            padding: 20px;
            max-width: 800px;
            margin: 0 auto;
        }
        h1, h2 {
            color: #333;
        }
        #chat-container {
            border: 1px solid #ddd;
            padding: 20px;
            margin-bottom: 20px;
            height: 300px;
            overflow-y: auto;
        }
        #user-input {
            width: 70%;
            padding: 10px;
        }
        button {
            padding: 10px 20px;
            background-color: #007bff;
            color: white;
            border: none;
            cursor: pointer;
        }
        button:hover {
            background-color: #0056b3;
        }
        #file-upload {
            margin-top: 20px;
        }
        #answer-container {
            margin-top: 20px;
            border: 1px solid #ddd;
            padding: 10px;
            background-color: #f9f9f9;
        }
    </style>
</head>
<body>
    <h1>College Buddy Assistant</h1>
    <div id="chat-container"></div>
    <input type="text" id="user-input" placeholder="Ask your question...">
    <button onclick="sendMessage()">Send</button>
    
    <div id="answer-container"></div>
    
    <h2>Popular Questions</h2>
    <div id="popular-questions">
        {% for question in example_questions %}
            <p onclick="askQuestion('{{ question }}')">{{ question }}</p>
        {% endfor %}
        </div>
        <div id="file-upload">
        <h2>Upload Document</h2>
        <input type="file" id="document-file" accept=".docx">
        <button onclick="uploadDocument()">Upload</button>
    </div>
    
    <button onclick="window.location.href='/database'">Manage Database</button>

    <script>
        function sendMessage() {
            var userInput = document.getElementById('user-input');
            var query = userInput.value.trim();
            if (query !== '') {
                askQuestion(query);
                userInput.value = '';
            }
        }

        function askQuestion(query) {
            var chatContainer = document.getElementById('chat-container');
            var answerContainer = document.getElementById('answer-container');
            
            chatContainer.innerHTML += '<p><strong>You:</strong> ' + query + '</p>';
            
            axios.post('/chat', { message: query })
                .then(function (response) {
                    chatContainer.innerHTML += '<p><strong>College Buddy:</strong> ' + response.data.response + '</p>';
                    chatContainer.scrollTop = chatContainer.scrollHeight;
                    
                    var answerHtml = '<h3>Detailed Answer:</h3>';
                    answerHtml += '<p>' + response.data.response + '</p>';
                    answerHtml += '<h4>Related Keywords:</h4>';
                    answerHtml += '<p>' + response.data.keywords.join(', ') + '</p>';
                    answerHtml += '<h4>Related Documents:</h4>';
                    var documents = new Set();
                    for (var intent in response.data.intent_data) {
                        response.data.intent_data[intent].metadata_results.forEach(function(result) {
                            if (!documents.has(result.id)) {
                                documents.add(result.id);
                                answerHtml += '<p>- ' + result.metadata.title + '</p>';
                            }
                        });
                    }
                    answerContainer.innerHTML = answerHtml;
                })
                .catch(function (error) {
                    console.error('Error:', error);
                    chatContainer.innerHTML += '<p><strong>Error:</strong> Unable to process your question. Please try again.</p>';
                });
        }

        function uploadDocument() {
            var fileInput = document.getElementById('document-file');
            var file = fileInput.files[0];
            if (!file) {
                alert('Please select a file to upload');
                return;
            }

            var formData = new FormData();
            formData.append('file', file);

            axios.post('/upload', formData, {
                headers: {
                    'Content-Type': 'multipart/form-data'
                }
            })
            .then(function (response) {
                alert('File uploaded successfully: ' + response.data.filename);
                fileInput.value = '';
            })
            .catch(function (error) {
                console.error('Error:', error);
                alert('Error uploading file');
            });
        }

        // Add event listener for Enter key
        document.getElementById('user-input').addEventListener('keypress', function (e) {
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
            margin: 0;
            padding: 20px;
            max-width: 800px;
            margin: 0 auto;
        }
        h1, h2 {
            color: #333;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 20px;
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
            padding: 5px;
            margin-bottom: 10px;
        }
        button {
            padding: 10px 20px;
            background-color: #007bff;
            color: white;
            border: none;
            cursor: pointer;
        }
        button:hover {
            background-color: #0056b3;
        }
    </style>
</head>
<body>
    <h1>Database Management</h1>
    <button onclick="window.location.href='/'">Back to Chat</button>

    <div id="metadata-container"></div>
    
    <h2>Add New Metadata</h2>
    <input type="text" id="new-title" placeholder="Title">
    <input type="text" id="new-tags" placeholder="Tags (comma-separated)">
    <input type="text" id="new-links" placeholder="Links">
    <button onclick="addMetadata()">Add Metadata</button>

    <script>
        function loadMetadata() {
            axios.get('/metadata')
                .then(function (response) {
                    var container = document.getElementById('metadata-container');
                    var table = '<table><tr><th>Title</th><th>Tags</th><th>Links</th><th>Actions</th></tr>';
                    response.data.forEach(function(item) {
                        table += '<tr>';
                        table += '<td>' + item[1] + '</td>';
                        table += '<td>' + item[2] + '</td>';
                        table += '<td>' + item[3] + '</td>';
                        table += '<td><button onclick="deleteMetadata(\'' + item[0] + '\')">Delete</button></td>';
                        table += '</tr>';
                    });
                    table += '</table>';
                    container.innerHTML = table;
                })
                .catch(function (error) {
                    console.error('Error:', error);
                });
        }

        function addMetadata() {
            var title = document.getElementById('new-title').value;
            var tags = document.getElementById('new-tags').value;
            var links = document.getElementById('new-links').value;
            
            axios.post('/metadata', {
                title: title,
                tags: tags,
                links: links
            })
            .then(function (response) {
                if (response.data.success) {
                    alert('Metadata added successfully');
                    loadMetadata();
                    document.getElementById('new-title').value = '';
                    document.getElementById('new-tags').value = '';
                    document.getElementById('new-links').value = '';
                } else {
                    alert('Failed to add metadata');
                }
            })
            .catch(function (error) {
                console.error('Error:', error);
                alert('Error adding metadata');
            });
        }

        function deleteMetadata(id) {
            if (confirm('Are you sure you want to delete this metadata?')) {
                axios.delete('/metadata/' + id)
                    .then(function (response) {
                        if (response.data.success) {
                            alert('Metadata deleted successfully');
                            loadMetadata();
                        } else {
                            alert('Failed to delete metadata');
                        }
                    })
                    .catch(function (error) {
                        console.error('Error:', error);
                        alert('Error deleting metadata');
                    });
            }
        }

        // Load metadata when the page loads
        loadMetadata();
    </script>
</body>
</html>
'''

if __name__ == '__main__':
    app.run(debug=True)
