import os
from flask import Flask, render_template_string, request, jsonify
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
import tiktoken
from tiktoken import get_encoding
import uuid
import time
import random
from docx import Document

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

# Function to extract text from DOCX
def extract_text_from_docx(file):
    doc = Document(file)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text

# Function to truncate text
def truncate_text(text, max_tokens):
    tokenizer = get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return tokenizer.decode(tokens[:max_tokens])

# Function to count tokens
def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

# Function to get embeddings
def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response.data[0].embedding

def upsert_to_pinecone(text, file_name, file_id, index):
    chunks = [text[i:i+8000] for i in range(0, len(text), 8000)]  # Split into 8000 character chunks
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

# Function to query Pinecone
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

# Function to insert metadata into Pinecone
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

# Function to get all metadata from Pinecone
def get_all_metadata():
    results = index_metadata.query(vector=[0]*1536, top_k=10000, include_metadata=True)
    return [(match['id'], match['metadata']['title'], match['metadata']['tags'], match['metadata']['links']) for match in results['matches']]

# Function to update metadata in Pinecone
def update_metadata(id, title, tags, links):
    metadata = {
        "title": title,
        "tags": tags,
        "links": links
    }
    embedding = get_embedding(f"{title} {tags} {links}")
    index_metadata.upsert(vectors=[(id, embedding, metadata)])

# Function to delete metadata from Pinecone
def delete_metadata(id):
    index_metadata.delete(ids=[id])

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
    truncated_context = truncate_text(context, max_context_tokens)
    
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
    return render_template_string(HOME_TEMPLATE)

@app.route('/metadata')
def metadata():
    return render_template_string(METADATA_TEMPLATE)

@app.route('/get_metadata')
def get_metadata():
    metadata = get_all_metadata()
    return jsonify(metadata)

@app.route('/add_metadata', methods=['POST'])
def add_metadata():
    data = request.json
    success = insert_metadata(data['title'], data['tags'], data['links'])
    return jsonify({'success': success})

@app.route('/update_metadata', methods=['POST'])
def update_metadata_route():
    data = request.json
    update_metadata(data['id'], data['title'], data['tags'], data['links'])
    return jsonify({'success': True})

@app.route('/delete_metadata', methods=['POST'])
def delete_metadata_route():
    data = request.json
    delete_metadata(data['id'])
    return jsonify({'success': True})

@app.route('/chat', methods=['POST'])
def chat():
    user_query = request.json['message']
    final_answer, intent_data, all_keywords = get_answer(user_query)
    return jsonify({
        'answer': final_answer,
        'keywords': all_keywords,
        'related_documents': [
            {
                'id': match['id'],
                'title': match['metadata']['title'],
                'tags': match['metadata']['tags'],
                'links': match['metadata']['links']
            }
            for intent, data in intent_data.items()
            for match in data['metadata_results']
        ]
    })

# HTML Templates
HOME_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>College Buddy Assistant</title>
    <script src="https://cdn.jsdelivr.net/npm/axios/dist/axios.min.js"></script>
    <style>
        body { font-family: Arial, sans-serif; line-height: 1.6; padding: 20px; }
        h1 { color: #333; }
        #chat-container { margin-top: 20px; }
        #user-input { width: 70%; padding: 10px; }
        button { padding: 10px 15px; background-color: #007bff; color: white; border: none; cursor: pointer; }
        #popular-questions-container { margin-top: 20px; }
        .popular-question { margin: 10px 0; padding: 10px; background-color: #f0f0f0; cursor: pointer; }
        #answer-container { margin-top: 20px; border: 1px solid #ddd; padding: 15px; }
    </style>
</head>
<body>
    <h1>College Buddy Assistant</h1>
    <p>Welcome to College Buddy! I am here to help you stay organized, find information fast and provide assistance. Feel free to ask me a question below.</p>
    
    <div id="chat-container">
        <input type="text" id="user-input" placeholder="Ask your question...">
        <button onclick="sendMessage()">Send</button>
    </div>
    
    <div id="answer-container"></div>
    
    <div id="popular-questions">
        <h2>Popular Questions</h2>
        <div id="popular-questions-container"></div>
    </div>
    
    <a href="/metadata">Manage Metadata</a>

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
                const div = document.createElement('div');
                div.className = 'popular-question';
                div.textContent = question;
                div.onclick = () => askPopularQuestion(question);
                container.appendChild(div);
            });
        }

        function askPopularQuestion(question) {
            document.getElementById('user-input').value = question;
            sendMessage();
        }

        function sendMessage() {
            const userInput = document.getElementById('user-input');
            const message = userInput.value.trim();
            if (message) {
                axios.post('/chat', { message: message })
                    .then(response => {
                        displayAnswer(response.data);
                        userInput.value = '';
                    })
                    .catch(error => console.error('Error:', error));
            }
        }

        function displayAnswer(data) {
            const container = document.getElementById('answer-container');
            let html = `<h3>Answer:</h3><p>${data.answer}</p>`;
            html += `<h4>Related Keywords:</h4><p>${data.keywords.join(', ')}</p>`;
            html += '<h4>Related Documents:</h4>';
            data.related_documents.forEach(doc => {
                html += `<div><strong>${doc.title}</strong><br>Tags: ${doc.tags}<br>Link: <a href="${doc.links}" target="_blank">${doc.links}</a></div><br>`;
            });
            container.innerHTML = html;
        }

        loadPopularQuestions();
    </script>
</body>
</html>
'''

METADATA_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Manage Metadata</title>
    <script src="https://cdn.jsdelivr.net/npm/axios/dist/axios.min.js"></script>
    <style>
body { font-family: Arial, sans-serif; line-height: 1.6; padding: 20px; }
        h1, h2 { color: #333; }
        table { width: 100%; border-collapse: collapse; margin-bottom: 20px; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        input[type="text"] { width: 100%; padding: 5px; margin-bottom: 10px; }
        button { padding: 10px 15px; background-color: #007bff; color: white; border: none; cursor: pointer; margin-right: 10px; }
    </style>
</head>
<body>
    <h1>Manage Metadata</h1>
    
    <div id="metadata-table"></div>
    
    <h2>Add Metadata</h2>
    <input type="text" id="new-title" placeholder="Title">
    <input type="text" id="new-tags" placeholder="Tags (comma-separated)">
    <input type="text" id="new-links" placeholder="Links">
    <button onclick="addMetadata()">Add Metadata</button>
    
    <h2>Update Metadata</h2>
    <input type="text" id="update-id" placeholder="ID">
    <input type="text" id="update-title" placeholder="New Title">
    <input type="text" id="update-tags" placeholder="New Tags">
    <input type="text" id="update-links" placeholder="New Links">
    <button onclick="updateMetadata()">Update Metadata</button>
    
    <h2>Delete Metadata</h2>
    <input type="text" id="delete-id" placeholder="ID">
    <button onclick="deleteMetadata()">Delete Metadata</button>
    
    <br><br>
    <a href="/">Back to Home</a>

    <script>
        function loadMetadata() {
            axios.get('/get_metadata')
                .then(response => {
                    const metadata = response.data;
                    let tableHtml = '<table><tr><th>ID</th><th>Title</th><th>Tags</th><th>Links</th></tr>';
                    metadata.forEach(item => {
                        tableHtml += `<tr><td>${item[0]}</td><td>${item[1]}</td><td>${item[2]}</td><td>${item[3]}</td></tr>`;
                    });
                    tableHtml += '</table>';
                    document.getElementById('metadata-table').innerHTML = tableHtml;
                })
                .catch(error => console.error('Error:', error));
        }

        function addMetadata() {
            const title = document.getElementById('new-title').value;
            const tags = document.getElementById('new-tags').value;
            const links = document.getElementById('new-links').value;
            axios.post('/add_metadata', { title, tags, links })
                .then(response => {
                    if (response.data.success) {
                        alert('Metadata added successfully!');
                        loadMetadata();
                    } else {
                        alert('Failed to add metadata. Please ensure all fields are filled.');
                    }
                })
                .catch(error => console.error('Error:', error));
        }

        function updateMetadata() {
            const id = document.getElementById('update-id').value;
            const title = document.getElementById('update-title').value;
            const tags = document.getElementById('update-tags').value;
            const links = document.getElementById('update-links').value;
            axios.post('/update_metadata', { id, title, tags, links })
                .then(response => {
                    if (response.data.success) {
                        alert('Metadata updated successfully!');
                        loadMetadata();
                    } else {
                        alert('Failed to update metadata.');
                    }
                })
                .catch(error => console.error('Error:', error));
        }

        function deleteMetadata() {
            const id = document.getElementById('delete-id').value;
            axios.post('/delete_metadata', { id })
                .then(response => {
                    if (response.data.success) {
                        alert('Metadata deleted successfully!');
                        loadMetadata();
                    } else {
                        alert('Failed to delete metadata.');
                    }
                })
                .catch(error => console.error('Error:', error));
        }

        loadMetadata();
    </script>
</body>
</html>
'''

if __name__ == '__main__':
    app.run(debug=True)
