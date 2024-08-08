import os
from flask import Flask, render_template_string, request, jsonify
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
import tiktoken
from tiktoken import get_encoding
import uuid
import time
import random
import sqlite3
from difflib import SequenceMatcher

app = Flask(__name__)

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
        (1, "TEXAS TECH", "Universities, Texas Tech University, College Life, Student Wellness, Financial Tips for Students, Campus Activities, Study Strategies", "https://www.ttu.edu/"),
        (2, "ADVISING", "Advising, Campus Advising, Registration, Financial Management, Raider Success Hub, Degree Works, Visual Schedule Builder", "https://www.depts.ttu.edu/advising/current-students/advising/"),
        (3, "COURSE PREFIXES", "courses, Undergraduate Degrees, Academic Programs, Degree Concentrations, College Majors, University Programs, Bachelor's Degrees", "https://www.depts.ttu.edu/advising/current-students/course-prefixes/"),
        (4, "NEW STUDENT", "New Student Information, University Advising, Red Raider Orientation, TTU New Students, Academic Advising, Career Planning, Student Success", "https://www.depts.ttu.edu/advising/current-students/new-student-information/"),
        (5, "DECLARE YOUR MAJOR", "Declaring your major, Major Declaration, Academic Transfer Form, College Requirements, GPA Requirements, Advisor Appointment, Major Transfer Process", "https://www.depts.ttu.edu/advising/current-students/declare-your-major/"),
        (6, "Texas Tech University Students Handbook-chunk 1", "Students Handbook, Student Conduct, Hearing Panel, Disciplinary Procedures, University Policy, Academic Integrity, Student Rights", "https://www.depts.ttu.edu/dos/Studenthandbook2022forward/Student-Handbook-2023-2024.pdf"),
        (7, "Texas Tech University Students Handbook-chunk 2", "Students Handbook, Texas Tech University, Student Conduct Code, University Policies, Academic Integrity, Misconduct Reporting, FERPA Privacy", "https://www.depts.ttu.edu/dos/Studenthandbook2022forward/Student-Handbook-2023-2024.pdf"),
        (8, "Texas Tech University Students Handbook-chunk 3", "Students Handbook, Student Conduct, University Policies, Code of Conduct, Disciplinary Procedures, Student Rights, University Regulations", "https://www.depts.ttu.edu/dos/Studenthandbook2022forward/Student-Handbook-2023-2024.pdf"),
        (9, "Texas Tech University Students Handbook-chunk 4", "Students Handbook, Student Conduct Procedures, Conduct Investigations, Disciplinary Actions, University Adjudication, Student Rights and Responsibilities, Conduct Hearings", "https://www.depts.ttu.edu/dos/Studenthandbook2022forward/Student-Handbook-2023-2024.pdf"),
        (10, "Texas Tech University Students Handbook-chunk 5", "Students Handbook, Disciplinary Sanctions, Conduct Appeals, Student Conduct Records, Sexual Misconduct Policy, Title IX Procedures, University Sanctions", "https://www.depts.ttu.edu/dos/Studenthandbook2022forward/Student-Handbook-2023-2024.pdf"),
        (11, "Texas Tech University Students Handbook-chunk 6", "Students Handbook, Non-Title IX Sexual Misconduct, Interpersonal Violence, Sexual Harassment, Sexual Assault Reporting, Supportive Measures, University Sexual Misconduct Policy", "https://www.depts.ttu.edu/dos/Studenthandbook2022forward/Student-Handbook-2023-2024.pdf"),
        (12, "Texas Tech University Students Handbook-chunk 7", "Students Handbook, Amnesty Provisions, Sexual Misconduct Reporting, Incident Response, Formal Complaint Process, Title IX Coordinator, Supportive Measures", "https://www.depts.ttu.edu/dos/Studenthandbook2022forward/Student-Handbook-2023-2024.pdf"),
        (13, "Texas Tech University Students Handbook-chunk 8", "Students Handbook, Title IX Hearings, Non-Title IX Grievance Process, Sexual Misconduct Sanctions, Hearing Panel Procedures, Informal Resolution, Grievance Process", "https://www.depts.ttu.edu/dos/Studenthandbook2022forward/Student-Handbook-2023-2024.pdf"),
        (14, "Texas Tech University Students Handbook-chunk 9", "Students Handbook, Sexual Misconduct Hearings, Grievance Process, Administrative and Panel Hearings, Title IX Coordinator, Disciplinary Sanctions, Appeal Procedures", "https://www.depts.ttu.edu/dos/Studenthandbook2022forward/Student-Handbook-2023-2024.pdf"),
        (15, "Texas Tech University Students Handbook-chunk 10", "Students Handbook, Student Organization Conduct, Code of Student Conduct, Investigation Process, Interim Actions, Voluntary Resolution, University Sanctions", "https://www.depts.ttu.edu/dos/Studenthandbook2022forward/Student-Handbook-2023-2024.pdf"),
        (16, "Texas Tech University Students Handbook-chunk 11", "Students Handbook, Student Organization Hearings, Pre-Hearing Process, Investigation Report, Conduct Procedures, Sanction Only Hearing, Appeals Process", "https://www.depts.ttu.edu/dos/Studenthandbook2022forward/Student-Handbook-2023-2024.pdf"),
        (17, "Texas Tech University Students Handbook-chunk 12", "Students Handbook, Academic Integrity, Anti-Discrimination Policy, Alcohol Policy, Class Absences, Grievance Procedures, Student Conduct", "https://www.depts.ttu.edu/dos/Studenthandbook2022forward/Student-Handbook-2023-2024.pdf"),
        (18, "Texas Tech University Students Handbook-chunk 13", "Students Handbook, Disability Services, FERPA Guidelines, Disciplinary Actions, Employment Grievances, Academic Appeals, Student Support Resources", "https://www.depts.ttu.edu/dos/Studenthandbook2022forward/Student-Handbook-2023-2024.pdf"),
        (19, "Texas Tech University Students Handbook-chunk 14", "Students Handbook, Student Organization Registration, Solicitation and Advertising, Student Government Association, Military and Veteran Programs, Student Identification, Student Support Services", "https://www.depts.ttu.edu/dos/Studenthandbook2022forward/Student-Handbook-2023-2024.pdf"),
        (20, "Texas Tech University Students Handbook-chunk 15", "Students Handbook, Campus Grounds Use, Expressive Activities, Amplification Equipment, Voluntary Withdrawal, Involuntary Withdrawal, Student Safety", "https://www.depts.ttu.edu/dos/Studenthandbook2022forward/Student-Handbook-2023-2024.pdf"),
        (21, "Texas Tech University Students Handbook-chunk 16", "Students Handbook, Student Organization Training, Campus Grounds Use, Facility Reservations, Amplification Equipment, Expressive Activities, Student Records", "https://www.depts.ttu.edu/dos/Studenthandbook2022forward/Student-Handbook-2023-2024.pdf"),
        (22, "Texas Tech University Students Handbook-chunk 17", "Students Handbook, Student Conduct Definitions, University Policies, Behavioral Intervention, Sexual Misconduct Definitions, Disciplinary Actions, Student Records", "https://www.depts.ttu.edu/dos/Studenthandbook2022forward/Student-Handbook-2023-2024.pdf")
        # Add more initial data as needed
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

# NLP functions (keep your existing NLP functions here)
# ...

# Flask routes
@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/chat', methods=['POST'])
def chat():
    user_query = request.json['message']
    answer, intent_data = get_answer(user_query)
    return jsonify({'response': answer, 'intent_data': intent_data})

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

# NLP functions (you should include your existing NLP functions here)
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
        model="gpt-4",
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

if __name__ == '__main__':
    # Initialize database and load initial data
    conn = get_database_connection()
    init_db(conn)
    load_initial_data()
    app.run(debug=True)
