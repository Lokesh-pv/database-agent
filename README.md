🤖 Database Agent: Chat with Your Data
This project is a powerful Streamlit web application that acts as an AI agent for searching and interacting with various SQL databases using natural language. It leverages LangChain and Groq's large language models to enable users to ask questions about their data in plain English and receive structured, tabular results.

✨ Features
Natural Language Database Queries: Interact with your database using conversational language.

Multi-Database Support: Connects to PostgreSQL, MySQL, and SQLite databases.

Dual-Agent Architecture:

Query Agent: Converts natural language questions into SQL queries, executes them, and fetches raw results.

Table Agent: Processes the raw query results and intelligently formats them into clean, readable markdown tables.

Interactive Streamlit UI: A user-friendly chat interface for seamless interaction.

Dynamic Schema Display: View your connected database's schema directly within the application.

Data Export: Download generated tables as CSV or Excel files.

Groq LLM Integration: Utilizes the high-performance Llama3-70b-8192 model for both agents, ensuring fast and accurate responses.

Robust Error Handling: Includes mechanisms to catch and display database connection errors and parsing issues.

🚀 Technologies Used
Python

Streamlit: For building the interactive web application.

LangChain: Framework for orchestrating the LLM agents and database interactions.

Groq API: Provides the underlying large language models (Llama3-70b-8192) for natural language processing.

Pandas: For data manipulation and displaying tabular results.

SQLAlchemy: Python SQL toolkit and Object Relational Mapper (ORM) for database abstraction.

psycopg2-binary: PostgreSQL adapter for Python.

mysql-connector-python: MySQL adapter for Python.

sqlite3 (built-in): For SQLite database connectivity.

python-dotenv: For securely loading environment variables.

openpyxl: For enabling Excel file downloads.

⚙️ Setup and Installation
Follow these steps to get the project up and running on your local machine.

Prerequisites
Python 3.8+

Access to a database (PostgreSQL, MySQL, or a SQLite file) with your data.

Two Groq API Keys (one for each agent).

Steps
Clone the Repository (or download the files):

git clone https://github.com/YourUsername/Database_Agent.git # Updated repository name suggestion
cd Database_Agent

(If you're directly uploading, create a folder named Database_Agent and place the files inside it.)

Create a Virtual Environment (Recommended):

python -m venv venv

Activate the virtual environment:

Windows:

.\venv\Scripts\activate

macOS/Linux:

source venv/bin/activate

Install Dependencies:

pip install -r requirements.txt

Configure Environment Variables (Optional but Recommended):
Create a file named .env in the root of your project directory (Database_Agent/) and add your database connection details. Replace the placeholder values with your actual credentials. This file is ignored by Git for security.

# Example for PostgreSQL:
# DB_HOST=your_postgres_host
# DB_PORT=5432
# DB_USER=your_postgres_user
# DB_PASSWORD=your_postgres_password
# DB_NAME=your_postgres_database

# Example for MySQL:
# DB_HOST=your_mysql_host
# DB_PORT=3306
# DB_USER=your_mysql_user
# DB_PASSWORD=your_mysql_password
# DB_NAME=your_mysql_database

# Example for SQLite:
# SQLITE_PATH=path/to/your/database.db
# (e.g., SQLITE_PATH=student.db if in the same directory as app.py)

Important: Ensure your chosen database is running and accessible from where you are running the application.

Run the Streamlit Application:

streamlit run app.py

Access the Application:
Your web browser will automatically open to the Streamlit application (usually http://localhost:8501).

💡 How to Use
Enter Groq API Keys: On the Streamlit application interface, you will see two input fields for "Groq API Key for Query Agent" and "Groq API Key for Table Agent." Paste your respective Groq API keys into these fields.

Select Database Type: In the sidebar, choose the type of database you want to connect to (PostgreSQL, MySQL, or SQLite).

Provide Database Details:

For PostgreSQL or MySQL, enter the Host, Port, User, Password, and Database Name.

For SQLite, provide the file path to your .db file.
(Note: If you configured .env variables, these fields will be pre-filled.)

View Schema (Optional): Expand the "📘 View Database Schema" section to see the tables and columns detected in your connected database. This can help you formulate your queries.

Start Chatting: Use the chat input box at the bottom to ask questions about your database. The AI agent will process your query, fetch data, and present it in a readable table format.

Example Queries:
(Provide examples relevant to your database's schema and data.)

"Show me the total number of orders."

"List all customers from New York."

"What are the top 5 best-selling products?"

"Give me the average price of items in the 'Electronics' category."
