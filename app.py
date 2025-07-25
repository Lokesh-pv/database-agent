import streamlit as st
import pandas as pd
import re
from io import StringIO, BytesIO
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
from langchain.agents import create_sql_agent
from langchain.sql_database import SQLDatabase
from langchain.agents.agent_types import AgentType
from langchain.callbacks import StreamlitCallbackHandler
from sqlalchemy import create_engine, inspect
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
import sqlite3 # Import for SQLite connection

load_dotenv('.env')
st.set_page_config(page_title="Database Agent", page_icon="🧠") # Updated page_title

# Layout
st.title("🔗 Database Agent: Chat with Your Data") # Updated title
col1, col2 = st.columns(2)
api_key_query = col1.text_input("🔑 Groq API Key for Query Agent", type="password")
api_key_table = col2.text_input("🔑 Groq API Key for Table Agent", type="password")

# Guard clause for API keys
if not (api_key_query and api_key_table):
    st.info("Please enter both API keys to continue.")
    st.stop()

# LLMs
llm_query = ChatGroq(groq_api_key=api_key_query, model_name="llama3-70b-8192", temperature=0.1, streaming=True)
llm_table = ChatGroq(groq_api_key=api_key_table, model_name="llama3-70b-8192", temperature=0.1)

# Database Configuration in Sidebar
st.sidebar.header("Database Configuration")
db_type = st.sidebar.radio(
    "Choose Database Type:",
    ("PostgreSQL", "MySQL", "SQLite (Local File)")
)

db_host = os.getenv('DB_HOST')
db_port = os.getenv('DB_PORT')
db_user = os.getenv('DB_USER')
db_password = os.getenv('DB_PASSWORD')
db_name = os.getenv('DB_NAME')
sqlite_path = os.getenv('SQLITE_PATH')

# Input fields based on DB type
if db_type == "PostgreSQL":
    db_host = st.sidebar.text_input("PostgreSQL Host", value=db_host if db_host else "")
    db_port = st.sidebar.text_input("PostgreSQL Port", value=db_port if db_port else "5432")
    db_user = st.sidebar.text_input("PostgreSQL User", value=db_user if db_user else "")
    db_password = st.sidebar.text_input("PostgreSQL Password", type="password", value=db_password if db_password else "")
    db_name = st.sidebar.text_input("PostgreSQL Database Name", value=db_name if db_name else "")
    if not all([db_host, db_user, db_password, db_name]):
        st.info("Please provide all PostgreSQL connection details.")
        st.stop()
    connection_string = f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
elif db_type == "MySQL":
    db_host = st.sidebar.text_input("MySQL Host", value=db_host if db_host else "")
    db_port = st.sidebar.text_input("MySQL Port", value=db_port if db_port else "3306")
    db_user = st.sidebar.text_input("MySQL User", value=db_user if db_user else "")
    db_password = st.sidebar.text_input("MySQL Password", type="password", value=db_password if db_password else "")
    db_name = st.sidebar.text_input("MySQL Database Name", value=db_name if db_name else "")
    if not all([db_host, db_user, db_password, db_name]):
        st.info("Please provide all MySQL connection details.")
        st.stop()
    connection_string = f"mysql+mysqlconnector://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
elif db_type == "SQLite (Local File)":
    sqlite_path = st.sidebar.text_input("SQLite Database File Path", value=sqlite_path if sqlite_path else "student.db")
    if not sqlite_path:
        st.info("Please provide the path to your SQLite database file.")
        st.stop()
    # For SQLite, we need to ensure the file exists or can be created.
    # LangChain's SQLDatabase expects a file path.
    # For read-only mode, you can use: f"sqlite:///{sqlite_path}?mode=ro"
    # For read-write, just: f"sqlite:///{sqlite_path}"
    connection_string = f"sqlite:///{sqlite_path}" # Using f"sqlite:///{sqlite_path}" for simplicity

# Connect to DB
try:
    if db_type == "SQLite (Local File)":
        # For SQLite, create_engine needs a slightly different handling if the file path is relative
        # and we want to ensure it's absolute for robustness.
        # However, SQLAlchemy's create_engine for SQLite can handle relative paths directly.
        engine = create_engine(connection_string)
    else:
        engine = create_engine(connection_string)
    db = SQLDatabase(engine)
    st.sidebar.success(f"Successfully connected to {db_type}!")
except Exception as e:
    st.sidebar.error(f"Database connection failed: {e}")
    st.stop()

# Functions
@st.cache_resource(ttl="2h")
def get_schema_cached(engine_obj):
    inspector = inspect(engine_obj)
    schema_lines = []
    try:
        table_names = inspector.get_table_names()
        if not table_names:
            st.warning("No tables found in the database. Please ensure your database has tables.")
            return "No tables found."
        for table in table_names:
            cols = [f'"{c["name"]}" ({c["type"]})' for c in inspector.get_columns(table)]
            schema_lines.append(f"Table: {table}\nColumns: {', '.join(cols)}")
        return "\n\n".join(schema_lines)
    except Exception as e:
        st.error(f"Failed to retrieve schema: {e}. Please check database permissions or connection.")
        return "Error retrieving schema."

# Call the cached function
db_schema = get_schema_cached(engine)

def parse_text_to_table(text):
    try:
        # Try parsing as markdown table first
        if "|" in text and "---" in text:
            lines = [line.strip() for line in text.split("\n") if '|' in line]
            if lines:
                # Remove separator line
                lines = [line for line in lines if not line.strip().startswith('|-')]
                if lines:
                    header_line = lines[0]
                    data_lines = lines[1:]
                    
                    # Clean header and data lines
                    header = [h.strip() for h in header_line.split('|') if h.strip()]
                    data = []
                    for d_line in data_lines:
                        row = [item.strip() for item in d_line.split('|') if item.strip()]
                        if len(row) == len(header): # Ensure row matches header length
                            data.append(row)
                    
                    if header and data:
                        df = pd.DataFrame(data, columns=header)
                        return df
        
        # Fallback to CSV parsing if markdown table parsing fails or isn't applicable
        df = pd.read_csv(StringIO(text))
        return df if not df.empty else None
    except Exception as e:
        st.warning(f"⚠️ Parsing failed: {e}")
        return None

# Agent for DB queries
custom_prefix = f"""You are a SQL expert. Follow these rules strictly:
1. Only use tables: {', '.join(inspect(engine).get_table_names()) if db_schema != "No tables found." else "No tables available, cannot query."}
2. Follow this schema:
{db_schema}
3. Never make up table names or columns.
"""
toolkit = SQLDatabaseToolkit(db=db, llm=llm_query)

agent = create_sql_agent(
    llm=llm_query,
    toolkit=toolkit,
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True,
    max_iterations=8,
    early_stopping_method="generate",
    agent_kwargs={
        'prefix': custom_prefix,
        'max_execution_time': 15,
        'handle_parse_errors': "Check schema and try again"
    }
)

# Chat History
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Ask a question about the database!"}]

# Show schema info
with st.expander("📘 View Database Schema"):
    st.code(db_schema)

# Display chat messages
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# User input
if prompt := st.chat_input("Ask something about the database"):
    st.chat_message("user").write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandler(st.container())
        try:
            raw_result = agent.run(prompt, callbacks=[st_callback])
            answer = raw_result.split("Final Answer:")[-1].strip() if "Final Answer:" in raw_result else raw_result
            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.write("🧠 **Query Result:**")
            st.write(answer)

            # Automatically send to 2nd agent (Table Converter)
            with st.spinner("🧮 Converting to table..."):
                table_prompt = f"Convert the following text into a markdown table format only:\n{answer}"
                output = llm_table.invoke(table_prompt)
                text_table = output.content if hasattr(output, 'content') else output
                df = parse_text_to_table(text_table)

                if df is not None and not df.empty:
                    st.success("📊 Table Generated from Answer")
                    st.dataframe(df)

                    # Downloads
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button("📥 Download CSV", csv, "converted_table.csv", "text/csv")
                    excel_buffer = BytesIO()
                    df.to_excel(excel_buffer, index=False, engine='openpyxl')
                    st.download_button("📥 Download Excel", excel_buffer.getvalue(), "converted_table.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                else:
                    st.error("❌ Could not convert response to table format.")

        except Exception as e:
            err_msg = str(e)
            st.session_state.messages.append({"role": "assistant", "content": err_msg})
            st.error(f"🔥 Error: {err_msg}")
