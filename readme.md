# LLM Web Search POC

## Execution Steps

### 1. Navigate to Project Directory
```bash
cd project
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Start Ollama Server
```bash
ollama serve
ollama pull llama3.1
```

### 4. Run the Application
```bash
streamlit run app.py
```
This launches the Streamlit app at [http://localhost:8501](http://localhost:8501).

---

## File Descriptions
- **`app.py`**: Entry point that runs the Streamlit UI.
- **`backend.py`**: Handles all backend logic, including LLM calls, vector DB, and web crawling.
- **`ui.py`**: Defines the Streamlit frontend for user interaction and query display.

