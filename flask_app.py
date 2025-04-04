from flask import Flask, request, jsonify
from workflow import app as workflow_app
import asyncio

app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello, World!"

@app.route('/invoke', methods=['POST'])
async def invoke_workflow():
    """
    Invoke the LangGraph workflow with the given prompt.
    Expects JSON body with 'prompt' field.
    Returns the workflow result as JSON.
    """
    try:
        data = request.get_json()
        prompt = data.get('prompt', '')
        # Initialize state as a dictionary
        state = {"prompt": prompt, "web_search_count": 0}
        # Run the workflow (replace with your actual workflow call)
        result = await workflow_app.ainvoke(state)
        return jsonify({"answer": result["response"]})
    except Exception as e:
        print("ex : " , e)
        return jsonify({"error": "An error occurred"}), 500

if __name__ == '__main__':
    app.run(debug=True)
