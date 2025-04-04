from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from workflow import app as workflow_app 

app_api = FastAPI()

class QueryRequest(BaseModel):
    prompt: str

@app_api.get("/")
def hello():
    return {"Hello, World!"}

@app_api.post("/invoke")
async def invoke_workflow(request: QueryRequest):
    """
    Invoke the LangGraph workflow with the given prompt.
    Expects JSON body with 'prompt' field.
    Returns the workflow result as JSON.
    """
    try:
        state = {"prompt": request.prompt, "web_search_count": 0}
        result = await workflow_app.ainvoke(state)
        return {"answer": result["response"]}
    except Exception as e:
        print("Exception:", e)
        raise HTTPException(status_code=500, detail="An error occurred")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app_api, host="0.0.0.0", port=8000)
