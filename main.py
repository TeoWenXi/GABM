from typing import Union
from scripts import gemini
import json
from scripts import agent
from fastapi import FastAPI, Body

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/user/prompt/{name}")
def get_user_prompt(name: str):
    try:
        new_agent = agent.Agent(name)
        new_agent.set_agent()
        return {"User Prompt": new_agent.get_user_prompt()}
    except Exception as e:
        return {"error": str(e)}
    
@app.get("/user/generate/{name}")
def generate_prompt(name: str):
    try:
        new_agent = agent.Agent(name)
        new_agent.set_agent()
        new_agent.set_user_prompt()
        new_gemini = gemini.Gemini(new_agent)
        response = new_gemini.get_response()
        updated_response = new_agent.get_new_generated_prompt()
        return {"response": updated_response}
    except Exception as e:
        return {"error": str(e)}

@app.post("/user/update/{name}")
def update_prompt_data(name: str, jsonData: str = Body(...)):
    try:
        curr_agent = agent.Agent(name)
        curr_agent.set_agent()
        new_path = curr_agent.update_prompt(jsonData)
        return new_path
    except Exception as e:
        return {"error": str(e)}
    
@app.delete("/user/clean_prev_prompts/{name}")
def delete_prev_prompt_data(name: str, data: str = Body(...)):
    try:
        curr_agent = agent.Agent(name)
        curr_agent.set_agent()
        output_info = curr_agent.delete_prev_prompt_files()
        return output_info
    except Exception as e:
        return {"error": str(e)}
    
@app.post("/user/overwrite/{name}")
def overwrite_prompt_data(name: str, jsonData: str = Body(...)):
    try:
        curr_agent = agent.Agent(name)
        curr_agent.set_agent()
        new_path = curr_agent.overwrite_initial_prompt_data(jsonData)
        return new_path
    except Exception as e:
        return {"error": str(e)}