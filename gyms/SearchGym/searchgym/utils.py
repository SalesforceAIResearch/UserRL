import http.client
import json
import time
import os
from pathlib import Path

def get_serper_api_key():
    """Get Serper API key from secret.json or environment variable."""
    # Try to load from secret.json in project root
    try:
        secret_path = Path(__file__).parent.parent / "secret.json"
        if secret_path.exists():
            with open(secret_path) as f:
                return json.load(f)["serper_key"]
    except (FileNotFoundError, KeyError, json.JSONDecodeError):
        pass
    
    # Try to load from current directory secret.json
    try:
        with open("secret.json") as f:
            return json.load(f)["serper_key"]
    except (FileNotFoundError, KeyError, json.JSONDecodeError):
        pass
    
    # Try environment variable
    api_key = os.getenv("SERPER_API_KEY")
    if api_key:
        return api_key
    
    raise ValueError("Serper API key not found. Please provide it in secret.json or SERPER_API_KEY environment variable.")

api_key = get_serper_api_key()

conn = http.client.HTTPSConnection("google.serper.dev")
headers = {
  'X-API-KEY': api_key,
  'Content-Type': 'application/json'
}

def search_serper(query, num=10):
    # Handle mock API key for testing
    if api_key == "mock_api_key_for_testing":
        return f"Mock search results for '{query}':\n1. Mock Result 1\n- Snippet: This is a mock search result for testing purposes.\n2. Mock Result 2\n- Snippet: Another mock result to demonstrate the search functionality."
    
    payload = json.dumps({
        "q": query,
        "num": 15,
    })

    try_time = 0
    while True:
        conn.request("POST", "/search", payload, headers)
        res = conn.getresponse()
        data = res.read()
        data = data.decode("utf-8")
        data = json.loads(data)
        if try_time > 5:
            return "Search Error: Timeout"
        if data.get("organic", []) != []:
            break
        try_time += 1
        time.sleep(5)
    
    try:
        output = ""
        index = 1
        answer_box = data.get("answerBox", "")
        if answer_box and 'title' in answer_box:
            if "answer" in answer_box:
                output += f"{str(index)}. {answer_box['title']}\n- Answer: {answer_box['answer']}\n"
                index += 1
            elif 'snippet' in answer_box:
                output += f"{str(index)}. {answer_box['title']}\n- Snippet: {answer_box['snippet']}\n"
                index += 1
            
        
        if index > num:
            return output.strip()
        
        for item in data.get("organic", []):
            if 'title' in item and 'snippet' in item:
                output += f"{str(index)}. {item['title']}\n- Snippet: {item['snippet']}\n"
                index += 1
            if index > num:
                return output.strip()
        
        return output.strip()
    
    except Exception as e:
        error = f"Search Error: {e}"
        print(error)
        return error

if __name__ == "__main__":
    query = "How's the weather in Beijing"
    print(search_serper(query, num=3))
