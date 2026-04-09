import glob
import os
import json

class Agent:
    def __init__(self, name):
        
        self.system_prompt = None
        self.user_prompt = None
        self.name = name
        self.personality = None
        self.vision = f"./images/user/{name}/"
        self.prompt_path = f"./prompts/user/{name}/"

    def get_system_prompt(self):
        return self.system_prompt
    
    def get_user_prompt(self):
        return self.user_prompt
    
    def get_personality(self):
        return self.personality
    
    def get_vision(self):
        return self.vision
    
    def get_prompt_path(self):
        return self.prompt_path
    
    def set_agent(self):
        # Read system prompt from file
        with open("./prompts/system/systemprompt.txt", "r", encoding="utf-8") as f:
            self.system_prompt = f.read()
        self.user_prompt = self.set_initial_prompt()
        self.personality = self.user_prompt['personality']
    
    def set_initial_prompt(self):
        with open(f"./prompts/user/{self.name}/{self.name}_gemini_response_1.json", "r", encoding="utf-8") as f:
            self.user_prompt = f.read()
            self.user_prompt = self.clean_prompt(self.user_prompt)
            self.user_prompt = json.loads(self.user_prompt)
            return self.user_prompt

    def set_user_prompt(self):
        # Get file data
        updated_file = glob.glob(os.path.join(self.prompt_path, f"updated/{self.name}_gemini_response_*.json"))
        previous_file = glob.glob(os.path.join(self.prompt_path, f"{self.name}_gemini_response_*.json"))

        # Combine both lists (updated and previous files)
        all_files = updated_file + previous_file

        if not all_files:
            # Fallback to initial json if no outputs yet
            with open(f"./prompts/user/{self.name}/{self.name}_gemini_response_1.json", "r", encoding="utf-8") as f:
                prompt_str = f.read()
                prompt_str = self.clean_prompt(prompt_str)
                prompt_dict = json.loads(prompt_str)
                return prompt_dict
        
        # Sort files by extracting the index from filenames
        all_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]), reverse=True)

        # Get the highest index value from the sorted list
        highest_index = int(all_files[0].split('_')[-1].split('.')[0])

        # Check if the highest index file exists in both updated and previous, if so, prefer the updated one
        updated_highest = next((f for f in updated_file if int(f.split('_')[-1].split('.')[0]) == highest_index), None)

        # If an updated file with the highest index exists, use it, otherwise use the highest from previous
        files = updated_highest if updated_highest else all_files[0]

        # Open and parse the selected file, with extra checks for empty or invalid content
        try:
            with open(files, "r", encoding="utf-8") as f:
                file_content = f.read().strip()  # Remove leading/trailing whitespace

                if not file_content:  # Check if the file is empty
                    raise ValueError(f"File {files} is empty, cannot parse it.")

                # Proceed to clean and load the JSON content
                self.user_prompt = self.clean_prompt(file_content)
                self.user_prompt = json.loads(self.user_prompt)  # Parse the cleaned content as JSON
                return self.user_prompt
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in file {files}: {e}")
        except Exception as e:
            raise ValueError(f"Error reading or parsing file {files}: {e}")
        
    def get_new_generated_prompt(self):
        # Get latest prompt data
        files = glob.glob(os.path.join(self.prompt_path, f"{self.name}_gemini_response_*.json"))
        if not files:
            # Fallback to initial json if no outputs yet
            with open(f"./prompts/user/{self.name}/{self.name}_gemini_response_1.json", "r", encoding="utf-8") as f:
                prompt_str = f.read()
                prompt_str = self.clean_prompt(prompt_str)
                prompt_dict = json.loads(prompt_str)
                return prompt_dict
            
        files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        with open(files[-1], "r", encoding="utf-8") as f:
            self.user_prompt = f.read()
            self.user_prompt = self.clean_prompt(self.user_prompt)
            self.user_prompt = json.loads(self.user_prompt)
            return self.user_prompt

    def clean_prompt(self, prompt):
        cleaned_text = prompt.strip()
        if cleaned_text.startswith("```json"):
            cleaned_text = cleaned_text[7:].strip()
        if cleaned_text.startswith("```"):
            cleaned_text = cleaned_text[3:].strip()
        if cleaned_text.endswith("```"):
            cleaned_text = cleaned_text[:-3].strip()
        return cleaned_text
    
    def update_prompt(self, jsonData):
        # Get file
        files = glob.glob(os.path.join(self.prompt_path, f"{self.name}_gemini_response_*.json"))
        if not files:
            raise Exception(detail="No agent data to update, check agent name or initial prompt data if it exists.")
    
        # Get latest file index
        files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
        latest_file = files[-1]
        latest_index = int(latest_file.split("_")[-1].split(".")[0])

        # Target file in updates folder
        updates_dir = os.path.join(self.get_prompt_path(), f"updated/")

        # Ensure the directory exists
        os.makedirs(updates_dir, exist_ok=True)

        # Make new file name for the updated prompt data
        new_filepath = os.path.join(updates_dir, f"{self.name}_gemini_response_{latest_index}.json")
        
        # Write new JSON prompt
        try:
            json_obj = json.loads(jsonData)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON string: {e}")

        with open(new_filepath, "w", encoding="utf-8") as f:
            json.dump(json_obj, f, indent=2, ensure_ascii=False)

        return new_filepath

    def delete_prev_prompt_files(self):
        base = self.get_prompt_path()
        keep = os.path.join(base, f"{self.name}_gemini_response_1.json")
        deleted = []

        # delete all main *.json except the _1.json
        for p in glob.glob(os.path.join(base, f"{self.name}_gemini_response_*.json")):
            if p != keep and os.path.isfile(p):
                try:
                    os.remove(p)
                    deleted.append(p)
                except Exception as e:
                    deleted.append(f"{p} (ERROR: {e})")

        # delete everything in updated/
        upd = os.path.join(base, "updated")
        if os.path.isdir(upd):
            for p in glob.glob(os.path.join(upd, "*")):
                if os.path.isfile(p):
                    try:
                        os.remove(p)
                        deleted.append(p)
                    except Exception as e:
                        deleted.append(f"{p} (ERROR: {e})")

        #return {"kept": keep, "deleted_count": len(deleted), "deleted": deleted}
        return f"Deleted count: {deleted}"
    
    def overwrite_initial_prompt_data(self, jsonData):
        # Get file
        files = glob.glob(os.path.join(self.prompt_path, f"{self.name}_gemini_response_1.json"))
        if not files:
            raise Exception(detail="No agent data to update, check agent name or initial prompt data if it exists.")
        
        # Write new JSON prompt
        try:
            json_obj = json.loads(jsonData)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON string: {e}")

        with open(files, "w", encoding="utf-8") as f:
            json.dump(json_obj, f, indent=2, ensure_ascii=False)

        return "Update Success: " + self.name

if __name__ == "__main__":
    agent = Agent("maria")
    agent.set_agent()
    print(agent.get_user_prompt())
    print(agent.get_system_prompt())
    print(agent.get_personality())
    print(agent.get_vision())
    print(agent.get_prompt_path())