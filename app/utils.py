import os

def delete_temp_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)

def load_openai_api_key(config_path):
    with open(config_path, "r") as f:
        for line in f:
            if line.startswith("OPENAI_API_KEY="):
                return line.strip().split("=")[1]