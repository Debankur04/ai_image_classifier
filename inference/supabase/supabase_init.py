import os
from dotenv import load_dotenv, find_dotenv # 1. Import find_dotenv
from supabase import create_client, Client

# 2. Automatically find and load the .env file from any parent folder
load_dotenv(find_dotenv()) 

url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")

# Debugging: Print to verify variables are loaded
print(f"URL: {url}")
print(f"Key: {'Loaded' if key else 'None'}")

# 3. Initialize the client
supabase: Client = create_client(url, key)



print('supabase setup done')