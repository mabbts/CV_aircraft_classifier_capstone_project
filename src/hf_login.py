#put the following code in your local .env (ensure .gitignore includes the line .env)
#HF_TOKEN=enter your token text here
#
#usually only need to run once per environment


from dotenv import load_dotenv
import os
from huggingface_hub import login

def get_hf_token():
    #load environmental variables from .env
    load_dotenv()

    #Get the hugging face token
    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN not set")
    return token

def login_to_huggingface():
    #load environmental variables from .env
    load_dotenv()

    #Get the hugging face token
    hf_token = os.getenv("HF_TOKEN")

    #login
    login(token=hf_token)

login_to_huggingface()