import os

from dotenv import load_dotenv

load_dotenv()
COMET_API_KEY = os.getenv("COMET_API_KEY")
