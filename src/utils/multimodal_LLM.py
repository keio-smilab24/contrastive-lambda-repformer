import openai
import os
import json
from openai import OpenAI
import time

client = OpenAI()

def get_gpt3_embeddings(text, model="text-embedding-3-large"):
    time.sleep(1)
    if openai.api_key is None:
        openai.api_key = os.environ["OPENAI_API_KEY"]
    text = text.replace("\n", " ")
    if text == "":
        text = "empty"
    return client.embeddings.create(input = [text], model=model).data[0].embedding


def get_scene_narrative_instruct_blip(key, json_path):
    with open(json_path) as f:
        inst_blip = json.load(f)
    response = inst_blip[key]
    return response

def remove_non_ascii_characters(string):
    return string.encode('ascii', 'ignore').decode('ascii')
