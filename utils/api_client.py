import json
import os

import requests

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
MOCK_PATH = os.path.join(BASE_PATH, "..", "mock")

def get_all_entities():
    with open(os.path.join(MOCK_PATH, "API-1.JSON"), "r") as f:
        data = json.load(f)
    return data.get("dataEntities", [])

def get_entity_details(entity_id):
    with open(os.path.join(MOCK_PATH, "API-2.JSON"), "r") as f:
        data = json.load(f)
    for entity in data.get("dataEntities", []):
        if entity["dataEntityId"] == entity_id:
            return entity
    return {}

def get_attribute_details(attribute_id):
    with open(os.path.join(MOCK_PATH, "API-3.JSON"), "r") as f:
        data = json.load(f)
    for attr in data.get("results", []):
        if attr["standardizedAttributeId"] == attribute_id:
            return attr
    return {}

def generate_response_all_entities():
    # Base API endpoint (replace with your actual URL)
    API_URL = "" # "https://api.example.com/entities"

    entity_ids = ['123', '456', '789']

    # Combined JSON list
    combined_response = []

    # Loop through each entity ID
    for entity_id in entity_ids:
        try:
            response = requests.get(API_URL, params={'entity_id': entity_id})
            response.raise_for_status()  # Raise an error for bad responses
            data = response.json()
            combined_response.append(data)
        except requests.exceptions.RequestException as e:
            print(f"Failed to fetch data for entity_id={entity_id}: {e}")

    # Save or print the combined response
    print(json.dumps(combined_response, indent=2))

    # Optionally, write to a file
    with open('combined_entities.json', 'w') as f:
        json.dump(combined_response, f, indent=2)