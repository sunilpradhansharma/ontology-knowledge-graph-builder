import json
import os

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