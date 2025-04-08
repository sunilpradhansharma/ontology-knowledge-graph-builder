import unittest
from utils.api_client import get_all_entities, get_entity_details, get_attribute_details

class TestAPIClient(unittest.TestCase):

    def test_get_all_entities(self):
        entities = get_all_entities()
        self.assertIsInstance(entities, list)
        self.assertGreater(len(entities), 0)
        self.assertIn("dataEntityId", entities[0])

    def test_get_entity_details(self):
        entities = get_all_entities()
        entity_id = entities[0]["dataEntityId"]
        details = get_entity_details(entity_id)
        self.assertIsInstance(details, dict)
        self.assertEqual(details["dataEntityId"], entity_id)
        self.assertIn("standardizedAttributes", details)

    def test_get_attribute_details(self):
        entities = get_all_entities()
        entity_id = entities[0]["dataEntityId"]
        details = get_entity_details(entity_id)
        attributes = details.get("standardizedAttributes", [])
        if attributes:
            attr_id = attributes[0]["standardizedAttributeId"]
            attr_details = get_attribute_details(attr_id)
            self.assertEqual(attr_details["standardizedAttributeId"], attr_id)
            self.assertIn("standardizedAttributeName", attr_details)
        else:
            self.skipTest("No attributes available to test.")

if __name__ == '__main__':
    unittest.main()