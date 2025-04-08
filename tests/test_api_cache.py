import unittest
from utils.api_client import get_all_entities, get_entity_details, get_attribute_details
from utils.cache_handler import compute_hash, load_cache, save_cache
import os

class TestAPIClient(unittest.TestCase):
    def test_get_all_entities(self):
        entities = get_all_entities()
        self.assertIsInstance(entities, list)
        self.assertGreater(len(entities), 0)

    def test_get_entity_details(self):
        details = get_entity_details("d1366f41-ebf0-4d9a-a872-2d9855ab1f1f")
        self.assertIn("commonLabel", details)

    def test_get_attribute_details(self):
        attr = get_attribute_details("65955631-8ffd-45e5-a616-fe95ae86a6aa")
        self.assertIn("standardizedAttributeName", attr)

class TestCacheHandler(unittest.TestCase):
    def test_hash_consistency(self):
        data = {"name": "Test", "value": 42}
        self.assertEqual(compute_hash(data), compute_hash(data))

    def test_cache_save_load(self):
        data = {"entity1": "hashval"}
        save_cache(data)
        loaded = load_cache()
        self.assertEqual(data, loaded)
        os.remove("entity_hashes.json")