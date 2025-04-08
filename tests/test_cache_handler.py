import unittest
import os
from utils.cache_handler import load_cache, save_cache, compute_hash

class TestCacheHandler(unittest.TestCase):

    def setUp(self):
        self.test_cache_file = "entity_hashes.json"
        self.sample_data = {"entity-1": "hashval123"}

    def test_compute_hash_consistency(self):
        data = {"name": "Sample Entity", "value": 42}
        hash1 = compute_hash(data)
        hash2 = compute_hash(data)
        self.assertEqual(hash1, hash2, "Hash should be consistent for same input")

    def test_save_and_load_cache(self):
        # Save to cache
        save_cache(self.sample_data)
        self.assertTrue(os.path.exists(self.test_cache_file))

        # Load from cache
        loaded_cache = load_cache()
        self.assertEqual(self.sample_data, loaded_cache)

    def tearDown(self):
        if os.path.exists(self.test_cache_file):
            os.remove(self.test_cache_file)

if __name__ == '__main__':
    unittest.main()