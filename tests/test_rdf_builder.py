import unittest
from rdflib import Graph
from utils.api_client import get_all_entities, get_entity_details, get_attribute_details
from utils.rdf_builder import build_rdf_graph
from utils.cache_handler import compute_hash

class TestRDFBuilder(unittest.TestCase):

    def test_rdf_graph_generation(self):
        entities = get_all_entities()
        old_cache = {}
        g, new_cache = build_rdf_graph(entities, old_cache, get_entity_details, get_attribute_details, compute_hash)
        self.assertIsInstance(g, Graph)
        self.assertGreater(len(g), 0)
        for entity in entities:
            self.assertIn(entity["dataEntityId"], new_cache)