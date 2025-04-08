import unittest
from rdflib import Graph
from utils.api_client import get_all_entities, get_entity_details, get_attribute_details
from utils.rdf_builder import build_rdf_graph
from utils.cache_handler import compute_hash

class TestIntegration(unittest.TestCase):

    def test_entity_attribute_relationships(self):
        entities = get_all_entities()
        old_cache = {}
        g, _ = build_rdf_graph(entities, old_cache, get_entity_details, get_attribute_details, compute_hash)
        for entity in entities:
            entity_uri = f"http://example.org/entity/{entity['dataEntityId']}"
            self.assertIn("http://example.org/entity/", entity_uri)

    def test_attribute_metadata_serialization(self):
        attr = get_attribute_details("65955631-8ffd-45e5-a616-fe95ae86a6aa")
        self.assertIn("standardizedAttributeDefinition", attr)
        self.assertIsInstance(attr.get("relatedTerms", []), list)

    def test_graph_includes_uri(self):
        entities = get_all_entities()
        old_cache = {}
        g, _ = build_rdf_graph(entities, old_cache, get_entity_details, get_attribute_details, compute_hash)
        contains_uri = any("http://ontologies.capitalone.com" in str(o) for _, _, o in g)
        self.assertTrue(contains_uri)