import unittest
from rdflib import Graph

class TestRDFBuilder(unittest.TestCase):

    def setUp(self):
        self.graph = Graph()
        self.graph.parse("knowledge_graph.ttl", format="turtle")

    def test_graph_not_empty(self):
        self.assertGreater(len(self.graph), 0, "Graph should not be empty")

    def test_entity_and_attribute_presence(self):
        entity_found = any("entity/" in str(s) for s, _, _ in self.graph)
        attribute_found = any("attribute/" in str(o) for _, _, o in self.graph)
        self.assertTrue(entity_found, "At least one entity should exist")
        self.assertTrue(attribute_found, "At least one attribute should exist")

    def test_has_attribute_relationships(self):
        has_attr_predicates = [p for _, p, _ in self.graph if "hasAttribute" in str(p)]
        self.assertGreater(len(has_attr_predicates), 0, "Should contain 'hasAttribute' relationships")


    def test_data_quality_rule_nodes(self):
        rules = [s for s, p, _ in self.graph if "hasDataQualityRule" in str(p)]
        self.assertGreater(len(rules), 0, "Should link to data quality rules")

    def test_attribute_metadata(self):
        sample = next((s for s, p, _ in self.graph if "standardizedAttributeName" in str(p)), None)
        self.assertIsNotNone(sample, "standardizedAttributeName should be present")

if __name__ == '__main__':
    unittest.main()