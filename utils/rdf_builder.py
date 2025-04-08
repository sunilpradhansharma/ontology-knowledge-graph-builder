from rdflib import Graph, URIRef, Literal, Namespace

EX = Namespace("http://example.org/")
ONT = Namespace("http://ontologies.org.com/")

def build_rdf_graph(entities, old_cache, get_entity_details, get_attribute_details, compute_hash):
    g = Graph()
    g.bind("ex", EX)
    g.bind("ont", ONT)
    new_cache = {}

    for entity in entities:
        entity_id = entity.get("dataEntityId")
        entity_uri = URIRef(EX[f"entity/{entity_id}"])

        details = get_entity_details(entity_id)
        full_data = {**entity, **details}
        entity_hash = compute_hash(full_data)
        new_cache[entity_id] = entity_hash

        if old_cache.get(entity_id) == entity_hash:
            print(f"Skipping unchanged entity {entity_id}")
            continue

        for key, val in entity.items():
            if key != "dataEntityId":
                g.add((entity_uri, EX[key], Literal(val)))

        for key, val in details.items():
            if key == "standardizedAttributes":
                for attr in val:
                    attr_id = attr['standardizedAttributeId']
                    attr_uri = URIRef(EX[f"attribute/{attr_id}"])
                    g.add((entity_uri, EX["hasAttribute"], attr_uri))

                    attr_details = get_attribute_details(attr_id)
                    for attr_key, attr_val in attr_details.items():
                        if attr_key == "relatedEntities":
                            for rel in attr_val:
                                rel_uri = URIRef(EX[f"entity/{rel['dataEntityId']}"])
                                g.add((attr_uri, EX["relatedEntity"], rel_uri))
                                g.add((rel_uri, EX["commonLabel"], Literal(rel.get("commonLabel", ""))))
                        elif attr_key == "standards":
                            for idx, rule in enumerate(attr_val.get("dataQualityRules", [])):
                                rule_node = URIRef(EX[f"dataQualityRule/{attr_id}/{idx}"])
                                g.add((attr_uri, EX["hasDataQualityRule"], rule_node))
                                regex = rule.get("regularExpressionRuleDetails", {}).get("regularExpression")
                                if regex:
                                    g.add((rule_node, EX["ruleType"], Literal("REGULAR_EXPRESSION")))
                                    g.add((rule_node, EX["pattern"], Literal(regex)))
                        elif isinstance(attr_val, list):
                            for i in attr_val:
                                g.add((attr_uri, EX[attr_key], Literal(i)))
                        elif isinstance(attr_val, dict):
                            for sub_key, sub_val in attr_val.items():
                                g.add((attr_uri, EX[f"{attr_key}_{sub_key}"], Literal(sub_val)))
                        else:
                            g.add((attr_uri, EX[attr_key], Literal(attr_val)))
            elif key != "dataEntityId":
                g.add((entity_uri, EX[key], Literal(val)))

    return g, new_cache

def save_graph_to_file(graph, filename="knowledge_graph.ttl"):
    graph.serialize(destination=filename, format="turtle")
    print(f"RDF graph saved to {filename}")