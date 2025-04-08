from dotenv import load_dotenv
import argparse
import os
from utils.api_client import get_all_entities, get_entity_details, get_attribute_details
from utils.cache_handler import load_cache, save_cache, compute_hash
from utils.rdf_builder import build_rdf_graph, save_graph_to_file

def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="RDF Knowledge Graph Loader")
    parser.add_argument("--output", type=str, help="Output Turtle file")
    parser.add_argument("--endpoint", type=str, help="Neptune SPARQL endpoint (optional)")
    args = parser.parse_args()

    output_file = args.output or os.getenv("OUTPUT_FILE", "knowledge_graph.ttl")
    endpoint = args.endpoint or os.getenv("NEPTUNE_ENDPOINT")

    print("Fetching all entities...")
    entities = get_all_entities()

    print("Loading cache...")
    old_cache = load_cache()

    print("Building RDF graph with versioning support...")
    rdf_graph, new_cache = build_rdf_graph(
        entities,
        old_cache,
        get_entity_details,
        get_attribute_details,
        compute_hash
    )

    if len(rdf_graph) == 0:
        print("No changes detected. RDF graph is up to date.")
        return

    print(f"Saving RDF graph to file: {output_file}")
    save_graph_to_file(rdf_graph, filename=output_file)

    print("Saving updated cache...")
    save_cache(new_cache)

if __name__ == "__main__":
    main()