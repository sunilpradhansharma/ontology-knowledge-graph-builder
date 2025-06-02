import gradio as gr
import rdflib
from rdflib import Graph, Namespace, Literal, URIRef
import networkx as nx
import plotly.graph_objects as go
import openai
import os
from dotenv import load_dotenv
import json
from datetime import datetime
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import numpy as np

# Load environment variables from .env
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in .env file.")

# Initialize OpenAI client with explicit API key
openai.api_key = OPENAI_API_KEY

# Initialize RDF graph
g = Graph()
g.parse("knowledge_graph.ttl", format="turtle")

# Initialize ChromaDB
chroma_client = chromadb.Client(Settings(
    persist_directory="chroma_db",
    anonymized_telemetry=False
))

# Initialize sentence transformer
model = SentenceTransformer('all-MiniLM-L6-v2')

# Create or get collection
try:
    collection = chroma_client.get_collection("knowledge_graph")
except:
    collection = chroma_client.create_collection("knowledge_graph")

# Define namespaces
RDF = Namespace("http://www.w3.org/1999/02/22-rdf-syntax-ns#")
RDFS = Namespace("http://www.w3.org/2000/01/rdf-schema#")
OWL = Namespace("http://www.w3.org/2002/07/owl#")
XSD = Namespace("http://www.w3.org/2001/XMLSchema#")
KG = Namespace("http://example.org/kg/")

def get_relevant_context(query):
    """Get relevant context from the knowledge graph for a query."""
    # Perform semantic search
    semantic_results = semantic_search(query)
    
    # Get relevant entities and their relationships
    context = []
    for metadata in semantic_results['metadatas'][0]:
        entity_uri = metadata['subject']
        entity_info = get_entity_info(entity_uri)
        related = get_related_entities(entity_uri)
        
        context.append({
            "entity": entity_uri,
            "info": entity_info,
            "related": related
        })
    
    return context

def process_natural_language_query(query):
    """Process a natural language query using LLM and knowledge graph."""
    # Get relevant context
    context = get_relevant_context(query)
    
    # Format context for LLM
    context_str = json.dumps(context, indent=2)
    
    # Create prompt for LLM
    prompt = f"""You are a helpful assistant that answers questions about a business knowledge graph.
    Use the following context from the knowledge graph to answer the question:
    
    Context:
    {context_str}
    
    Question: {query}
    
    Please provide a clear and comprehensive answer based on the knowledge graph data.
    If the answer cannot be fully determined from the available context, say so and explain what information is available.
    Focus on business implications and relationships between entities.
    """
    
    # Get response from LLM
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that explains business entity relationships and metadata in a clear, professional manner."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1000
    )
    
    return response["choices"][0]["message"]["content"]

def index_knowledge_graph():
    """Index the knowledge graph into ChromaDB."""
    documents = []
    metadatas = []
    ids = []
    
    for s, p, o in g.triples((None, None, None)):
        if isinstance(o, Literal):
            doc = f"Entity: {s}\nPredicate: {p}\nObject: {o}"
            documents.append(doc)
            metadatas.append({
                "subject": str(s),
                "predicate": str(p),
                "object": str(o),
                "type": "literal"
            })
            ids.append(f"{s}_{p}_{o}")
        elif isinstance(o, URIRef):
            doc = f"Entity: {s}\nPredicate: {p}\nObject: {o}"
            documents.append(doc)
            metadatas.append({
                "subject": str(s),
                "predicate": str(p),
                "object": str(o),
                "type": "uri"
            })
            ids.append(f"{s}_{p}_{o}")
    
    if documents:
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )

def semantic_search(query, n_results=5):
    """Perform semantic search using ChromaDB."""
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )
    return results

def get_entity_info(entity_uri):
    """Get information about an entity from the knowledge graph."""
    info = {}
    for s, p, o in g.triples((URIRef(entity_uri), None, None)):
        if isinstance(o, Literal):
            info[str(p)] = str(o)
        else:
            info[str(p)] = str(o)
    return info

def get_related_entities(entity_uri):
    """Get all entities related to a given entity."""
    related = []
    for s, p, o in g.triples((URIRef(entity_uri), None, None)):
        if isinstance(o, URIRef):
            related.append((str(p), str(o)))
    for s, p, o in g.triples((None, None, URIRef(entity_uri))):
        if isinstance(s, URIRef):
            related.append((str(p), str(s)))
    return related

def create_relationship_graph(entity_uri):
    """Create a NetworkX graph of relationships starting from an entity."""
    G = nx.Graph()
    G.add_node(entity_uri, label=entity_uri.split("/")[-1])
    
    def add_related_nodes(node_uri, depth=0, max_depth=2):
        if depth >= max_depth:
            return
        for s, p, o in g.triples((URIRef(node_uri), None, None)):
            if isinstance(o, URIRef):
                G.add_node(str(o), label=str(o).split("/")[-1])
                G.add_edge(node_uri, str(o), label=str(p).split("/")[-1])
                add_related_nodes(str(o), depth + 1, max_depth)
    
    add_related_nodes(entity_uri)
    return G

def plot_relationship_graph(G):
    """Create an interactive Plotly visualization of the relationship graph."""
    pos = nx.spring_layout(G)
    
    edge_trace = go.Scatter(
        x=[], y=[], line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines')
    
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace['x'] += (x0, x1, None)
        edge_trace['y'] += (y0, y1, None)
    
    node_trace = go.Scatter(
        x=[], y=[], text=[], mode='markers+text', hoverinfo='text',
        marker=dict(showscale=True, colorscale='YlGnBu', size=10))
    
    for node in G.nodes():
        x, y = pos[node]
        node_trace['x'] += (x,)
        node_trace['y'] += (y,)
        node_trace['text'] += (G.nodes[node]['label'],)
    
    fig = go.Figure(data=[edge_trace, node_trace],
                   layout=go.Layout(
                       showlegend=False,
                       hovermode='closest',
                       margin=dict(b=20,l=5,r=5,t=40),
                       xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                   )
    
    return fig

def get_llm_explanation(query, context):
    """Get an explanation from the LLM about relationships or metadata."""
    prompt = f"""Given the following context about a knowledge graph:
    {context}
    
    Please explain: {query}
    
    Provide a clear and concise explanation focusing on the business implications."""
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500
    )
    return response["choices"][0]["message"]["content"]

def search_entities(query):
    """Search for entities using both semantic and exact search."""
    # Semantic search
    semantic_results = semantic_search(query)
    semantic_entities = set()
    for metadata in semantic_results['metadatas'][0]:
        semantic_entities.add(metadata['subject'])
    
    # Exact search
    exact_results = []
    for s, p, o in g.triples((None, None, None)):
        if isinstance(o, Literal) and query.lower() in str(o).lower():
            exact_results.append(str(s))
        elif isinstance(s, URIRef) and query.lower() in str(s).lower():
            exact_results.append(str(s))
    
    # Combine results
    all_results = list(semantic_entities.union(set(exact_results)))
    return all_results

def generate_documentation(entity_uri=None, doc_type="technical"):
    """Generate documentation for entities or the entire knowledge graph."""
    if entity_uri:
        entity_info = get_entity_info(entity_uri)
        related = get_related_entities(entity_uri)
        context = f"Entity: {entity_uri}\nInfo: {json.dumps(entity_info, indent=2)}\nRelated: {json.dumps(related, indent=2)}"
    else:
        context = "Generate documentation for the entire knowledge graph"
    
    prompt = f"""Generate {doc_type} documentation for the following:
    {context}
    
    Include:
    1. Overview
    2. Entity/Relationship descriptions
    3. Business rules and constraints
    4. Usage guidelines
    5. Examples"""
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1000
    )
    return response["choices"][0]["message"]["content"]

def validate_data_quality(entity_uri=None, validation_type="completeness"):
    """Validate data quality for entities or the entire knowledge graph."""
    if entity_uri:
        entity_info = get_entity_info(entity_uri)
        related = get_related_entities(entity_uri)
        context = f"Entity: {entity_uri}\nInfo: {json.dumps(entity_info, indent=2)}\nRelated: {json.dumps(related, indent=2)}"
    else:
        context = "Validate data quality for the entire knowledge graph"
    
    prompt = f"""Perform {validation_type} validation on:
    {context}
    
    Check for:
    1. Data completeness
    2. Consistency
    3. Accuracy
    4. Timeliness
    5. Provide recommendations for improvement"""
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1000
    )
    return response["choices"][0]["message"]["content"]

def get_data_dictionary_info(query=None, entity_uri=None):
    """Get detailed information about data dictionary terms."""
    if entity_uri:
        entity_info = get_entity_info(entity_uri)
        related = get_related_entities(entity_uri)
        context = f"Entity: {entity_uri}\nInfo: {json.dumps(entity_info, indent=2)}\nRelated: {json.dumps(related, indent=2)}"
    elif query:
        # Use semantic search for better results
        semantic_results = semantic_search(query)
        context = f"Search query: {query}\nSemantic matches: {json.dumps(semantic_results['metadatas'][0], indent=2)}"
    else:
        context = "Provide overview of the data dictionary"
    
    prompt = f"""Provide detailed information about the data dictionary:
    {context}
    
    Include:
    1. Term definitions
    2. Business context
    3. Related terms
    4. Usage examples
    5. Data type and constraints"""
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1000
    )
    return response["choices"][0]["message"]["content"]

def validate_entity_hierarchy(entity_uri):
    """Validate entity hierarchy and relationships to ensure structural integrity."""
    entity_info = get_entity_info(entity_uri)
    related = get_related_entities(entity_uri)
    
    # Check for required attributes and relationships
    validation_results = {
        "is_valid": True,
        "warnings": [],
        "errors": [],
        "hierarchy": {},
        "constraints": []
    }
    
    # Analyze entity attributes
    for predicate, value in entity_info.items():
        if str(predicate).endswith("#type"):
            validation_results["hierarchy"]["type"] = value
        elif str(predicate).endswith("#required"):
            validation_results["constraints"].append({
                "attribute": str(predicate).split("#")[0],
                "constraint": "required",
                "value": value
            })
    
    # Analyze relationships
    for predicate, related_entity in related:
        # Check for circular references
        if related_entity == entity_uri:
            validation_results["errors"].append(f"Circular reference detected: {predicate}")
            validation_results["is_valid"] = False
        
        # Check relationship cardinality
        related_count = len([r for r in related if r[0] == predicate])
        if related_count > 1:
            validation_results["hierarchy"][predicate] = {
                "type": "one-to-many",
                "target": related_entity
            }
        else:
            validation_results["hierarchy"][predicate] = {
                "type": "one-to-one",
                "target": related_entity
            }
    
    return validation_results

def generate_code(entity_uri=None, code_type="model", language="python", framework=None):
    """Generate code based on knowledge graph entities and relationships."""
    if entity_uri:
        # Validate entity hierarchy first
        validation_results = validate_entity_hierarchy(entity_uri)
        if not validation_results["is_valid"]:
            error_message = "Entity validation failed:\n"
            error_message += "\n".join(validation_results["errors"])
            error_message += "\n\nWarnings:\n"
            error_message += "\n".join(validation_results["warnings"])
            return error_message
        
        entity_info = get_entity_info(entity_uri)
        related = get_related_entities(entity_uri)
        context = f"""Entity: {entity_uri}
Info: {json.dumps(entity_info, indent=2)}
Related: {json.dumps(related, indent=2)}
Hierarchy: {json.dumps(validation_results["hierarchy"], indent=2)}
Constraints: {json.dumps(validation_results["constraints"], indent=2)}"""
    else:
        context = "Generate code for the entire knowledge graph structure"
    
    # Determine framework-specific instructions
    framework_instructions = ""
    if framework:
        if framework.lower() == "django":
            framework_instructions = """Use Django model fields and relationships:
            - Use appropriate field types based on data types
            - Implement proper relationship fields (ForeignKey, ManyToManyField)
            - Add Meta class with constraints
            - Include model validation methods"""
        elif framework.lower() == "sqlalchemy":
            framework_instructions = """Use SQLAlchemy ORM models and relationships:
            - Define proper column types and constraints
            - Implement relationship() with correct backref
            - Add __table_args__ for constraints
            - Include validation methods"""
        elif framework.lower() == "pydantic":
            framework_instructions = """Use Pydantic models with proper type hints:
            - Define field types and validators
            - Implement relationship models
            - Add Config class with constraints
            - Include custom validators"""
    
    prompt = f"""Generate {language} code for the following knowledge graph structure:
    {context}
    
    Code Type: {code_type}
    Language: {language}
    Framework: {framework}
    
    Requirements:
    1. Generate clean, well-documented code
    2. Include proper type hints and docstrings
    3. Follow best practices for {language} and {framework if framework else 'the language'}
    4. Include necessary imports and dependencies
    5. {framework_instructions}
    6. Implement all validation rules and constraints
    7. Ensure proper handling of relationships and hierarchies
    8. Include data validation methods
    9. Add proper error handling for constraint violations
    
    The code should accurately represent the entities, relationships, and constraints from the knowledge graph.
    Make sure to maintain the hierarchical structure and implement all validation rules.
    """
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an expert software developer who generates high-quality, production-ready code with proper validation and error handling."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=2000
    )
    return response["choices"][0]["message"]["content"]

def generate_api_spec(entity_uri=None, api_type="rest", framework=None):
    """Generate API specification based on knowledge graph entities."""
    if entity_uri:
        entity_info = get_entity_info(entity_uri)
        related = get_related_entities(entity_uri)
        context = f"Entity: {entity_uri}\nInfo: {json.dumps(entity_info, indent=2)}\nRelated: {json.dumps(related, indent=2)}"
    else:
        context = "Generate API specification for the entire knowledge graph"
    
    prompt = f"""Generate {api_type.upper()} API specification for the following knowledge graph structure:
    {context}
    
    Requirements:
    1. Define endpoints for CRUD operations
    2. Include request/response schemas
    3. Specify authentication and authorization requirements
    4. Document query parameters and filters
    5. Include example requests and responses
    6. Follow RESTful or GraphQL best practices
    7. Consider rate limiting and pagination
    """
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an expert API designer who creates comprehensive and well-documented API specifications."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=2000
    )
    return response["choices"][0]["message"]["content"]

# Index the knowledge graph
index_knowledge_graph()

# Create Gradio interface
with gr.Blocks(title="Knowledge Graph Explorer") as demo:
    gr.Markdown("# Knowledge Graph Explorer")
    
    with gr.Tab("Natural Language Query"):
        with gr.Row():
            query_input = gr.Textbox(
                label="Ask any question about the knowledge graph",
                placeholder="Example: What are the relationships between customers and payments?",
                lines=3
            )
            query_button = gr.Button("Get Answer")
        
        with gr.Row():
            query_output = gr.Markdown(label="Answer")
        
        def process_query(query):
            if not query.strip():
                return "Please enter a question."
            return process_natural_language_query(query)
        
        query_button.click(
            process_query,
            inputs=query_input,
            outputs=query_output
        )
        
        query_input.submit(
            process_query,
            inputs=query_input,
            outputs=query_output
        )
    
    with gr.Tab("Search & Explore"):
        with gr.Row():
            search_input = gr.Textbox(label="Search Entities")
            search_button = gr.Button("Search")
        
        with gr.Row():
            results = gr.Dropdown(label="Found Entities", choices=[], interactive=True)
        
        with gr.Row():
            entity_info = gr.JSON(label="Entity Information")
        
        with gr.Row():
            graph = gr.Plot(label="Relationship Graph")
        
        with gr.Row():
            explanation = gr.Textbox(label="AI Explanation", lines=5)
        
        def search_and_display(query):
            entities = search_entities(query)
            return gr.Dropdown.update(choices=entities)
        
        def display_entity_info(entity):
            if not entity:
                return None, None, "Please select an entity"
            info = get_entity_info(entity)
            G = create_relationship_graph(entity)
            fig = plot_relationship_graph(G)
            context = f"Entity: {entity}\nInfo: {json.dumps(info, indent=2)}"
            exp = get_llm_explanation("Explain this entity and its relationships", context)
            return info, fig, exp
        
        search_button.click(search_and_display, inputs=search_input, outputs=results)
        results.change(display_entity_info, inputs=results, outputs=[entity_info, graph, explanation])
    
    with gr.Tab("Documentation Generator"):
        with gr.Row():
            doc_entity = gr.Textbox(label="Entity URI (optional)")
            doc_type = gr.Dropdown(label="Documentation Type", 
                                 choices=["technical", "business", "user"], 
                                 value="technical")
            doc_button = gr.Button("Generate Documentation")
        
        doc_output = gr.Markdown(label="Generated Documentation")
        
        doc_button.click(
            generate_documentation,
            inputs=[doc_entity, doc_type],
            outputs=doc_output
        )
    
    with gr.Tab("Data Quality Validator"):
        with gr.Row():
            val_entity = gr.Textbox(label="Entity URI (optional)")
            val_type = gr.Dropdown(label="Validation Type",
                                 choices=["completeness", "consistency", "accuracy", "timeliness"],
                                 value="completeness")
            val_button = gr.Button("Validate Data Quality")
        
        val_output = gr.Markdown(label="Validation Results")
        
        val_button.click(
            validate_data_quality,
            inputs=[val_entity, val_type],
            outputs=val_output
        )
    
    with gr.Tab("Data Dictionary Assistant"):
        with gr.Row():
            dict_query = gr.Textbox(label="Search Term (optional)")
            dict_entity = gr.Textbox(label="Entity URI (optional)")
            dict_button = gr.Button("Get Dictionary Info")
        
        dict_output = gr.Markdown(label="Dictionary Information")
        
        dict_button.click(
            get_data_dictionary_info,
            inputs=[dict_query, dict_entity],
            outputs=dict_output
        )
    
    with gr.Tab("Code Generator"):
        with gr.Row():
            code_entity = gr.Textbox(label="Entity URI (optional)")
            code_type = gr.Dropdown(label="Code Type",
                                  choices=["model", "api", "service", "repository"],
                                  value="model")
            code_language = gr.Dropdown(label="Programming Language",
                                      choices=["python", "typescript", "java", "csharp"],
                                      value="python")
            code_framework = gr.Dropdown(label="Framework (optional)",
                                       choices=["django", "sqlalchemy", "pydantic", "fastapi", "spring", "express"],
                                       value=None)
        
        with gr.Row():
            code_button = gr.Button("Generate Code")
        
        with gr.Row():
            validation_output = gr.Markdown(label="Validation Results")
            code_output = gr.Code(label="Generated Code", language="python")
        
        def generate_code_wrapper(entity, code_type, language, framework):
            if not entity:
                return "Please enter an entity URI.", ""
            
            # First validate the entity
            validation_results = validate_entity_hierarchy(entity)
            validation_message = "Entity Validation Results:\n\n"
            
            if validation_results["is_valid"]:
                validation_message += "✅ Entity structure is valid\n"
            else:
                validation_message += "❌ Entity structure has issues:\n"
                validation_message += "\n".join(f"- {error}" for error in validation_results["errors"])
            
            if validation_results["warnings"]:
                validation_message += "\n⚠️ Warnings:\n"
                validation_message += "\n".join(f"- {warning}" for warning in validation_results["warnings"])
            
            validation_message += "\n\nHierarchy Structure:\n"
            validation_message += json.dumps(validation_results["hierarchy"], indent=2)
            
            # Generate code if validation passed
            if validation_results["is_valid"]:
                code = generate_code(entity, code_type, language, framework)
                return validation_message, code
            else:
                return validation_message, "Code generation skipped due to validation errors."
        
        code_button.click(
            generate_code_wrapper,
            inputs=[code_entity, code_type, code_language, code_framework],
            outputs=[validation_output, code_output]
        )
    
    with gr.Tab("API Specification Generator"):
        with gr.Row():
            api_entity = gr.Textbox(label="Entity URI (optional)")
            api_type = gr.Dropdown(label="API Type",
                                 choices=["rest", "graphql"],
                                 value="rest")
            api_framework = gr.Dropdown(label="Framework (optional)",
                                      choices=["fastapi", "express", "spring", "django"],
                                      value=None)
        
        with gr.Row():
            api_button = gr.Button("Generate API Specification")
        
        api_output = gr.Code(label="API Specification", language="yaml")
        
        def generate_api_wrapper(entity, api_type, framework):
            return generate_api_spec(entity, api_type, framework)
        
        api_button.click(
            generate_api_wrapper,
            inputs=[api_entity, api_type, api_framework],
            outputs=api_output
        )

if __name__ == "__main__":
    demo.launch() 