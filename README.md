# 🧠 RDF Knowledge Graph Loader for Enterprise Data Dictionaries

This project transforms enterprise data dictionary metadata into RDF format and loads it into a graph database (GraphDB). It simulates real-world API responses using mock JSON files and builds a rich semantic knowledge graph including nested relationships such as related entities and data quality rules.

---

## 🚀 Features

- ✅ Simulated API endpoints using local `.json` files
- 🧱 RDF graph generation using `rdflib`
- 🔁 Incremental updates using entity version hashing
- 🧩 Full RDF sub-graphs for:
  - `relatedEntities`
  - `dataQualityRules`
  - `sampleValues`, `relatedTerms`
- 🐢 Output in Turtle (`.ttl`) format for compatibility with GraphDB or AWS Neptune
- 🔧 CLI and `.env` support for flexibility
- 🧪 Modular architecture with unit & integration tests

---

## 📁 Project Structure

```
rdf_kg_loader_project/
├── main.py
├── requirements.txt
├── .env.example
├── README.md
├── mock/                  # Simulated API responses
│   ├── API-1.JSON
│   ├── API-2.JSON
│   └── API-3.JSON
├── utils/                 # Modular code components
│   ├── api_client.py
│   ├── cache_handler.py
│   ├── rdf_builder.py
│   └── neptune_uploader.py
└── tests/                 # Unit & integration tests
    ├── test_rdf_builder.py
    ├── test_api_cache.py
    └── test_additional.py
```

---

## 🧰 Setup

### 1. Clone the Repo
```bash
git https://github.com/sunilpradhansharma/ontology-knowledge-graph-builder.git
cd ontology-knowledge-graph-builder
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Set Environment Variables
```bash
cp .env.example .env
```
Edit `.env` as needed.

---

## ▶️ Usage

### Run the loader using defaults from `.env`:
```bash
python main.py
```

### Or override via CLI:
```bash
python main.py --output custom_graph.ttl --endpoint https://your-neptune-endpoint
```

---

## 🧪 Run Tests

```bash
python -m unittest discover -s tests
```

---

## 🔍 Sample RDF Output

Each entity and attribute is converted into RDF triples:

```turtle
<http://example.org/entity/abc123> ex:commonLabel "Customer ID" ;
                                   ex:hasAttribute <http://example.org/attribute/xyz456> .

<http://example.org/attribute/xyz456> ex:dataType "STRING" ;
                                      ex:relatedEntity <http://example.org/entity/related-entity> ;
                                      ex:hasDataQualityRule <http://example.org/dataQualityRule/xyz456/0> .
```
---

## 📊 RDF Graph Structure

Below is a visual representation of the RDF graph structure:

![RDF Graph Structure](docs/rdf_graph_structure.png)

---
## 🧭 Step-by-Step: Load RDF in GraphDB

### ✅ 1. Download and Start GraphDB
- Visit [https://www.ontotext.com/products/graphdb/](https://www.ontotext.com/products/graphdb/)
- Download the **Free Edition**
- Unzip and run `graphdb` or `graphdb.bat`
- Open your browser at [http://localhost:7200](http://localhost:7200)

### ✅ 2. Create a New Repository
- Go to **Repositories → Create**
- Choose:
  - **Repository ID**: `rdf_loader`
  - **Type**: `GraphDB Free`
  - **Ruleset**: `RDFS+`
- Click **Create**

### ✅ 3. Import RDF Turtle File
- Go to **Import → Server Files**
- Upload your `.ttl` file generated from this project
- Select the file and click **Import**

### ✅ 4. Explore the Graph
- Use the **SPARQL tab** to run queries such as:
```sparql
SELECT ?s ?p ?o WHERE {
  ?s ?p ?o
}
LIMIT 100
```

### ✅ 5. Visualize
- Go to **Explore → Visual Graph**
- Enter an entity URI like:
```
http://example.org/entity/{your-entity-id}
```
- Click **Explore** to visually inspect your graph!
- 
---

## 🧠 Use Cases

- Enterprise data catalog and metadata management  
- Data lineage and semantic discovery  
- ML feature store dictionary  
- Data governance and compliance  
- Knowledge graph applications

---

## 🧑‍💻 Author

**Sunil Pradhan Sharma**  
Senior Solution Architect – Data & AI  
[LinkedIn](https://linkedin.com/in/sunilsharma)

---