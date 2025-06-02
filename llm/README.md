# Business Entity Explorer & Metadata Search

A powerful application that combines a Business Entity Relationship Explorer with a Metadata Search and Discovery Tool. This application leverages RDF knowledge graphs and Large Language Models (LLM) to provide an intuitive interface for exploring business entities, their relationships, and metadata.

## 🌟 Features

### Search and Discovery
- Natural language search for business entities
- Fuzzy matching for entity names and descriptions
- Real-time search results
- Detailed entity information display

### Relationship Visualization
- Interactive graph visualization of entity relationships
- Dynamic relationship mapping
- Zoom and pan capabilities
- Hover information for nodes and edges

### AI-Powered Insights
- Context-aware explanations of entity relationships
- Business context understanding
- Natural language descriptions of metadata
- Intelligent relationship analysis

### User Interface
- Modern, responsive Gradio interface
- Intuitive search and exploration
- Real-time updates
- Interactive visualizations

## 🛠️ Technical Architecture

### Core Components

1. **Knowledge Graph Engine**
   - RDFLib for RDF/Turtle parsing
   - SPARQL query support
   - Entity relationship management
   - Metadata handling

2. **Search Engine**
   - Entity-based search
   - Metadata indexing
   - Fuzzy matching
   - Real-time results

3. **Visualization Engine**
   - NetworkX for graph processing
   - Plotly for interactive visualizations
   - Dynamic graph layout
   - Custom styling and formatting

4. **AI Integration**
   - OpenAI GPT integration
   - Context-aware explanations
   - Natural language processing
   - Business context understanding

### Technology Stack

- **Backend**: Python 3.7+
- **Web Framework**: Gradio 4.19.2
- **Knowledge Graph**: RDFLib 7.0.0
- **Graph Processing**: NetworkX 3.2.1
- **Visualization**: Plotly 5.18.0
- **AI Integration**: OpenAI API
- **Environment Management**: python-dotenv 1.0.1

## 📋 Prerequisites

- Python 3.7 or higher
- OpenAI API key
- Internet connection for LLM functionality
- Basic understanding of RDF/Turtle format
- Git (for version control)

## 🚀 Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd business-entity-explorer
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root:
```
OPENAI_API_KEY=your_api_key_here
```

5. Ensure your knowledge graph file (`knowledge_graph.ttl`) is in the project root directory.

## 💻 Usage

### Starting the Application

Run the application:
```bash
python app.py
```

The application will start a local server and provide a URL (typically http://localhost:7860) where you can access the interface.

### Using the Interface

1. **Search**
   - Enter search terms in the search box
   - Press Enter or click the Search button
   - View matching entities in the results

2. **Explore Results**
   - View entity details in the information panel
   - Interact with the relationship graph
   - Read AI-generated explanations

3. **Graph Interaction**
   - Zoom: Mouse wheel or pinch gesture
   - Pan: Click and drag
   - Hover: View additional information
   - Click: Select nodes for detailed view

### Example Queries

- "Find customer-related entities"
- "Show payment processing relationships"
- "Display business customer attributes"
- "Explore complaint handling entities"

## 🔧 Configuration

### Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key
- Additional configuration can be added to the `.env` file

### Knowledge Graph Format

The application expects a Turtle (.ttl) format knowledge graph with the following structure:
- Entities with `commonLabel` properties
- Relationships between entities
- Metadata attributes
- Definitions and descriptions

## 📊 Data Structure

### Entity Format
```turtle
@prefix ex: <http://example.org/> .

<http://example.org/entity/xxx> 
    ex:commonLabel "Entity Name" ;
    ex:definition "Entity definition" ;
    ex:dictionary "Category" ;
    ex:hasAttribute <http://example.org/attribute/xxx> .
```

### Relationship Format
```turtle
<http://example.org/entity/xxx> 
    ex:hasAttribute <http://example.org/attribute/yyy> ;
    ex:relatedTo <http://example.org/entity/zzz> .
```

## 🔍 Troubleshooting

### Common Issues

1. **Graph Not Loading**
   - Check file path and format
   - Verify Turtle syntax
   - Ensure proper namespace definitions

2. **Search Not Working**
   - Verify entity labels exist
   - Check search query format
   - Ensure proper indexing

3. **Visualization Issues**
   - Clear browser cache
   - Check browser compatibility
   - Verify Plotly installation

### Debugging

Enable debug mode by setting:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

### Development Guidelines

- Follow PEP 8 style guide
- Add tests for new features
- Update documentation
- Maintain backward compatibility

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- OpenAI for GPT API
- RDFLib team for knowledge graph support
- Gradio team for the web interface
- NetworkX and Plotly for visualization

## 📞 Support

For support, please:
1. Check the troubleshooting guide
2. Search existing issues
3. Create a new issue with:
   - Detailed description
   - Steps to reproduce
   - Expected vs actual behavior
   - Environment details 