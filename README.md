# langchain-turbopuffer

Use Turbopuffer as a vector store for LangChain.

## Usage

```python
poetry add git+https://github.com/skyzh/langchain-turbopuffer
# see example.py for usage
```

## Local Development

```python
git clone https://github.com/skyzh/langchain-turbopuffer
cd langchain-turbopuffer
poetry env use 3.12
poetry install

ollama pull mxbai-embed-large llama3.2
ollama run llama3.2
export TURBOPUFFER_API_KEY=your_api_key
poetry run python example.py
poetry run python example.py --skip-load
```

In the example, you can ask questions like "What is prompt engineering?"
