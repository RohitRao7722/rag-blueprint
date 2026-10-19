FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    FASTEMBED_CACHE_PATH=/models/fastembed \
    TIKTOKEN_CACHE_DIR=/models/tiktoken

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the embedding model and tokenizer into the image so runs don't re-download them.
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5')" \
 && python -c "import tiktoken; tiktoken.get_encoding('o200k_base')"

COPY run.py compare_overlap.py no_context.py ./

# The paper is downloaded from arXiv at run time (not redistributed in the image).
ENTRYPOINT ["python"]
CMD ["run.py"]
