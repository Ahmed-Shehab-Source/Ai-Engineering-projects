from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

DATA_DIR = "chatbot_demo_data"

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=[
        "\n## ",
        "\n### ",
        "\n\n",
        "\n",
        ". ",
        " ",
        ""
    ]
)

documents = []

for md_file in Path(DATA_DIR).glob("*.md"):
    text = md_file.read_text(encoding="utf-8")

    doc = Document(
        page_content=text,
        metadata={"source": md_file.name}
    )

    documents.extend(splitter.split_documents([doc]))

print(f"Total chunks: {len(documents)}")


import json

with open("chunks.json", "w", encoding="utf-8") as f:
    json.dump(
        [
            {
                "content": d.page_content,
                "metadata": d.metadata
            }
            for d in documents
        ],
        f,
        ensure_ascii=False,
        indent=2,
    )