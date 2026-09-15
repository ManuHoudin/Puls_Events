from pathlib import Path
import os
import asyncio

from tests.ragas.RAGasEvaluator import RAGasEvaluator
from data.evaluation.rag_test_cases import TEST_CASES
from mistralai.client import Mistral
from langchain_openai import ChatOpenAI

from src.services.FaissRepository import FaissRepository
from src.services.RAGService import RagService
from ragas.embeddings import BaseRagasEmbeddings

# Création des services nécessaires pour l'évaluation
mistral_client = Mistral()
llm = ChatOpenAI(
    model="qwen3.8-max",
    api_key=os.environ["DASHSCOPE_API_KEY"],
    base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    temperature=0.2,
)

faiss_repository = FaissRepository(
    index_path=Path("data/index_faiss/evenements.faiss"),
    metadata_path=Path("data/index_faiss/metadata.parquet"),
)

faiss_repository.load()

rag_service = RagService(
    faiss_repository=faiss_repository,
    mistral_client=mistral_client,
    llm=llm,
)

evaluator = RAGasEvaluator(
    rag_service=rag_service
)

# Fonction d'évaluation de tous les cas de test
async def evaluate_all():

    results = []

    for test in TEST_CASES:

        print("=" * 80)
        print(test["question"])

        result = await evaluator.evaluate_case(
            question=test["question"],
            reference_uids=test["reference_uids"],
        )

        results.append(result)

        print(
            "Retrieved :",
            result["retrieved_uids"]
        )

        print(
            "Precision :",
            result["precision"]
        )

        print(
            "Recall :",
            result["recall"]
        )
    return results

results = asyncio.run(evaluate_all())

# Création d'un wrapper Ragas pour les embeddings Mistral
class MistralRagasEmbeddings(BaseRagasEmbeddings):

    def __init__(self, client: Mistral):
        self.client = client

    def embed_query(self, text: str) -> list[float]:
        response = self.client.embeddings.create(
            model="mistral-embed",
            inputs=[text],
        )

        embedding = response.data[0].embedding

        if embedding is None:
                    raise RuntimeError(
                        "Mistral n'a pas retourné d'embedding."
                    )
        return embedding

    def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:

        response = self.client.embeddings.create(
            model="mistral-embed",
            inputs=texts,
        )

        embeddings = []

        for item in sorted(
            response.data,
            key=lambda item: (
                item.index
                if item.index is not None
                else -1
            ),
        ):
            if item.embedding is None:
                raise RuntimeError(
                    "Mistral n'a pas retourné d'embedding pour un document."
                )

            embeddings.append(item.embedding)

        return embeddings

    async def aembed_query(
        self,
        text: str,
    ) -> list[float]:

        return self.embed_query(text)

    async def aembed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:

        return self.embed_documents(texts)