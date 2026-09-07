from pathlib import Path

from tests.RAGasEvaluator import RAGasEvaluator
from data.evaluation.rag_test_cases import TEST_CASES
from mistralai.client import Mistral
from langchain_mistralai import ChatMistralAI

from services.FaissRepository import FaissRepository
from services.RAGService import RagService


mistral_client = Mistral()
llm = ChatMistralAI(
            model_name="mistral-medium-latest",
            temperature=0.2,
            timeout=10,
            max_retries=0,
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


results = []

for test in TEST_CASES:

    print("=" * 80)
    print(test["question"])

    result = evaluator.evaluate_case(
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