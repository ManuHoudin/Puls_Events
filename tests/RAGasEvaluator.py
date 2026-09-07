from __future__ import annotations


from ragas import SingleTurnSample
from ragas.llms import llm_factory
from ragas.embeddings import embedding_factory

from ragas.metrics import (
    IDBasedContextPrecision,
    IDBasedContextRecall,
    Faithfulness,
    ResponseRelevancy,
)



class RAGasEvaluator:

    def __init__(self, rag_service):

        self.rag_service = rag_service

        self.id_precision = (
            IDBasedContextPrecision()
        )

        self.id_recall = (
            IDBasedContextRecall()
        )

    def evaluate_case(
        self,
        question: str,
        reference_uids: list[str | int],
    ) -> dict:

        # Exécution du RAG
        answer, contexts = (
            self.rag_service.ask_with_context(
                question
            )
        )

        # UID récupérés par le retriever
        retrieved_uids = list(
            dict.fromkeys(
                document.metadata["uid"]
                for document
                in self.rag_service.last_documents
            )
        )

        # Échantillon Ragas
        sample = SingleTurnSample(
            retrieved_context_ids=retrieved_uids,
            reference_context_ids=reference_uids,
        )

        # Scores
        precision = (
            self.id_precision.single_turn_ascore(
                sample
            )
        )

        recall = (
            self.id_recall.single_turn_ascore(
                sample
            )
        )

        return {
            "question": question,
            "reference_uids": reference_uids,
            "retrieved_uids": retrieved_uids,
            "answer": answer,
            "precision": precision,
            "recall": recall,
        }