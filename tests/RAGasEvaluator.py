from __future__ import annotations

import os

from mistralai.client import Mistral

from ragas.llms import llm_factory
from ragas.embeddings import embedding_factory

from ragas.metrics.collections import (
    IDBasedContextPrecision,
    IDBasedContextRecall,
    Faithfulness,
    ResponseRelevancy,
)


class RAGasEvaluator:
    """
    Évalue le retriever et les réponses du système RAG
    avec Ragas.
    """

    def __init__(self, rag_service):

        self.rag_service = rag_service

        # ----------------------------------------------------
        # Client Mistral utilisé par Ragas
        # ----------------------------------------------------

        self.mistral_client = Mistral(
            api_key=os.environ["MISTRAL_API_KEY"]
        )

        # ----------------------------------------------------
        # LLM utilisé comme juge par Ragas
        # ----------------------------------------------------

        self.evaluator_llm = llm_factory(
            "mistral-small-latest",
            provider="mistral",
            client=self.mistral_client,
        )

        # ----------------------------------------------------
        # Embeddings utilisés par ResponseRelevancy
        # ----------------------------------------------------

        self.evaluator_embeddings = embedding_factory(
            provider="mistral",
            model="mistral-embed",
            client=self.mistral_client,
        )

        # ----------------------------------------------------
        # Métriques
        # ----------------------------------------------------

        self.id_precision = (
            IDBasedContextPrecision()
        )

        self.id_recall = (
            IDBasedContextRecall()
        )

        self.faithfulness = Faithfulness(
            llm=self.evaluator_llm
        )

        self.response_relevancy = ResponseRelevancy(
            llm=self.evaluator_llm,
            embeddings=self.evaluator_embeddings,
        )

    def evaluate_case(
        self,
        question: str,
        reference_uids: list[str],
        reference_answer: str,
    ) -> dict:
        """
        Évalue une question.

        reference_uids :
            IDs des événements/chunks considérés comme pertinents.

        reference_answer :
            Réponse humaine de référence.
        """

        # ----------------------------------------------------
        # Exécution du RAG
        # ----------------------------------------------------

        answer, contexts = (
            self.rag_service.ask_with_context(
                question
            )
        )

        # Récupération des IDs réellement trouvés
        retrieved_uids = [
            document.metadata["uid"]
            for document in (
                self.rag_service.last_documents
            )
        ]

        # ----------------------------------------------------
        # Métriques ID-based
        # ----------------------------------------------------

        precision_result = (
            self.id_precision.score(
                retrieved_context_ids=retrieved_uids,
                reference_context_ids=reference_uids,
            )
        )

        recall_result = (
            self.id_recall.score(
                retrieved_context_ids=retrieved_uids,
                reference_context_ids=reference_uids,
            )
        )

        # ----------------------------------------------------
        # Faithfulness
        # ----------------------------------------------------

        faithfulness_result = (
            self.faithfulness.score(
                user_input=question,
                response=answer,
                retrieved_contexts=contexts,
            )
        )

        # ----------------------------------------------------
        # Response Relevancy
        # ----------------------------------------------------

        relevancy_result = (
            self.response_relevancy.score(
                user_input=question,
                response=answer,
            )
        )

        return {
            "question": question,
            "reference_answer": reference_answer,
            "generated_answer": answer,
            "retrieved_uids": retrieved_uids,
            "reference_uids": reference_uids,
            "id_precision": precision_result.value,
            "id_recall": recall_result.value,
            "faithfulness": faithfulness_result.value,
            "response_relevancy": relevancy_result.value,
        }