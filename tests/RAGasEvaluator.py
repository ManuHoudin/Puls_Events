from __future__ import annotations


from ragas import SingleTurnSample
from ragas.llms import llm_factory
from ragas.embeddings import embedding_factory
from ragas.embeddings import BaseRagasEmbeddings


from mistralai.client import Mistral

from ragas.metrics import (
    IDBasedContextPrecision,
    IDBasedContextRecall,
    Faithfulness,
    ResponseRelevancy,
)

# Utilisation d'un LLM Qwen3.8 et d'un Embedding Mistral pour l'évaluation faithfulness et response_relevancy
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
                    "Mistral n'a pas retourné d'embedding."
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

# Calcul des 4 métriques RAGAS
class RAGasEvaluator:

    def __init__(self, rag_service):

        self.rag_service = rag_service

        self.id_precision = (IDBasedContextPrecision())
        self.id_recall = (IDBasedContextRecall())

        self.faithfulness = Faithfulness(llm=rag_service.llm);
        self.ragas_embeddings = MistralRagasEmbeddings(rag_service.mistral_client)
        self.response_relevancy = ResponseRelevancy(
            llm=rag_service.llm,
            embeddings=self.ragas_embeddings,
        )

    async def evaluate_case(
        self,
        question: str,
        reference_uids: list[str | int],
    ) -> dict:

        # Exécution du RAG
        print(">>> EVALUATE_CASE START")
        answer, contexts = self.rag_service.ask_with_context(question)

        # UID récupérés par le retriever
        retrieved_uids = list(
            dict.fromkeys(
                document.metadata["uid"]
                for document in self.rag_service.last_documents
            )
        )


        print(">>> RETRIEVAL DONE")
        print("Retrieved :", retrieved_uids)
        # Échantillon Ragas
        sample_ids = SingleTurnSample(
            retrieved_context_ids=retrieved_uids,
            reference_context_ids=reference_uids,
        )

        sample_rag = SingleTurnSample(
            user_input=question,
            response=answer,
            retrieved_contexts=[
                context["content"]
                for context in contexts
            ],
        )

        # Scores
        precision = await self.id_precision.single_turn_ascore(sample_ids)
        print("Precision :", precision)
        recall = await self.id_recall.single_turn_ascore(sample_ids)
        print("Recall :", recall)
        faithfulness = await self.faithfulness.single_turn_ascore(sample_rag)
        print("Faithfulness :", faithfulness)
        response_relevancy = await self.response_relevancy.single_turn_ascore(sample_rag)
        print("Response Relevancy :",response_relevancy)


        return {
            # "question": question,
            # "reference_uids": reference_uids,
            "retrieved_uids": retrieved_uids,
            # "answer": answer,
            "precision": precision,
            "recall": recall,
            "faithfulness": faithfulness,
            "response_relevancy": response_relevancy,
        }