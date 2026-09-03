from __future__ import annotations

import numpy as np
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_mistralai import ChatMistralAI

from dotenv import load_dotenv


load_dotenv()

# ============================================================
# Gestionnaire RAG
# ============================================================

class RagService:
    """
    Orchestration du chatbot RAG.
    """

    def __init__(
        self,
        faiss_repository,
        mistral_client,
        llm : ChatMistralAI,
    ):
        self.faiss_repository = faiss_repository

        self.mistral_client = mistral_client

        self.llm = llm

        self.tool = self._create_tool()

        self.agent = create_agent(
            model=self.llm,
            tools=[self.tool],
            system_prompt="""
            Tu es un assistant spécialisé dans
            les événements culturels.

            Règles :

            1. Utilise l'outil de recherche lorsqu'une
            question concerne les événements culturels
            de la base.

            2. Pour une question générale ne nécessitant
            pas les données de la base, réponds directement.

            3. N'invente jamais un événement, une date,
            un horaire, un lieu ou une autre information
            absente des résultats.

            4. Si aucun résultat pertinent n'est trouvé,
            indique-le clairement.

            5. Prends en compte les contraintes explicites
            de lieu, de date et de catégorie.

            6. Réponds de manière claire et concise.
            """,
        )

    def _create_tool(self):

        service = self

        @tool
        def rechercher_evenements_culturels(
            question: str,
        ) -> str:
            """
            Recherche des événements culturels
            dans la base FAISS.
            """

            response = (
                service.mistral_client
                .embeddings
                .create(
                    model="mistral-embed",
                    inputs=[question],
                )
            )

            vecteur = np.asarray(
                [
                    response
                    .data[0]
                    .embedding
                ],
                dtype=np.float32,
            )

            documents = (
                service.faiss_repository
                .search(
                    vecteur,
                    k=5,
                )
            )

            if not documents:
                return (
                    "Aucun événement "
                    "pertinent trouvé."
                )

            resultats = []

            for document in documents:

                resultats.append(
                    f"""
                    Titre :
                    {document.metadata["title"]}

                    Score de similarité :
                    {document.metadata["score_similarite"]:.3f}

                    {document.page_content}
                    """.strip()
                )

            return "\n\n---\n\n".join(
                resultats
            )

        return rechercher_evenements_culturels

    def ask(
        self,
        question: str,
    ) -> str:

        response = self.agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": question,
                    }
                ]
            }
        )

        return (
            response["messages"][-1]
            .content
        )
