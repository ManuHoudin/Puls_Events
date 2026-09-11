from __future__ import annotations

import numpy as np
import time
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI


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
        llm : ChatOpenAI,
    ):
        self.faiss_repository = faiss_repository

        self.mistral_client = mistral_client

        self.llm = llm

        self.tool = self._create_tool()

        self.last_documents = []

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

            7. Limite toi à la région Bretagne, en France.
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

            print(">>> TOOL START")
            start_tool = time.perf_counter()

            response = (
                service.mistral_client
                .embeddings
                .create(
                    model="mistral-embed",
                    inputs=[question],
                )
            )
            print(f">>> EMBEDDING : "f"{time.perf_counter() - start_tool:.2f}s")

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

            print(f">>> FAISS : "f"{time.perf_counter() - start_tool:.2f}s")
            # print(">>> TOOL RESULTS :",[doc.metadata["uid"] for doc in documents])

            service.last_documents = documents

            if not documents:
                return ("Aucun événement pertinent trouvé.")

            resultats = []
            documents_uniques = []
            uids_vus = set()

            for document in documents:
                uid = document.metadata["uid"]
                if uid not in uids_vus:
                    documents_uniques.append(document)
                    uids_vus.add(uid)

            # For improvement
           
            print(documents_uniques[0].metadata)
            for document in documents_uniques:
                print(document.page_content[:5000])
                resultats.append(document.page_content)

            #   f"""
            # resultats.append
            # Titre :
            #                     {document.metadata["title"]}
            
            #                     Score de similarité :
            #                     {document.metadata["score_similarite"]:.3f}
            # """.strip()

            print(">>> TOOL END")
            print(f">>> TOOL END : "f"{time.perf_counter() - start_tool:.2f}s")
            return "\n\n---\n\n".join(
                resultats
            )

            
        return rechercher_evenements_culturels

    def ask(
        self,
        question: str,
    ) -> str:

        print(">>> 1. AVANT INVOKE")
        start_agent = time.perf_counter()
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

        # For improvement
        for message in response["messages"]:
            print(type(message).__name__, getattr(message, "response_metadata", None))
            print("CONTENT LENGTH :", len(message.content))
            print("CONTENT :", message.content[:1000])
            

        print(">>> 2. APRES INVOKE")
        print(f">>> 2. APRES INVOKE : "f"{time.perf_counter() - start_agent:.2f}s")
        # print(response)
        return response["messages"][-1].content
    
# Méthode utilisée pour les tests RAGAS
    def ask_with_context(self, question: str,):
        """
        Retourne la réponse du chatbot ainsi que
        les contextes récupérés par FAISS.
        """

    # Réinitialisation afin de ne pas conserver
    # les documents d'une question précédente.
        self.last_documents = []

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

        answer = response["messages"][-1].content

        contexts = [
             {
                "uid": document.metadata["uid"],
                "title": document.metadata["title"],
                "content": document.page_content,
                "score": document.metadata["score_similarite"],
            }
            for document in self.last_documents
        ]

        return answer, contexts
