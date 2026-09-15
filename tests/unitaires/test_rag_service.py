from types import SimpleNamespace
from unittest.mock import Mock

from langchain_core.documents import Document

from src.services.RAGService import RagService


def creer_document(
    uid,
    title,
    contenu,
    score=0.9,
):
    return Document(
        page_content=contenu,
        metadata={
            "uid": uid,
            "title": title,
            "score_similarite": score,
        },
    )


def creer_mistral_client(embedding):
    client = Mock()

    client.embeddings.create.return_value = SimpleNamespace(
        data=[
            SimpleNamespace(
                index=0,
                embedding=embedding,
            )
        ]
    )

    return client


def test_creation_service():
    faiss_repository = Mock()

    mistral_client = creer_mistral_client(
        [0.1, 0.2, 0.3]
    )

    llm = Mock()

    service = RagService(
        faiss_repository=faiss_repository,
        mistral_client=mistral_client,
        llm=llm,
    )

    assert service.faiss_repository is faiss_repository
    assert service.mistral_client is mistral_client
    assert service.llm is llm
    assert service.last_documents == []
    assert service.agent is not None
    assert service.tool is not None


def test_rechercher_evenements_culturels_retourne_les_resultats():
    documents = [
        creer_document(
            uid=1001,
            title="Concert classique",
            contenu="Concert classique à Vannes",
        ),
        creer_document(
            uid=1002,
            title="Festival",
            contenu="Festival à Rennes",
        ),
    ]

    faiss_repository = Mock()
    faiss_repository.search.return_value = documents

    mistral_client = creer_mistral_client(
        [0.1, 0.2, 0.3]
    )

    service = RagService(
        faiss_repository=faiss_repository,
        mistral_client=mistral_client,
        llm=Mock(),
    )

    resultat = service.tool.invoke(
        {
            "question": "Quels concerts sont prévus ?"
        }
    )

    assert "Concert classique à Vannes" in resultat
    assert "Festival à Rennes" in resultat

    mistral_client.embeddings.create.assert_called_once_with(
        model="mistral-embed",
        inputs=["Quels concerts sont prévus ?"],
    )

    faiss_repository.search.assert_called_once()

    assert service.last_documents == documents


def test_rechercher_evenements_culturels_supprime_les_doublons_uid():
    documents = [
        creer_document(
            uid=1001,
            title="Concert classique",
            contenu="Premier chunk",
        ),
        creer_document(
            uid=1001,
            title="Concert classique",
            contenu="Deuxième chunk",
        ),
        creer_document(
            uid=1002,
            title="Festival",
            contenu="Festival à Rennes",
        ),
    ]

    faiss_repository = Mock()
    faiss_repository.search.return_value = documents

    mistral_client = creer_mistral_client(
        [0.1, 0.2, 0.3]
    )

    service = RagService(
        faiss_repository=faiss_repository,
        mistral_client=mistral_client,
        llm=Mock(),
    )

    resultat = service.tool.invoke(
        {
            "question": "Quels événements sont prévus ?"
        }
    )

    assert "Premier chunk" in resultat
    assert "Deuxième chunk" not in resultat
    assert "Festival à Rennes" in resultat

    assert resultat.count("Premier chunk") == 1
    assert resultat.count("Deuxième chunk") == 0
    assert resultat.count("Festival à Rennes") == 1


def test_rechercher_evenements_culturels_sans_resultat():
    faiss_repository = Mock()
    faiss_repository.search.return_value = []

    mistral_client = creer_mistral_client(
        [0.1, 0.2, 0.3]
    )

    service = RagService(
        faiss_repository=faiss_repository,
        mistral_client=mistral_client,
        llm=Mock(),
    )

    resultat = service.tool.invoke(
        {
            "question": "Quels événements sont prévus ?"
        }
    )

    assert resultat == (
        "Aucun événement pertinent trouvé."
    )

    assert service.last_documents == []


def test_ask_retourne_la_reponse_de_l_agent():
    faiss_repository = Mock()
    mistral_client = creer_mistral_client(
        [0.1, 0.2, 0.3]
    )

    service = RagService(
        faiss_repository=faiss_repository,
        mistral_client=mistral_client,
        llm=Mock(),
    )

    message_final = SimpleNamespace(
        content="Il y a un concert à Vannes."
    )

    service.agent = Mock()

    service.agent.invoke.return_value = {
        "messages": [
            SimpleNamespace(
                content="Résultat intermédiaire"
            ),
            message_final,
        ]
    }

    resultat = service.ask(
        "Quels concerts sont prévus à Vannes ?"
    )

    assert resultat == (
        "Il y a un concert à Vannes."
    )

    service.agent.invoke.assert_called_once_with(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Quels concerts sont prévus à Vannes ?"
                    ),
                }
            ]
        }
    )

def test_ask_with_context_retourne_reponse_et_contextes():
    faiss_repository = Mock()

    document = creer_document(
        uid=1001,
        title="Concert classique",
        contenu="Concert à Vannes le 20 septembre",
        score=0.95,
    )

    faiss_repository.search.return_value = [document]

    mistral_client = creer_mistral_client(
        [0.1, 0.2, 0.3]
    )

    service = RagService(
        faiss_repository=faiss_repository,
        mistral_client=mistral_client,
        llm=Mock(),
    )

    service.agent = Mock()

    def invoke_agent(*args, **kwargs):
        # Simulation de l'appel de l'agent à l'outil
        service.last_documents = [document]

        return {
            "messages": [
                SimpleNamespace(
                    content="Un concert est prévu à Vannes."
                )
            ]
        }

    service.agent.invoke.side_effect = invoke_agent

    answer, contexts = service.ask_with_context(
        "Quels concerts sont prévus à Vannes ?")

    assert answer == ("Un concert est prévu à Vannes.")

    assert contexts == [
        {
            "uid": 1001,
            "title": "Concert classique",
            "content": (
                "Concert à Vannes le 20 septembre"
            ),
            "score": 0.95,
        }
    ]