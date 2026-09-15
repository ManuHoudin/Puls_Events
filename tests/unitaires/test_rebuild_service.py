from unittest.mock import Mock

import pandas as pd
import pytest

from src.services.RebuildService import RebuildService

# Mock des services
def creer_service():
    openagenda = Mock()
    processor = Mock()
    embedding_service = Mock()
    faiss_repository = Mock()

    service = RebuildService(
        openagenda_client=openagenda,
        event_processor=processor,
        embedding_service=embedding_service,
        faiss_repository=faiss_repository,
    )

    return (
        service,
        openagenda,
        processor,
        embedding_service,
        faiss_repository,
    )

# Test de l'appel de la méthode rebuild() et de l'exécution de toutes les étapes
def test_rebuild_execute_toutes_les_etapes():
    (
        service,
        openagenda,
        processor,
        embedding_service,
        faiss_repository,
    ) = creer_service()

    df_raw = pd.DataFrame(
        {
            "title": ["Concert"],
            "location.city": ["Vannes"],
        }
    )

    df_clean = pd.DataFrame(
        {
            "index_evenement": [0],
            "title": ["Concert"],
        }
    )

    df_chunks = pd.DataFrame(
        {
            "index_evenement": [0],
            "numero_chunk": [0],
            "texte_chunk": ["Concert à Vannes"],
        }
    )

    df_embeddings = pd.DataFrame(
        {
            "index_evenement": [0],
            "numero_chunk": [0],
            "texte_chunk": ["Concert à Vannes"],
            "embedding": [[0.1, 0.2, 0.3]],
        }
    )

    openagenda.fetch_bretagne_events.return_value = df_raw
    processor.prepare.return_value = df_clean
    processor.create_chunks.return_value = df_chunks
    embedding_service.embed_dataframe.return_value = df_embeddings
    faiss_repository.build.return_value = (1, 3)

    resultat = service.rebuild()

    assert resultat == (1, 3)

    openagenda.fetch_bretagne_events.assert_called_once_with()

    processor.prepare.assert_called_once()

    processor.create_chunks.assert_called_once_with(
        df_clean
    )

    embedding_service.embed_dataframe.assert_called_once_with(
        df_chunks
    )

    faiss_repository.build.assert_called_once_with(
        df_embeddings,
        df_clean,
    )

# Test de l'ajout de l'index_evenement dans le DataFrame envoyé à processor.prepare()
def test_rebuild_ajoute_index_evenement():
    (
        service,
        openagenda,
        processor,
        embedding_service,
        faiss_repository,
    ) = creer_service()

    df_raw = pd.DataFrame(
        {
            "title": [
                "Concert",
                "Festival",
            ],
        }
    )

    df_clean = pd.DataFrame(
        {
            "index_evenement": [0, 1],
        }
    )

    df_chunks = pd.DataFrame(
        {
            "index_evenement": [0, 1],
            "numero_chunk": [0, 0],
            "texte_chunk": [
                "Concert",
                "Festival",
            ],
        }
    )

    df_embeddings = pd.DataFrame(
        {
            "index_evenement": [0, 1],
            "numero_chunk": [0, 0],
            "texte_chunk": [
                "Concert",
                "Festival",
            ],
            "embedding": [
                [0.1, 0.2],
                [0.3, 0.4],
            ],
        }
    )

    openagenda.fetch_bretagne_events.return_value = df_raw
    processor.prepare.return_value = df_clean
    processor.create_chunks.return_value = df_chunks
    embedding_service.embed_dataframe.return_value = df_embeddings
    faiss_repository.build.return_value = (2, 2)

    service.rebuild()

    # On récupère le DataFrame réellement envoyé
    # à processor.prepare().
    df_prepare = (
        processor.prepare.call_args.args[0]
    )

    assert list(
        df_prepare["index_evenement"]
    ) == [0, 1]

# Test de la gestion des erreurs lorsque OpenAgenda ne retourne aucun événement
def test_rebuild_refuse_un_dataframe_openagenda_vide():
    (
        service,
        openagenda,
        processor,
        embedding_service,
        faiss_repository,
    ) = creer_service()

    openagenda.fetch_bretagne_events.return_value = (
        pd.DataFrame()
    )

    with pytest.raises(
        ValueError,
        match="OpenAgenda n'a retourné aucun événement",
    ):
        service.rebuild()

    processor.prepare.assert_not_called()
    processor.create_chunks.assert_not_called()
    embedding_service.embed_dataframe.assert_not_called()
    faiss_repository.build.assert_not_called()

# Test de la gestion des erreurs lorsque le DataFrame de chunks est vide
def test_rebuild_refuse_un_dataframe_chunks_vide():
    (
        service,
        openagenda,
        processor,
        embedding_service,
        faiss_repository,
    ) = creer_service()

    df_raw = pd.DataFrame(
        {
            "title": ["Concert"],
        }
    )

    df_clean = pd.DataFrame(
        {
            "index_evenement": [0],
            "title": ["Concert"],
        }
    )

    openagenda.fetch_bretagne_events.return_value = df_raw
    processor.prepare.return_value = df_clean
    processor.create_chunks.return_value = (
        pd.DataFrame()
    )

    with pytest.raises(
        ValueError,
        match="Aucun chunk n'a été généré",
    ):
        service.rebuild()

    embedding_service.embed_dataframe.assert_not_called()
    faiss_repository.build.assert_not_called()

# Test le renvoi du nombre de chunks et de la dimension des vecteurs d'embedding après la reconstruction
def test_rebuild_retourne_nombre_chunks_et_dimension():
    (
        service,
        openagenda,
        processor,
        embedding_service,
        faiss_repository,
    ) = creer_service()

    df_raw = pd.DataFrame(
        {
            "title": ["Concert"],
        }
    )

    df_clean = pd.DataFrame(
        {
            "index_evenement": [0],
            "title": ["Concert"],
        }
    )

    df_chunks = pd.DataFrame(
        {
            "index_evenement": [0],
            "numero_chunk": [0],
            "texte_chunk": ["Concert"],
        }
    )

    df_embeddings = pd.DataFrame(
        {
            "index_evenement": [0],
            "numero_chunk": [0],
            "texte_chunk": ["Concert"],
            "embedding": [[0.1, 0.2, 0.3]],
        }
    )

    openagenda.fetch_bretagne_events.return_value = df_raw
    processor.prepare.return_value = df_clean
    processor.create_chunks.return_value = df_chunks
    embedding_service.embed_dataframe.return_value = df_embeddings
    faiss_repository.build.return_value = (42, 1024)

    resultat = service.rebuild()

    assert resultat == (42, 1024)