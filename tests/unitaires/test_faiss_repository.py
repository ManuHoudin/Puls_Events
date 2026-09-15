import json

import faiss
import numpy as np
import pandas as pd
from prompt_toolkit import document
from prompt_toolkit import document
import pytest

from src.services.FaissRepository import FaissRepository


def creer_metadata_evenements():
    return pd.DataFrame(
        {
            "index_evenement": [0, 1],
            "uid": [1001, 1002],
            "title": [
                "Concert classique",
                "Festival de musique",
            ],
            "timings": [
                json.dumps(
                    [
                        {
                            "begin": "2026-09-20T18:00:00+02:00",
                            "end": "2026-09-20T20:00:00+02:00",
                        }
                    ]
                ),
                json.dumps([]),
            ],
            "location.name": [
                "Jardin de l'hôtel de Limur",
                "Le Liberté",
            ],
            "location.city": [
                "Vannes",
                "Rennes",
            ],
            "location.region": [
                "Bretagne",
                "Bretagne",
            ],
            "location.address": [
                "1 rue Thiers",
                "2 boulevard de la Liberté",
            ],
            "agenda_titre_source": [
                "Agenda Vannes",
                "Agenda Rennes",
            ],
        }
    )


def creer_df_embeddings():
    return pd.DataFrame(
        {
            "index_evenement": [0, 0, 1],
            "numero_chunk": [0, 1, 0],
            "texte_chunk": [
                "Premier chunk du concert",
                "Deuxième chunk du concert",
                "Chunk du festival",
            ],
            "embedding": [
                [1.0, 0.0, 0.0],
                [0.9, 0.1, 0.0],
                [0.0, 1.0, 0.0],
            ],
        }
    )


# Test de la fonction build pour vérifier que l'index FAISS et les métadonnées sont correctement construits.
def test_build_construit_index_et_metadonnees(
    tmp_path,
):
    repository = FaissRepository(
        index_path=tmp_path / "test.faiss",
        metadata_path=tmp_path / "metadata.parquet",
    )

    nb_elements, dimension = repository.build(
        creer_df_embeddings(),
        creer_metadata_evenements(),
    )

    assert nb_elements == 3
    assert dimension == 3

    assert repository.index is not None
    assert repository.metadata is not None

    assert repository.index.ntotal == 3
    assert len(repository.metadata) == 3

    assert list(
        repository.metadata["faiss_id"]
    ) == [0, 1, 2]

    assert (
        repository.index_path.exists()
    )

    assert (
        repository.metadata_path.exists()
    )


# Test de la fonction build pour vérifier que les métadonnées sont correctement associées aux embeddings.
def test_build_associe_correctement_les_metadonnees(
    tmp_path,
):
    repository = FaissRepository(
        index_path=tmp_path / "test.faiss",
        metadata_path=tmp_path / "metadata.parquet",
    )

    repository.build(
        creer_df_embeddings(),
        creer_metadata_evenements(),
    )

    metadata = repository.metadata

    assert metadata is not None

    # Deux chunks appartiennent à l'événement 0.
    assert (
        metadata["index_evenement"].tolist()
        == [0, 0, 1]
    )

    assert metadata["uid"].tolist() == [
        1001,
        1001,
        1002,
    ]

    assert metadata["numero_chunk"].tolist() == [
        0,
        1,
        0,
    ]

    assert metadata["title"].tolist() == [
        "Concert classique",
        "Concert classique",
        "Festival de musique",
    ]

# Test pour le refus de suvegarde sans index ou sans métadonnées.
def test_save_refuse_sans_index(tmp_path):
    repository = FaissRepository(
        index_path=tmp_path / "test.faiss",
        metadata_path=tmp_path / "metadata.parquet",
    )

    repository.metadata = pd.DataFrame()

    with pytest.raises(
        RuntimeError,
        match="Aucun index FAISS",
    ):
        repository.save()

# Test de la recharge correcte de l'index FAISS et des métadonnées à partir des fichiers sauvegardés.
def test_load_recharge_index_et_metadonnees(
    tmp_path,
):
    index_path = tmp_path / "test.faiss"
    metadata_path = tmp_path / "metadata.parquet"

    repository = FaissRepository(
        index_path=index_path,
        metadata_path=metadata_path,
    )

    repository.build(
        creer_df_embeddings(),
        creer_metadata_evenements(),
    )

    nouveau_repository = FaissRepository(
        index_path=index_path,
        metadata_path=metadata_path,
    )

    nouveau_repository.load()

    assert nouveau_repository.index is not None
    assert nouveau_repository.metadata is not None

    assert (
        nouveau_repository.index.ntotal
        == 3
    )

    assert len(
        nouveau_repository.metadata
    ) == 3

    assert (
        nouveau_repository.metadata["uid"]
        .tolist()
        == [1001, 1001, 1002]
    )

# Test la pertinence des documents retournés par la recherche FAISS en fonction d'un vecteur de question donné.
def test_search_retourne_documents_pertinents(
    tmp_path,
):
    repository = FaissRepository(
        index_path=tmp_path / "test.faiss",
        metadata_path=tmp_path / "metadata.parquet",
    )

    repository.build(
        creer_df_embeddings(),
        creer_metadata_evenements(),
    )

    # Le vecteur est très proche du premier
    # événement.
    question_vector = np.array(
        [[1.0, 0.0, 0.0]],
        dtype=np.float32,
    )

    documents = repository.search(
        question_vector,
        k=2,
    )

    assert len(documents) == 2

    premier = documents[0]

    assert premier.metadata["uid"] == 1001
    assert (premier.metadata["faiss_id"] == 0)

    assert (premier.metadata["title"] == "Concert classique")

    assert (premier.metadata[ "agenda_titre_source"]
        == "Agenda Vannes")

    assert (premier.metadata["score_similarite"] > 0.9)
    assert "Titre : Concert classique" in (premier.page_content)
    assert ("Ville :\nVannes" in premier.page_content)
    assert ("Région :\nBretagne" in premier.page_content)
    assert "Lieu :\nJardin de l'hôtel de Limur" in premier.page_content
    assert "Adresse :\n1 rue Thiers" in premier.page_content

    assert (
        "20/09/2026 18:00 - 20:00"
        in premier.page_content
    )

# test le formatage des horaires y compris le cas > 5 occurences pour un événement récurrent.
def test_search_formate_horaires_recurrents(
    tmp_path,
):
    repository = FaissRepository(
        index_path=tmp_path / "test.faiss",
        metadata_path=tmp_path / "metadata.parquet",
    )

    timings = [
        {
            "begin": (
                f"2026-09-{20 + i:02d}"
                "T18:00:00+02:00"
            ),
            "end": (
                f"2026-09-{20 + i:02d}"
                "T20:00:00+02:00"
            ),
        }
        for i in range(6)
    ]

    metadata = creer_metadata_evenements()

    metadata.loc[
        metadata["index_evenement"] == 0,
        "timings",
    ] = json.dumps(timings)

    repository.build(
        creer_df_embeddings(),
        metadata,
    )

    question_vector = np.array(
        [[1.0, 0.0, 0.0]],
        dtype=np.float32,
    )

    documents = repository.search(
        question_vector,
        k=1,
    )

    contenu = documents[0].page_content

    assert (
        "Du 20/09/2026 au 25/09/2026"
        in contenu
    )

    assert (
        "créneaux récurrents : 18:00 - 20:00"
        in contenu
    )