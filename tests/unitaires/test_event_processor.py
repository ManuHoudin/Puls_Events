import pandas as pd

from src.services.EventProcessor import EventProcessor


def test_supprimer_colonnes_peu_remplies():
    processor = EventProcessor(
        seuil_remplissage=0.5
    )

    df = pd.DataFrame(
        {
            "title": [
                "Événement 1",
                "Événement 2",
                "Événement 3",
                "Événement 4",
            ],
            "description": [
                "Description 1",
                None,
                None,
                None,
            ],
        }
    )

    result = processor.supprimer_colonnes_peu_remplies(df)

    assert "title" in result.columns
    assert "description" not in result.columns


def test_normalize_types_convertit_booleens():
    processor = EventProcessor()

    df = pd.DataFrame(
        {
            "registration": [123, 456],
            "gratuit": [
                "oui",
                "non",
            ],
        }
    )

    result = processor.normalize_types(df)

    assert str(result["registration"].dtype) == "string"

    assert result["gratuit"].tolist() == [
        True,
        False,
    ]


def test_serialize_json_columns():
    processor = EventProcessor()

    df = pd.DataFrame(
        {
            "title": [
                "Concert",
                "Festival",
            ],
            "keywords": [
                ["musique", "classique"],
                ["festival"],
            ],
        }
    )

    result = processor.serialize_json_columns(df)

    assert isinstance(
        result.loc[0, "keywords"],
        str,
    )

    assert (
        result.loc[0, "keywords"]
        == '["musique", "classique"]'
    )


def test_prepare_enchaine_les_transformations():
    processor = EventProcessor()

    df = pd.DataFrame(
        {
            "title": [
                "Concert",
                "Festival",
            ],
            "gratuit": [
                "oui",
                "non",
            ],
            "keywords": [
                ["musique"],
                ["festival"],
            ],
        }
    )

    result = processor.prepare(df)

    assert result["gratuit"].tolist() == [
        True,
        False,
    ]

    assert isinstance(
        result.loc[0, "keywords"],
        str,
    )


def test_create_chunks_cree_des_chunks():
    processor = EventProcessor(
        chunk_size=100,
        chunk_overlap=20,
    )

    df = pd.DataFrame(
        {
            "title": [
                "Concert classique",
            ],
            "description": [
                "Une très longue description "
                "qui doit être suffisamment longue "
                "pour produire plusieurs chunks. "
                "Le concert présente plusieurs œuvres "
                "de musique classique avec différents "
                "interprètes et instruments.",
            ],
            "location.city": [
                "Vannes",
            ],
        }
    )

    result = processor.create_chunks(df)

    assert not result.empty

    assert {
        "index_evenement",
        "numero_chunk",
        "texte_chunk",
    }.issubset(result.columns)

    assert result["numero_chunk"].tolist() == list(
        range(len(result))
    )

    assert all(
        isinstance(texte, str)
        and texte.strip()
        for texte in result["texte_chunk"]
    )


def test_tous_les_chunks_d_un_evenement_ont_le_meme_id():
    processor = EventProcessor(
        chunk_size=100,
        chunk_overlap=20,
    )

    df = pd.DataFrame(
        {
            "title": [
                "Concert classique",
                "Festival de musique",
            ],
            "description": [
                (
                    "Description très longue du premier "
                    "événement. " * 20
                ),
                (
                    "Description très longue du deuxième "
                    "événement. " * 20
                ),
            ],
            "location.city": [
                "Vannes",
                "Rennes",
            ],
        },
        index=[
            12345,
            67890,
        ],
    )

    result = processor.create_chunks(df)

    # Les deux événements doivent avoir produit
    # des chunks.
    assert set(result["index_evenement"]) == {
        12345,
        67890,
    }

    # Tous les chunks du premier événement
    # doivent conserver son identifiant.
    chunks_evenement_1 = result[
        result["index_evenement"] == 12345
    ]

    assert len(chunks_evenement_1) > 1
    assert (
        chunks_evenement_1["index_evenement"]
        .nunique()
        == 1
    )

    # Même vérification pour le deuxième.
    chunks_evenement_2 = result[
        result["index_evenement"] == 67890
    ]

    assert len(chunks_evenement_2) > 1
    assert (
        chunks_evenement_2["index_evenement"]
        .nunique()
        == 1
    )