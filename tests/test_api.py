import requests


BASE_URL = "http://127.0.0.1:8000"


# ============================================================
# TEST 1 — Question normale
# ============================================================

def test_question_normale():
    response = requests.post(
        f"{BASE_URL}/ask",
        json={
            "question": "Quels concerts sont prévus à Vannes ?"
        },
        timeout=30,
    )

    print("\nTEST 1 — Question normale")
    print("Status :", response.status_code)
    print("Réponse :", response.json())


# ============================================================
# TEST 2 — Valeur vide
# ============================================================

def test_question_vide():
    response = requests.post(
        f"{BASE_URL}/ask",
        json={
            "question": ""
        },
        timeout=30,
    )

    print("\nTEST 2 — Question vide")
    print("Status :", response.status_code)
    print("Réponse :", response.json())

    assert response.status_code == 422


# ============================================================
# TEST 3 — Champ question absent
# ============================================================

def test_question_absente():
    response = requests.post(
        f"{BASE_URL}/ask",
        json={},
        timeout=30,
    )

    print("\nTEST 3 — Champ question absent")
    print("Status :", response.status_code)
    print("Réponse :", response.json())

    assert response.status_code == 422


# ============================================================
# TEST 4 — Requête incohérente
# ============================================================

def test_question_incoherente():
    response = requests.post(
        f"{BASE_URL}/ask",
        json={
            "question": "blablabla xzy 12345"
        },
        timeout=30,
    )

    print("\nTEST 4 — Question incohérente")
    print("Status :", response.status_code)
    print("Réponse :", response.json())


# ============================================================
# Exécution
# ============================================================

if __name__ == "__main__":
    test_question_normale()
    test_question_vide()
    test_question_absente()
    test_question_incoherente()