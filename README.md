# mlops-ml-project - TP2 MLOps (Iris)

Mini-projet de classification Iris en trois briques reutilisables : `src/data.py`, `src/features.py`, `src/model.py`.

## Installation (Windows PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Si Python est gere par uv, on peut remplacer la derniere ligne par `uv pip install --python .venv\Scripts\python.exe -r requirements.txt`.

## Execution

```powershell
.venv\Scripts\python.exe scripts/train.py
.venv\Scripts\python.exe scripts/evaluate.py
.venv\Scripts\python.exe -m pytest -q
```

## Artefacts non versions

- `artifacts/model.joblib` : pipeline ML serialize
- `artifacts/metrics.json` : accuracy et F1 macro sur les 30 exemples de test
- `artifacts/confusion_matrix.png` : matrice de confusion
- `artifacts/report.json` : rapport de l'evaluation sur tout Iris (cas volontairement biaisé de la question Q2)

## Git

`main` = baseline initiale, `dev` = integration (clipping, fusion max_iter), `feature/preprocessing` = ajout du clipping, `feature/max-iter` = variante d'hyperparametre, `feature/param-C` = configuration C=10 pour la Pull Request.

Les donnees et artefacts, ainsi que `.venv`, ne sont pas commités. Les branches et commits sont inclus dans le ZIP, mais le depot GitHub, la PR, la review et la release restent a publier.

## Attention

Ne pas se fier au score de `evaluate.py`, car il evalue les 150 fleurs incluant les exemples d'entrainement. Pour une evaluation honnête il faut reconstituer exactement le jeu de test avec les memes parametres de split ou le sauvegarder.
