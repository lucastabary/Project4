# Génération musicale symbolique avec LSTM

Ce projet expérimente la génération de musique symbolique (MIDI) à l'aide de modèles LSTM entraînés sur le dataset MAESTRO.

**Résumé**
- **But :** Entraîner et évaluer des modèles LSTM pour générer des séquences MIDI tokenisées.
- **Données :** jeux MAESTRO (voir le dossier `datasets`).

**Détails techniques**

Ce dépôt est un projet de recherche personnel visant à explorer la génération musicale symbolique avec des modèles LSTM. Voici un résumé technique des évolutions mises en place et des choix d'implémentation.

- Tokenisation et représentation
	- Séquences représentées par des événements en quad : `PITCH`, `VELOCITY`, `DURATION`, `DELTA` (temps depuis l'événement précédent).
	- Trois tokens spéciaux : `PAD`, `BOS`, `EOS`.
	- `VELOCITY` : 16 buckets (pas fixe, indexés en multiples de 8).
	- `DURATION` et `DELTA` : buckets hybrides (petits pas linéaires puis log-espacés) pour couvrir une large gamme en ms.
	- Plages de `PITCH` : prototypes initiaux utilisaient 0–127 ; versions ultérieures (test4/test5) restreignent la plage utile à 21–108 (piano).
	- Fonctions de conversion : `process_midi_file(...)` et `write_midi_file(...)` (présentes dans `test1.py`..`test5.py`) utilisent `pretty_midi` pour convertir MIDI ⇄ tokens.

- Pipeline de données
	- `data_manager.find_all_midi_files(root_dir)` pour lister les fichiers MIDI (utilisé par `main.py`).
	- Plusieurs variantes de `MIDIDataset` ont été développées :
		- `test1.py` : version simple traitant un fichier à la fois et découpant en séquences aléatoires.
		- `test2.py`/`test3.py` : ajout d'API pour pré-traiter et sauvegarder les séquences (par fichier ou en un seul fichier `.pt`).
		- `test4.py`/`test5.py` : compatibilité avec chargement de datasets pré-traités (`load_processed`) et utilitaires `save_processed`, `count_tokens`.
	- Format de sauvegarde : objets torch sauvegardés en `.pt` dans `datasets/` (ex : `maestro-reduced.pt`) et fichiers individuels selon le prototype.

- Évolution des modèles
	- `test1.py` : `LSTM1` — 2 couches LSTM + embedding basique ; premières fonctions de génération et tests rapides.
	- `test2.py` : `LSTM` — ajout de dropout, meilleur entraînement (logging, sauvegarde par dossier `checkpoints/`), ajustements d'optimiseur (`AdamW`) et scheduler `ReduceLROnPlateau`.
	- `test3.py` : modèle étendu à 3 couches LSTM pour plus de capacité.
	- `test4.py` : API plus mature — ajout de `step()` (forward token‑par‑token) pour génération incrémentale et génération stochastique/validée sans reappliquer le réseau sur toute la séquence. Scheduler `OneCycleLR`, gestion améliorée des checkpoints (`trainings/`), logging dédié et entraînement avec `scheduler.step()` par itération.
	- `test5.py` (*WiP*) : prototype d'approche LoRA + TBPTT
		- Implémentation expérimentale `LoRA_LSTM_Layer` (adaptation Low-Rank pour couches LSTM) pour séparer entraînement de courte vs longue portée (base LSTM + LoRA pour dépendances longues).
		- Fonctions distinctes `launch_base_training` et placeholder `launch_lora_training` (travail en cours).

- Entraînement et optimisation
	- Optimiseur principal : `torch.optim.AdamW` (variantes de weight decay selon version).
	- Clipping de gradient (`clip_grad_norm_`) pour stabiliser l'entraînement.
	- Schedulers utilisés : `ReduceLROnPlateau` (versions initiales) → `OneCycleLR` (versions récentes) pour un entraînement plus agressif.
	- Options DataLoader : `num_workers`, `pin_memory`, etc., ajustées selon prototype.

- Génération et contraintes
	- `generate_valid_sequence(...)` : génération contrainte respectant l'ordre d'événements (pitch → velocity → duration → delta) pour produire des séquences valides.
	- `generate_stochastic_sequence(...)` : génération avec temperature + masquage des logits pour restreindre l'espace de tokens suivant le type attendu.
	- `quick_test()` : utilitaire pour exporter rapidement un MIDI de test dans `generated/`.

- Checkpoints et artefacts
	- Sauvegarde régulière de checkpoints pendant l'entraînement : `checkpoints/` et `trainings/` (naming convention inclut le nom du modèle et l'époque).

- Outils et dépendances utilisés (détails dans `requirements.txt`)
	- `torch`, `pretty_midi`, `numpy`, `matplotlib` (utilisé dans `main.py` pour l'analyse d'embeddings), plus utilitaires standards (`logging`).
