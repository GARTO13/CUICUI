# API biosound-cluster

Ce document décrit l'API `biosound-cluster` exposée pour une interface web, par exemple Lovable.

L'API permet de :

- uploader un fichier audio `.wav` avec ses métadonnées de terrain ;
- lancer le pipeline de détection, séparation, clustering et export en tâche asynchrone ;
- suivre l'état du traitement ;
- récupérer les clusters, les événements audio, les spectrogrammes et les métadonnées.

Important : l'API ne classifie pas les espèces. Les clusters retournés sont des familles acoustiques non supervisées, à valider par un humain.

## URL De Base

En local :

```text
http://127.0.0.1:8000
```

En déploiement, remplacer par l'URL publique du serveur :

```text
https://<votre-domaine-api>
```

Toutes les routes ci-dessous sont relatives à cette URL de base.

## Lancer L'API Pour Lovable

Depuis le dossier backend :

```bash
cd /Users/thomas/CUICUI/biosound-cluster
```

Activer l'environnement virtuel :

```bash
source .venv/bin/activate
```

Installer ou mettre à jour le projet :

```bash
pip install -e .
```

Lancer FastAPI avec `uvicorn` :

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

Alternative équivalente propre au package :

```bash
biosound-api
```

Pour que Lovable puisse appeler l'API depuis internet, ouvrir un second terminal et créer un tunnel HTTPS.

Avec Cloudflare Tunnel :

```bash
cloudflared tunnel --url http://127.0.0.1:8000
```

Avec ngrok :

```bash
ngrok http 8000
```

Le tunnel renvoie une URL publique du type :

```text
https://xxxxx.trycloudflare.com
```

Dans Lovable, utiliser cette URL comme URL de base de l'API.

Exemple :

```text
https://xxxxx.trycloudflare.com/health
https://xxxxx.trycloudflare.com/api/jobs
```

Note : le fichier `main.py` à la racine du projet existe uniquement pour rendre la commande
`uvicorn main:app` compatible avec les guides de déploiement. Le code réel de l'API reste dans
`src/biosound_cluster/api.py`.

## Authentification

Version actuelle :

```text
Aucune authentification applicative n'est encore implémentée dans le code.
```

Conséquence :

- les endpoints sont accessibles directement si le serveur est exposé ;
- en production, il faut protéger l'API au niveau infrastructure ou ajouter une authentification avant exposition publique.

Recommandation de déploiement :

- mettre l'API derrière un reverse proxy ;
- limiter CORS au domaine du frontend Lovable ;
- ajouter une clé API ou un JWT avant un usage public ;
- limiter la taille maximale d'upload ;
- limiter le nombre de workers pour éviter les traitements parallèles trop lourds.

Variables utiles :

```bash
BIOSOUND_API_HOST=127.0.0.1
BIOSOUND_API_PORT=8000
BIOSOUND_API_ROOT=outputs/api_jobs
BIOSOUND_API_MAX_UPLOAD_MB=2048
BIOSOUND_API_WORKERS=1
BIOSOUND_API_CORS_ORIGINS="*"
```

## Format D'Erreur

Les erreurs FastAPI sont retournées en JSON :

```json
{
  "detail": "Message d'erreur"
}
```

Codes fréquents :

```text
400  fichier invalide, fichier vide, ou fichier non-WAV
404  job ou fichier introuvable
409  job pas encore terminé
413  fichier trop volumineux
500  erreur interne pendant le traitement
```

## 1. Health Check

Vérifie que l'API répond.

```http
GET /health
```

### Réponse

```json
{
  "status": "ok",
  "max_upload_mb": 2048,
  "workers": 1,
  "api_root": "outputs/api_jobs"
}
```

## 2. Créer Un Job D'Analyse Audio

Upload un fichier `.wav`, enregistre les métadonnées, puis lance le traitement en tâche de fond.

```http
POST /api/jobs
Content-Type: multipart/form-data
```

### Corps Multipart

Champs fichier :

| Champ | Type | Obligatoire | Description |
|---|---:|---:|---|
| `file` | file | oui | Fichier audio `.wav`. Le header RIFF/WAVE est vérifié. |

Métadonnées de terrain :

| Champ | Type | Obligatoire | Description |
|---|---:|---:|---|
| `sensor_id` | string | non | Identifiant du capteur ou du déploiement. |
| `sensor_latitude` | float | non | Latitude du capteur, entre -90 et 90. |
| `sensor_longitude` | float | non | Longitude du capteur, entre -180 et 180. |
| `sensor_elevation_m` | float | non | Altitude du capteur en mètres. |
| `environment_type` | string | non | Type d'environnement, par exemple `tropical_forest`. |
| `recording_start_time` | string | non | Heure de début de l'enregistrement, idéalement ISO-8601. |
| `recording_timezone` | string | non | Fuseau horaire textuel, par exemple `Europe/Paris`. |

Options de traitement :

| Champ | Type | Défaut | Description |
|---|---:|---:|---|
| `sample_rate` | int | `32000` | Fréquence d'échantillonnage cible. |
| `min_cluster_size` | int | `10` | Taille minimale HDBSCAN des clusters. |
| `max_events` | int ou null | `null` | Limite optionnelle du nombre d'événements, utile pour debug. |
| `generate_spectrograms` | bool | `true` | Génère les images `.png`. |
| `enable_polyphony_handling` | bool | `true` | Active la gestion des sons superposés. |
| `enable_clusterability_filtering` | bool | `true` | Active le routage des événements ambigus hors clustering normal. |

### Exemple Curl

```bash
curl -X POST http://127.0.0.1:8000/api/jobs \
  -F "file=@path/to/recording.wav" \
  -F "sensor_id=GUYANE_001" \
  -F "sensor_latitude=4.9372" \
  -F "sensor_longitude=-52.3260" \
  -F "sensor_elevation_m=18.5" \
  -F "environment_type=tropical_forest" \
  -F "recording_start_time=2026-05-20T06:30:00+02:00" \
  -F "recording_timezone=Europe/Paris" \
  -F "min_cluster_size=10"
```

### Réponse `202 Accepted`

```json
{
  "job_id": "9f734fa93f7743a7b90f814a6f3a6a35",
  "status": "queued",
  "status_url": "http://127.0.0.1:8000/api/jobs/9f734fa93f7743a7b90f814a6f3a6a35",
  "result_url": "http://127.0.0.1:8000/api/jobs/9f734fa93f7743a7b90f814a6f3a6a35/result"
}
```

Le frontend doit ensuite poller `status_url` jusqu'à obtenir `status: "done"` ou `status: "failed"`.

## 3. Lire Le Statut D'Un Job

```http
GET /api/jobs/{job_id}
```

### Paramètres

| Paramètre | Type | Description |
|---|---:|---|
| `job_id` | string | Identifiant retourné par `POST /api/jobs`. |

### Réponse En Attente

```json
{
  "job_id": "9f734fa93f7743a7b90f814a6f3a6a35",
  "status": "running",
  "created_at": "2026-05-20T13:05:01.000000+00:00",
  "updated_at": "2026-05-20T13:05:10.000000+00:00",
  "input_path": "outputs/api_jobs/9f734.../upload/recording.wav",
  "output_dir": "outputs/api_jobs/9f734.../run",
  "error": null,
  "result": null,
  "metadata": {
    "request": {
      "sensor_id": "GUYANE_001",
      "sensor_latitude": 4.9372,
      "sensor_longitude": -52.326,
      "sensor_elevation_m": 18.5,
      "environment_type": "tropical_forest",
      "recording_start_time": "2026-05-20T06:30:00+02:00",
      "recording_timezone": "Europe/Paris",
      "saved_size_bytes": 123456789,
      "original_filename": "recording.wav"
    },
    "config": {
      "sample_rate": 32000,
      "min_cluster_size": 10,
      "enable_polyphony_handling": true,
      "enable_clusterability_filtering": true
    }
  }
}
```

### Réponse Terminée

```json
{
  "job_id": "9f734fa93f7743a7b90f814a6f3a6a35",
  "status": "done",
  "created_at": "2026-05-20T13:05:01.000000+00:00",
  "updated_at": "2026-05-20T13:09:41.000000+00:00",
  "input_path": "outputs/api_jobs/9f734.../upload/recording.wav",
  "output_dir": "outputs/api_jobs/9f734.../run",
  "error": null,
  "result": {
    "input_path": "outputs/api_jobs/9f734.../upload/recording.wav",
    "output_dir": "outputs/api_jobs/9f734.../run",
    "duration_sec": 1800.0,
    "sample_rate": 32000,
    "n_events": 245,
    "n_clusters": 12,
    "n_noise": 38,
    "events_csv": "outputs/api_jobs/9f734.../run/events.csv",
    "clusters_csv": "outputs/api_jobs/9f734.../run/clusters.csv",
    "report_md": "outputs/api_jobs/9f734.../run/report.md",
    "index_html": "outputs/api_jobs/9f734.../run/index.html"
  },
  "metadata": {}
}
```

### Réponse Échouée

```json
{
  "job_id": "9f734fa93f7743a7b90f814a6f3a6a35",
  "status": "failed",
  "error": "Message d'erreur Python ou pipeline",
  "result": null
}
```

## 4. Récupérer Le Résultat Complet

Retourne le job, les métadonnées globales, les URLs des fichiers principaux, les clusters, les dossiers de review et tous les événements.

```http
GET /api/jobs/{job_id}/result
```

### Comportement

Si le job n'est pas terminé, l'endpoint retourne simplement l'objet job courant.

Si le job est terminé, l'endpoint retourne le payload complet.

### Réponse Terminée

```json
{
  "job": {
    "job_id": "9f734fa93f7743a7b90f814a6f3a6a35",
    "status": "done",
    "created_at": "2026-05-20T13:05:01.000000+00:00",
    "updated_at": "2026-05-20T13:09:41.000000+00:00",
    "input_path": "outputs/api_jobs/9f734.../upload/recording.wav",
    "output_dir": "outputs/api_jobs/9f734.../run",
    "error": null,
    "result": {
      "duration_sec": 1800.0,
      "n_events": 245,
      "n_clusters": 12,
      "n_noise": 38
    },
    "metadata": {}
  },
  "run_metadata": {
    "input_path": "outputs/api_jobs/9f734.../upload/recording.wav",
    "duration_sec": 1800.0,
    "sample_rate": 32000,
    "recording_metadata": {
      "sensor_id": "GUYANE_001",
      "sensor_latitude": 4.9372,
      "sensor_longitude": -52.326,
      "sensor_elevation_m": 18.5,
      "environment_type": "tropical_forest",
      "recording_start_time": "2026-05-20T06:30:00+02:00",
      "recording_timezone": "Europe/Paris"
    }
  },
  "event_metadata_url": "http://127.0.0.1:8000/api/jobs/9f734.../files/event_metadata.json",
  "events_csv_url": "http://127.0.0.1:8000/api/jobs/9f734.../files/events.csv",
  "clusters_csv_url": "http://127.0.0.1:8000/api/jobs/9f734.../files/clusters.csv",
  "report_url": "http://127.0.0.1:8000/api/jobs/9f734.../files/report.md",
  "index_url": "http://127.0.0.1:8000/api/jobs/9f734.../files/index.html",
  "clusters": [],
  "review_folders": {
    "mixed": [],
    "low_confidence_noise": [],
    "ambiguous_review": [],
    "short_review": []
  },
  "events": []
}
```

## 5. Récupérer Les Clusters

Retourne uniquement les clusters normaux et les dossiers de review.

```http
GET /api/jobs/{job_id}/clusters
```

### Réponse

```json
{
  "job_id": "9f734fa93f7743a7b90f814a6f3a6a35",
  "clusters": [
    {
      "cluster_id": 0,
      "size": 42,
      "folder_name": "cluster_000_size_042",
      "mean_probability": 0.83,
      "representative_event_ids": "event_000012,event_000034",
      "mean_purity_score": 0.72,
      "n_component_events": 4,
      "n_original_events": 38,
      "mean_clusterability_score": 0.69,
      "mean_stability_score": 0.81,
      "acoustic_prefamily": "tonal_whistle",
      "events": [
        {
          "event_id": "event_000012",
          "start_sec": 12.34,
          "end_sec": 13.82,
          "duration_sec": 1.48,
          "cluster_id": 0,
          "is_noise": false,
          "source_type": "original",
          "clip_path": "cluster_000_size_042/event_000012__12.340-13.820.wav",
          "clip_path_url": "http://127.0.0.1:8000/api/jobs/9f734.../files/cluster_000_size_042/event_000012__12.340-13.820.wav",
          "spectrogram_path": "cluster_000_size_042/event_000012__12.340-13.820.png",
          "spectrogram_path_url": "http://127.0.0.1:8000/api/jobs/9f734.../files/cluster_000_size_042/event_000012__12.340-13.820.png",
          "metadata_url": "http://127.0.0.1:8000/api/jobs/9f734.../files/cluster_000_size_042/event_000012__12.340-13.820.json",
          "metadata": {
            "event_id": "event_000012",
            "recording_start_time": "2026-05-20T06:30:00+02:00",
            "clip_start_sec": 12.34,
            "clip_end_sec": 13.82,
            "absolute_start_time": "2026-05-20T06:30:12.340000+02:00",
            "absolute_end_time": "2026-05-20T06:30:13.820000+02:00"
          }
        }
      ],
      "representatives": [
        {
          "event_id": "event_000012",
          "representative_score": 0.88,
          "clip_path_url": "http://127.0.0.1:8000/api/jobs/9f734.../files/cluster_000_size_042/event_000012__12.340-13.820.wav",
          "spectrogram_path_url": "http://127.0.0.1:8000/api/jobs/9f734.../files/cluster_000_size_042/event_000012__12.340-13.820.png"
        }
      ]
    }
  ],
  "review_folders": {
    "mixed": [],
    "low_confidence_noise": [],
    "ambiguous_review": [],
    "short_review": []
  }
}
```

### Erreur Si Le Job N'Est Pas Terminé

```json
{
  "detail": "Job is running, not done."
}
```

Code HTTP :

```text
409 Conflict
```

## 6. Récupérer Tous Les Événements

Retourne la liste complète des événements exportés, y compris les événements clusterisés et les événements routés en review.

```http
GET /api/jobs/{job_id}/events
```

### Réponse

```json
{
  "job_id": "9f734fa93f7743a7b90f814a6f3a6a35",
  "events": [
    {
      "event_id": "event_000123",
      "start_sec": 45.1,
      "end_sec": 46.2,
      "duration_sec": 1.1,
      "cluster_id": 1,
      "is_noise": false,
      "cluster_probability": 0.91,
      "rms_db": -28.4,
      "peak_db": -10.2,
      "spectral_centroid": 3520.7,
      "source_type": "component",
      "parent_event_id": "event_000123",
      "component_id": 0,
      "is_component": true,
      "is_overlapping": true,
      "is_mixed": false,
      "n_components": 2,
      "polyphony_score": 0.61,
      "purity_score": 0.73,
      "clusterability_score": 0.78,
      "embedding_stability_score": 0.82,
      "representative_score": 0.86,
      "clip_path": "cluster_001_size_018/event_000123_component_0__45.100-46.200.wav",
      "clip_path_url": "http://127.0.0.1:8000/api/jobs/9f734.../files/cluster_001_size_018/event_000123_component_0__45.100-46.200.wav",
      "spectrogram_path": "cluster_001_size_018/event_000123_component_0__45.100-46.200.png",
      "spectrogram_path_url": "http://127.0.0.1:8000/api/jobs/9f734.../files/cluster_001_size_018/event_000123_component_0__45.100-46.200.png",
      "context_clip_path": "cluster_001_size_018/event_000123_component_0__context_original.wav",
      "context_clip_path_url": "http://127.0.0.1:8000/api/jobs/9f734.../files/cluster_001_size_018/event_000123_component_0__context_original.wav",
      "metadata_url": "http://127.0.0.1:8000/api/jobs/9f734.../files/cluster_001_size_018/event_000123_component_0__45.100-46.200.json",
      "metadata": {
        "event_id": "event_000123",
        "clip_start_sec": 45.1,
        "clip_end_sec": 46.2,
        "sensor_id": "GUYANE_001",
        "environment_type": "tropical_forest"
      }
    }
  ]
}
```

## 7. Télécharger Ou Lire Un Fichier Exporté

Sert un fichier généré par le pipeline : audio `.wav`, spectrogramme `.png`, JSON sidecar, CSV, rapport Markdown ou index HTML.

```http
GET /api/jobs/{job_id}/files/{relative_output_path}
```

### Paramètres

| Paramètre | Type | Description |
|---|---:|---|
| `job_id` | string | Identifiant du job. |
| `relative_output_path` | string | Chemin relatif dans le dossier de sortie du job. |

### Exemples

```http
GET /api/jobs/9f734.../files/events.csv
GET /api/jobs/9f734.../files/clusters.csv
GET /api/jobs/9f734.../files/report.md
GET /api/jobs/9f734.../files/index.html
GET /api/jobs/9f734.../files/cluster_000_size_042/event_000012__12.340-13.820.wav
GET /api/jobs/9f734.../files/cluster_000_size_042/event_000012__12.340-13.820.png
GET /api/jobs/9f734.../files/cluster_000_size_042/event_000012__12.340-13.820.json
```

### Réponse

La réponse est le fichier demandé avec le type MIME inféré par FastAPI/Starlette.

Sécurité :

- le chemin est résolu côté serveur ;
- l'API refuse tout accès hors du dossier de sortie du job ;
- un chemin de type `../../secret.txt` retourne `404`.

## Flux Recommandé Côté Lovable

1. L'utilisateur choisit un fichier `.wav`.
2. Le frontend collecte les métadonnées : capteur, coordonnées, environnement, heure de début.
3. Le frontend appelle `POST /api/jobs`.
4. Le frontend stocke `job_id`.
5. Le frontend poll `GET /api/jobs/{job_id}` toutes les quelques secondes.
6. Quand `status` vaut `done`, le frontend appelle `GET /api/jobs/{job_id}/result`.
7. Le frontend affiche :
   - résumé du run ;
   - clusters normaux ;
   - représentants par cluster ;
   - lecteurs audio via `clip_path_url` ;
   - spectrogrammes via `spectrogram_path_url` ;
   - dossiers de review : mixed, low confidence, ambiguous, short review.
8. Si `status` vaut `failed`, afficher `error`.

## Champs Importants Pour L'Interface

Pour un affichage produit, les champs les plus utiles sont :

```text
event_id
start_sec
end_sec
duration_sec
cluster_id
source_type
is_noise
is_mixed
is_component
parent_event_id
component_id
cluster_probability
clusterability_score
embedding_stability_score
representative_score
clip_path_url
spectrogram_path_url
context_clip_path_url
metadata_url
metadata.recording_start_time
metadata.absolute_start_time
metadata.absolute_end_time
metadata.sensor_id
metadata.sensor_latitude
metadata.sensor_longitude
metadata.environment_type
```

## Notes De Sécurité Pour Les Longs Audios

L'API actuelle applique déjà :

- upload streamé par chunks de 1 MB ;
- limite de taille via `BIOSOUND_API_MAX_UPLOAD_MB` ;
- vérification extension `.wav` ;
- vérification header `RIFF/WAVE` ;
- un pool de workers borné ;
- écriture des jobs dans `BIOSOUND_API_ROOT` ;
- protection contre la lecture de fichiers hors du dossier du job.

Pour un vrai déploiement public, ajouter aussi :

- authentification ;
- quota par utilisateur ;
- rate limiting ;
- antivirus ou scan fichier si exposition publique ;
- stockage objet si les sorties deviennent volumineuses ;
- nettoyage automatique des anciens jobs ;
- monitoring disque, RAM et CPU ;
- domaine CORS précis au lieu de `*`.
