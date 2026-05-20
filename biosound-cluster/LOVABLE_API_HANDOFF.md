# Lovable API handoff

Ce document donne les informations à utiliser dans Lovable pour connecter le site à l'API `biosound-cluster`.

## URL De Base Actuelle

```text
https://alpha-admission-distances-pest.trycloudflare.com
```

Health check :

```text
https://alpha-admission-distances-pest.trycloudflare.com/health
```

Réponse attendue :

```json
{
  "status": "ok",
  "max_upload_mb": 2048,
  "workers": 1,
  "api_root": "outputs/api_jobs"
}
```

Important : cette URL Cloudflare est temporaire. Elle reste active tant que le tunnel `cloudflared` tourne sur l'ordinateur qui héberge l'API.

## Authentification

Actuellement :

```text
Pas d'authentification applicative.
```

Pour un prototype Lovable, appeler directement l'URL Cloudflare.

Pour une mise en production, ajouter une authentification, des quotas et un domaine CORS strict.

## Workflow Frontend

Le frontend doit suivre ce flux :

1. L'utilisateur sélectionne un fichier audio `.wav`.
2. L'utilisateur remplit les métadonnées :
   - identifiant du capteur ;
   - latitude ;
   - longitude ;
   - altitude optionnelle ;
   - type d'environnement ;
   - heure de début de l'enregistrement ;
   - timezone.
3. Le frontend envoie un `POST /api/jobs` en `multipart/form-data`.
4. L'API renvoie un `job_id`.
5. Le frontend poll `GET /api/jobs/{job_id}` jusqu'à `status: "done"` ou `status: "failed"`.
6. Quand le job est terminé, le frontend appelle `GET /api/jobs/{job_id}/result`.
7. Le frontend affiche :
   - les clusters ;
   - les audios associés ;
   - les spectrogrammes ;
   - les métadonnées ;
   - les dossiers de review : mixed, low confidence, ambiguous, short review.

## Endpoints À Utiliser

### 1. Vérifier Que L'API Répond

```http
GET /health
```

URL complète :

```text
https://alpha-admission-distances-pest.trycloudflare.com/health
```

### 2. Créer Un Job

```http
POST /api/jobs
Content-Type: multipart/form-data
```

URL complète :

```text
https://alpha-admission-distances-pest.trycloudflare.com/api/jobs
```

Champs `multipart/form-data` :

```text
file                              fichier .wav obligatoire
sensor_id                         string optionnel
sensor_latitude                   number optionnel
sensor_longitude                  number optionnel
sensor_elevation_m                number optionnel
environment_type                  string optionnel
recording_start_time              string optionnel, idéalement ISO-8601
recording_timezone                string optionnel
sample_rate                       number, défaut 32000
min_cluster_size                  number, défaut 10
max_events                        number optionnel
generate_spectrograms             boolean, défaut true
enable_polyphony_handling         boolean, défaut true
enable_clusterability_filtering   boolean, défaut true
```

Exemple de réponse :

```json
{
  "job_id": "9f734fa93f7743a7b90f814a6f3a6a35",
  "status": "queued",
  "status_url": "https://alpha-admission-distances-pest.trycloudflare.com/api/jobs/9f734fa93f7743a7b90f814a6f3a6a35",
  "result_url": "https://alpha-admission-distances-pest.trycloudflare.com/api/jobs/9f734fa93f7743a7b90f814a6f3a6a35/result"
}
```

### 3. Lire Le Statut D'Un Job

```http
GET /api/jobs/{job_id}
```

États possibles :

```text
queued
running
done
failed
```

Si `failed`, afficher le champ `error`.

### 4. Récupérer Le Résultat Complet

```http
GET /api/jobs/{job_id}/result
```

Ce endpoint retourne :

```text
job
run_metadata
event_metadata_url
events_csv_url
clusters_csv_url
report_url
index_url
clusters
review_folders
events
```

### 5. Récupérer Seulement Les Clusters

```http
GET /api/jobs/{job_id}/clusters
```

Retourne :

```text
clusters
review_folders
```

Chaque cluster contient :

```text
cluster_id
size
folder_name
representative_event_ids
events
representatives
```

Chaque événement contient notamment :

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
clip_path_url
spectrogram_path_url
context_clip_path_url
metadata_url
metadata
```

### 6. Récupérer Tous Les Événements

```http
GET /api/jobs/{job_id}/events
```

Retourne la liste complète des événements, y compris ceux routés hors clusters normaux.

### 7. Lire Un Fichier Exporté

```http
GET /api/jobs/{job_id}/files/{relative_output_path}
```

Cet endpoint sert :

```text
.wav
.png
.json
.csv
.md
.html
```

Dans Lovable, utiliser directement les champs déjà fournis :

```text
clip_path_url
spectrogram_path_url
context_clip_path_url
metadata_url
```

## Prompt Court À Donner À Lovable

```text
Connecte le frontend à cette API FastAPI :

Base URL:
https://alpha-admission-distances-pest.trycloudflare.com

L'application doit permettre d'uploader un fichier audio .wav avec des métadonnées :
- sensor_id
- sensor_latitude
- sensor_longitude
- sensor_elevation_m
- environment_type
- recording_start_time
- recording_timezone

Créer un job avec POST /api/jobs en multipart/form-data.
Après la réponse, stocker job_id, puis poller GET /api/jobs/{job_id} toutes les 3 à 5 secondes.
Quand status vaut "done", appeler GET /api/jobs/{job_id}/result.

Afficher ensuite :
- résumé du run ;
- liste des clusters ;
- représentants de chaque cluster avec lecteur audio HTML ;
- spectrogrammes PNG ;
- métadonnées de chaque clip ;
- dossiers review_folders : mixed, low_confidence_noise, ambiguous_review, short_review.

Ne jamais afficher les clusters comme des espèces. Les appeler "familles acoustiques".
Les labels biologiques doivent rester une validation humaine.
```

## Notes Produit Importantes

Ne pas écrire :

```text
espèce prédite
classification automatique
animal identifié
```

Écrire plutôt :

```text
famille acoustique
cluster acoustique
segment candidat
son à valider
annotation humaine
```

## Commandes Actuellement Utilisées Côté Backend

Terminal 1 :

```bash
cd /Users/thomas/CUICUI/biosound-cluster
source .venv/bin/activate
uvicorn main:app --host 0.0.0.0 --port 8000
```

Terminal 2 :

```bash
cloudflared tunnel --url http://127.0.0.1:8000
```

Si l'ordinateur s'éteint, si le terminal est fermé, ou si `cloudflared` est arrêté, l'URL publique ne fonctionnera plus.
