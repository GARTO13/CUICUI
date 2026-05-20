# Lovable API handoff

Ce document donne les informations à utiliser dans Lovable pour connecter le site à l'API `biosound-cluster`.

## URL De Base Actuelle

```text
https://minister-tulsa-tones-pennsylvania.trycloudflare.com
```

Health check :

```text
https://minister-tulsa-tones-pennsylvania.trycloudflare.com/health
```

Réponse attendue :

```json
{
  "status": "ok",
  "max_upload_mb": 5120,
  "max_chunk_mb": 50,
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

Pour les fichiers courts, le frontend peut utiliser `POST /api/jobs` directement.

Pour les fichiers longs ou lourds, par exemple 2-3 Go, le frontend doit utiliser l'upload par chunks. Ne pas essayer d'envoyer 2-3 Go en un seul `POST /api/jobs`, car les tunnels/proxys comme Cloudflare peuvent bloquer les requêtes autour de 100 Mo.

Le workflow recommandé est donc :

1. L'utilisateur sélectionne un fichier audio `.wav`.
2. L'utilisateur remplit les métadonnées :
   - identifiant du capteur ;
   - latitude ;
   - longitude ;
   - altitude optionnelle ;
   - type d'environnement ;
   - heure de début de l'enregistrement ;
   - timezone.
3. Si le fichier fait moins de 50 Mo, le frontend peut appeler `POST /api/jobs`.
4. Si le fichier fait plus de 50 Mo, le frontend doit :
   - appeler `POST /api/uploads/init` ;
   - découper le fichier en chunks de 25 à 50 Mo côté navigateur ;
   - envoyer chaque chunk à `POST /api/uploads/{upload_id}/chunks/{chunk_index}` ;
   - appeler `POST /api/uploads/{upload_id}/complete` avec les métadonnées.
5. L'API renvoie un `job_id`.
6. Le frontend poll `GET /api/jobs/{job_id}` jusqu'à `status: "done"` ou `status: "failed"`.
7. Quand le job est terminé, le frontend appelle `GET /api/jobs/{job_id}/result`.
8. Le frontend affiche :
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
https://minister-tulsa-tones-pennsylvania.trycloudflare.com/health
```

### 2. Créer Un Job

À utiliser seulement pour des fichiers raisonnablement petits.

```http
POST /api/jobs
Content-Type: multipart/form-data
```

URL complète :

```text
https://minister-tulsa-tones-pennsylvania.trycloudflare.com/api/jobs
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
  "status_url": "https://minister-tulsa-tones-pennsylvania.trycloudflare.com/api/jobs/9f734fa93f7743a7b90f814a6f3a6a35",
  "result_url": "https://minister-tulsa-tones-pennsylvania.trycloudflare.com/api/jobs/9f734fa93f7743a7b90f814a6f3a6a35/result"
}
```

### 2B. Upload Par Chunks Pour Gros Fichiers

À utiliser pour les fichiers de 2-3 Go.

#### Initialiser L'Upload

```http
POST /api/uploads/init
Content-Type: multipart/form-data
```

Champs :

```text
filename            nom du fichier .wav
total_size_bytes    taille totale du fichier en octets
```

Réponse :

```json
{
  "upload_id": "925651b9c12e4f0d84b37c19f8d9cf59",
  "max_chunk_mb": 50,
  "chunk_url_template": "https://minister-tulsa-tones-pennsylvania.trycloudflare.com/api/uploads/925651b9c12e4f0d84b37c19f8d9cf59/chunks/{chunk_index}",
  "complete_url": "https://minister-tulsa-tones-pennsylvania.trycloudflare.com/api/uploads/925651b9c12e4f0d84b37c19f8d9cf59/complete"
}
```

#### Envoyer Chaque Chunk

```http
POST /api/uploads/{upload_id}/chunks/{chunk_index}
Content-Type: multipart/form-data
```

Champs :

```text
file    blob/chunk du fichier audio
```

`chunk_index` commence à `0`.

Réponse :

```json
{
  "upload_id": "925651b9c12e4f0d84b37c19f8d9cf59",
  "chunk_index": 0,
  "size_bytes": 52428800
}
```

#### Finaliser Et Lancer Le Traitement

```http
POST /api/uploads/{upload_id}/complete
Content-Type: multipart/form-data
```

Envoyer les mêmes métadonnées que pour `POST /api/jobs` :

```text
sensor_id
sensor_latitude
sensor_longitude
sensor_elevation_m
environment_type
recording_start_time
recording_timezone
sample_rate
min_cluster_size
max_events
generate_spectrograms
enable_polyphony_handling
enable_clusterability_filtering
```

Réponse :

```json
{
  "job_id": "81876fce0f554c0c81b45f836653ad3a",
  "status": "queued",
  "status_url": "https://minister-tulsa-tones-pennsylvania.trycloudflare.com/api/jobs/81876fce0f554c0c81b45f836653ad3a",
  "result_url": "https://minister-tulsa-tones-pennsylvania.trycloudflare.com/api/jobs/81876fce0f554c0c81b45f836653ad3a/result"
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
https://minister-tulsa-tones-pennsylvania.trycloudflare.com

L'application doit permettre d'uploader un fichier audio .wav avec des métadonnées :
- sensor_id
- sensor_latitude
- sensor_longitude
- sensor_elevation_m
- environment_type
- recording_start_time
- recording_timezone

Pour les fichiers de moins de 50 Mo, créer un job avec POST /api/jobs en multipart/form-data.

Pour les gros fichiers, notamment 2-3 Go, ne pas utiliser un upload unique.
Utiliser l'upload par chunks :
1. POST /api/uploads/init avec filename et total_size_bytes.
2. Découper le fichier côté navigateur en chunks de 25 à 50 Mo.
3. Envoyer chaque chunk avec POST /api/uploads/{upload_id}/chunks/{chunk_index}.
4. Appeler POST /api/uploads/{upload_id}/complete avec les métadonnées.

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
