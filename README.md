# Data-analyse-gamma-beta


## Docker deployment

Er zijn nu bestanden toegevoegd om de Streamlit-app eenvoudig met Docker te draaien.

Build en run (lokale server):

```bash
# image bouwen
docker build -t gamma-beta-app .

# container run
docker run --rm -p 8501:8501 gamma-beta-app
```

Met docker-compose:

```bash
docker compose up --build
```

De app komt beschikbaar op http://localhost:8501. Voor productie-omgevingen zet je een reverse proxy (Nginx/Caddy) voor TLS en eventueel toegangsbescherming.

Startscript voor ontwikkelaars: zie `start.sh` (maak uitvoerbaar met `chmod +x start.sh`).

