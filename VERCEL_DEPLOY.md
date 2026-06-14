# Despliegue en Vercel

## Archivos incluidos

- `app.py`
- `predictor_mundial2026.py`
- `data/mundial2026_ML_dataset.xlsx`
- `templates/`
- `public/`
- `requirements.txt`
- `.python-version`

El Excel forma parte del despliegue como un archivo de datos de solo lectura. Para
actualizar la información, reemplaza el archivo en `data/`, crea un commit y vuelve
a desplegar.

## GitHub

Este workspace tiene actualmente un remoto llamado `origin` que apunta a
`JECK-dev/clear_finances`. No publiques esta aplicación en ese remoto.

La opción recomendada es crear un repositorio nuevo y limpio para el predictor.
Incluye solamente los archivos enumerados en la sección anterior, además de
`.gitignore`, `.vercelignore` y `VERCEL_DEPLOY.md`.

```bash
git add app.py predictor_mundial2026.py data templates public requirements.txt .python-version .vercelignore VERCEL_DEPLOY.md
git commit -m "Prepare Mundial 2026 predictor for Vercel"
git remote add origin https://github.com/USUARIO/predictor-mundial-2026.git
git push -u origin main
```

## Vercel

1. Importa el repositorio desde GitHub.
2. Usa el directorio raíz del repositorio.
3. Selecciona Flask como Framework Preset si Vercel lo solicita.
4. No configures Build Command ni Output Directory.
5. Despliega.

La aplicación detecta automáticamente:

- Predictor: `predictor_mundial2026.py`
- Dataset: `data/mundial2026_ML_dataset.xlsx`

Las variables `MUNDIAL_PREDICTOR_PATH` y `MUNDIAL_DATASET_PATH` quedan disponibles
como alternativas para otros entornos, pero no son necesarias en Vercel.
