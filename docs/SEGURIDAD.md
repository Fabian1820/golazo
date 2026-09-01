# Credencial expuesta en la historia de git

## Qué pasó

`src/kaggle.json` —un token de la API de Kaggle, con `username` y `key`— se
commiteó en `51b0e08` («first commit») y se retiró del árbol de trabajo en
`d533479` («Sacar del repositorio el entorno virtual, la caché y las
credenciales»).

Retirarlo del árbol **no lo retira de la historia**. El fichero sigue siendo
recuperable con `git show 51b0e08:src/kaggle.json` por cualquiera que clone el
repositorio, que es público.

## Qué hacer, en este orden

### 1. Revocar el token — esto es lo único que de verdad importa

En <https://www.kaggle.com/settings> → *API* → **Expire API Token**, y generar
uno nuevo si hace falta.

**Hazlo aunque no reescribas la historia.** Reescribir no des-filtra nada:

- GitHub conserva los objetos alcanzables por caché durante un tiempo, y un
  commit borrado sigue accesible por su SHA si alguien lo conoce;
- puede haber clones, forks o réplicas fuera de tu control;
- los repositorios públicos se indexan de forma continua por *scrapers* que
  buscan precisamente credenciales. Un token que ha estado expuesto se
  considera comprometido, punto.

La rotación es obligatoria y suficiente. La reescritura es higiene.

### 2. Reescribir la historia

`scripts/purge_history.sh` prepara el trabajo. Purga de todos los commits:

| qué | por qué | peso |
|---|---|---|
| `src/kaggle.json` | credencial | — |
| `venv/` | entorno virtual commiteado (13.254 de los 13.273 ficheros que han existido en el repo) | ~93 MB |
| `src/soccer/*.csv` | volcados originales, sustituidos por `data/matches.csv` | ~78 MB |

El repositorio pasa de **156 MB a unos 5 MB**.

El script **no hace push**. Deja la historia reescrita en local para que la
revises antes de publicarla.

### 3. Publicar

```bash
git push --force-with-lease origin main
```

Esto reescribe la rama pública. Los SHA de todos los commits cambian, así que
cualquier clon existente queda desincronizado y hay que volver a clonarlo. Con
4 commits, un autor y una rama, el coste es mínimo — pero es irreversible.

### 4. Pedir a GitHub que purgue la caché

Tras el push, abre un ticket en <https://support.github.com/> pidiendo la
eliminación de los objetos huérfanos. Sin eso, los commits antiguos siguen
siendo accesibles por SHA a través de la web durante un tiempo.

## Cómo se evita en adelante

- `.gitignore` bloquea `kaggle.json`, `*.key`, `.env` y `venv/`.
- La documentación indica guardar el token en `~/.kaggle/kaggle.json`, **fuera
  del repositorio**.
- Ninguna parte del código necesita credenciales: la fuente viva (Understat) es
  pública y no requiere autenticación.

## Comprobar si sigue expuesto

```bash
git log --all --pretty=format: --name-only --diff-filter=A | sort -u | grep -iE 'kaggle|\.env$|\.key$|secret|token'
```

Sin salida, la historia está limpia.
