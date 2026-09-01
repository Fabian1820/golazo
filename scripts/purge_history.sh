#!/usr/bin/env bash
#
# Reescribe la historia de git para eliminar la credencial filtrada, el entorno
# virtual commiteado y los volcados de datos originales.
#
#   bash scripts/purge_history.sh
#
# NO HACE PUSH. Deja la historia reescrita en local para que la revises.
# Antes de ejecutarlo, lee docs/SEGURIDAD.md — y revoca el token de Kaggle,
# que es lo único que de verdad cierra la exposición.
#
set -euo pipefail

ROJO=$'\033[31m'; VERDE=$'\033[32m'; AMARILLO=$'\033[33m'; FIN=$'\033[0m'
info()  { echo "${VERDE}==>${FIN} $*"; }
aviso() { echo "${AMARILLO}!!!${FIN} $*"; }
error() { echo "${ROJO}ERROR:${FIN} $*" >&2; exit 1; }

cd "$(dirname "$0")/.."
RAIZ=$(pwd)

# --- comprobaciones previas -------------------------------------------------

[[ -d .git ]] || error "No estás en la raíz de un repositorio git."

if [[ -n "$(git status --porcelain)" ]]; then
    error "Hay cambios sin commitear. Haz commit o guárdalos antes de reescribir la historia."
fi

if ! command -v git-filter-repo >/dev/null 2>&1; then
    cat <<'EOF'
ERROR: falta git-filter-repo.

  pip install git-filter-repo        (o: brew install git-filter-repo)

Es la herramienta que recomienda el propio GitHub. `git filter-branch` también
funcionaría, pero es mucho más lento y tiene modos de fallo desagradables.
EOF
    exit 1
fi

# --- copia de seguridad -----------------------------------------------------

RESPALDO="${RAIZ}/../golazo-respaldo-$(date +%Y%m%d-%H%M%S)"
info "Copia de seguridad completa en:"
echo "    ${RESPALDO}"
cp -R "${RAIZ}" "${RESPALDO}"
[[ -d "${RESPALDO}/.git" ]] || error "La copia de seguridad falló. Se aborta."

# --- estado previo ----------------------------------------------------------

ANTES=$(du -sh .git | cut -f1)
info "Tamaño de .git antes: ${ANTES}"

# --- reescritura ------------------------------------------------------------

info "Purgando de TODOS los commits:"
echo "    src/kaggle.json      (credencial filtrada)"
echo "    venv/                (entorno virtual commiteado, ~93 MB)"
echo "    src/soccer/          (volcados originales, ~78 MB)"
echo

git filter-repo --force \
    --invert-paths \
    --path src/kaggle.json \
    --path venv \
    --path src/soccer

# --- verificación -----------------------------------------------------------

info "Verificando que no quedan rastros..."
RESTOS=$(git log --all --pretty=format: --name-only --diff-filter=A \
         | sort -u | grep -iE 'kaggle\.json|^venv/|^src/soccer/' || true)

if [[ -n "${RESTOS}" ]]; then
    aviso "Todavía aparecen estos ficheros en la historia:"
    echo "${RESTOS}"
    error "La purga no fue completa. La copia de seguridad sigue en ${RESPALDO}"
fi

git reflog expire --expire=now --all
git gc --prune=now --aggressive >/dev/null 2>&1

DESPUES=$(du -sh .git | cut -f1)

# --- resumen ----------------------------------------------------------------

cat <<EOF

${VERDE}Historia reescrita.${FIN}

    .git: ${ANTES}  ->  ${DESPUES}
    respaldo: ${RESPALDO}

${AMARILLO}Nada se ha publicado todavía.${FIN} Revisa el resultado:

    git log --oneline
    git log --all --pretty=format: --name-only | sort -u | head -30
    pytest

Cuando estés conforme:

    git remote add origin <url>            # filter-repo elimina el remoto por seguridad
    git push --force-with-lease origin main

${ROJO}Esto reescribe la rama pública y es irreversible.${FIN} Todos los SHA cambian y
cualquier clon existente hay que rehacerlo.

Y recuerda: reescribir NO des-filtra el token. Revócalo en
https://www.kaggle.com/settings si aún no lo has hecho.
EOF
