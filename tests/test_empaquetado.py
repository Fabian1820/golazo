"""El paquete debe estar completo en el repositorio.

Existe por un fallo real: `.gitignore` contenía `models/` para excluir los
artefactos de modelos entrenados, y ese patrón —sin anclar— también excluía
`golazo/models/`, el paquete con el código de los modelos. `git add -A` lo
omitió en silencio, todo funcionaba en local, y el repositorio publicado era
ininstalable.

Los tests normales no lo detectan porque importan desde el árbol de trabajo.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parent.parent


def _es_repo_git() -> bool:
    return (RAIZ / ".git").exists()


@pytest.mark.skipif(not _es_repo_git(), reason="fuera de un repositorio git")
def test_todo_el_codigo_fuente_esta_en_git():
    seguidos = subprocess.run(
        ["git", "ls-files", "golazo"], cwd=RAIZ, capture_output=True, text=True, check=True
    ).stdout.split()
    seguidos = {p for p in seguidos if p.endswith(".py")}

    en_disco = {
        str(p.relative_to(RAIZ))
        for p in (RAIZ / "golazo").rglob("*.py")
        if "__pycache__" not in p.parts
    }

    faltan = en_disco - seguidos
    assert not faltan, (
        "Estos ficheros del paquete no están en git y el repositorio publicado "
        f"quedaría roto: {sorted(faltan)}. Comprueba .gitignore con "
        "`git check-ignore -v <fichero>`."
    )


@pytest.mark.skipif(not _es_repo_git(), reason="fuera de un repositorio git")
def test_ningun_subpaquete_esta_ignorado():
    """Comprueba directamente lo que falló: un patrón sin anclar."""
    paquetes = [p for p in (RAIZ / "golazo").rglob("__init__.py")]
    assert len(paquetes) >= 3, "se esperaban al menos golazo, golazo.models y golazo.sources"

    for init in paquetes:
        rel = str(init.relative_to(RAIZ))
        r = subprocess.run(["git", "check-ignore", "-v", rel], cwd=RAIZ, capture_output=True, text=True)
        assert r.returncode != 0, f"{rel} está ignorado por: {r.stdout.strip()}"


def test_los_subpaquetes_se_pueden_importar():
    import golazo.models
    import golazo.sources

    assert golazo.models.DixonColes is not None
    assert golazo.sources.UnderstatSource is not None
