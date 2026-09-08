"""Testy dokumentace ``docs/JAK_TO_FUNGUJE.md``.

Výklad cituje kód doslova. Bez těchto testů by se text tiše rozešel se
zdrojáky při první úpravě algoritmu – a nesprávný výklad je horší než žádný.
"""

from __future__ import annotations

import os
import re
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

DOC_PATH = os.path.join(ROOT, "docs", "JAK_TO_FUNGUJE.md")

#: Blok kódu, za kterým následuje řádek ``<sub>`soubor.py`, funkce …</sub>``.
QUOTE_RE = re.compile(r"```python\n(.*?)```\n<sub>`([^`]+\.py)`", re.S)


@pytest.fixture(scope="module")
def doc() -> str:
    with open(DOC_PATH, encoding="utf-8") as handle:
        return handle.read()


def _source(name: str) -> str:
    with open(os.path.join(ROOT, name), encoding="utf-8") as handle:
        return handle.read()


def test_documentation_exists(doc):
    assert len(doc) > 5000, "výklad je podezřele krátký"


def test_every_quoted_snippet_is_verbatim_in_its_source(doc):
    """Každý citovaný úryvek musí být doslova v uvedeném souboru."""
    quotes = QUOTE_RE.findall(doc)
    assert len(quotes) >= 15, f"nalezeno jen {len(quotes)} citací kódu"

    mismatched = []
    for code, source_name in quotes:
        body = code.rstrip("\n")
        if body not in _source(source_name):
            first_line = body.strip().splitlines()[0]
            mismatched.append(f"{source_name}: {first_line}")

    assert not mismatched, (
        "dokumentace cituje kód, který ve zdrojáku není – po úpravě algoritmu "
        "je potřeba srovnat i výklad:\n  " + "\n  ".join(mismatched)
    )


def test_quoted_sources_exist(doc):
    for _code, source_name in QUOTE_RE.findall(doc):
        assert os.path.isfile(os.path.join(ROOT, source_name)), f"chybí {source_name}"


def test_referenced_images_exist(doc):
    images = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", doc)
    assert images, "výklad nemá žádné obrázky"
    for relative in images:
        path = os.path.join(os.path.dirname(DOC_PATH), relative)
        assert os.path.isfile(path), f"chybí obrázek {relative}"
        assert os.path.getsize(path) > 1024, f"obrázek {relative} je prázdný"


def test_internal_links_point_to_real_headings(doc):
    """Odkazy v obsahu musí mířit na existující nadpis (slug jako na GitHubu)."""

    def slug(title: str) -> str:
        text = title.strip().lower()
        text = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE)  # interpunkce se maže
        return text.replace(" ", "-")

    headings = {slug(h) for h in re.findall(r"^##+ (.+)$", doc, re.M)}
    broken = [link for link in re.findall(r"\]\(#([^)]+)\)", doc) if link not in headings]
    assert not broken, f"odkazy bez cíle: {broken}"


def test_figure_generator_is_importable():
    """Skript na obrázky musí zůstat spustitelný (jinak je dokumentace nereprodukovatelná)."""
    path = os.path.join(ROOT, "tools", "make_docs_figures.py")
    assert os.path.isfile(path)
    with open(path, encoding="utf-8") as handle:
        compile(handle.read(), path, "exec")


def test_readme_links_the_deep_dive():
    with open(os.path.join(ROOT, "README.md"), encoding="utf-8") as handle:
        readme = handle.read()
    assert "docs/JAK_TO_FUNGUJE.md" in readme
