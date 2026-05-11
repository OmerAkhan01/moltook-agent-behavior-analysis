from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import streamlit as st


@dataclass(frozen=True)
class PlotImage:
    path: str
    title: str
    caption: str | None = None


def repo_root() -> Path:
    # moltbook/src/ui.py -> moltbook/src -> moltbook (repo root)
    return Path(__file__).resolve().parents[1]


def abs_path(relative_path: str) -> str:
    return str((repo_root() / relative_path).resolve())


def apply_theme() -> None:
    st.markdown(
        """
<style>
  :root {
    --ma-bg: #0b1220;
    --ma-surface: rgba(255, 255, 255, 0.06);
    --ma-border: rgba(255, 255, 255, 0.10);
    --ma-text: rgba(255, 255, 255, 0.92);
    --ma-muted: rgba(255, 255, 255, 0.70);
  }

  [data-testid="stSidebar"] { background-color: #0f172a; }
  [data-testid="stSidebar"] * { color: var(--ma-text) !important; }

  .ma-card {
    background: var(--ma-surface);
    border: 1px solid var(--ma-border);
    border-radius: 14px;
    padding: 14px 14px 10px 14px;
  }

  .ma-title {
    font-size: 0.95rem;
    font-weight: 650;
    line-height: 1.2;
    color: var(--ma-text);
    margin: 0 0 0.25rem 0;
  }

  .ma-caption {
    font-size: 0.85rem;
    color: var(--ma-muted);
    margin: 0.2rem 0 0 0;
  }

  .block-container { padding-top: 1.25rem; padding-bottom: 2rem; }
</style>
        """,
        unsafe_allow_html=True,
    )


def image_card(img: PlotImage, *, container: Optional[st.delta_generator.DeltaGenerator] = None) -> None:
    c = container or st
    p = abs_path(img.path)
    if not Path(p).exists():
        c.warning(f"Görsel bulunamadı: `{img.path}`")
        return

    c.markdown('<div class="ma-card">', unsafe_allow_html=True)
    c.markdown(f'<div class="ma-title">{img.title}</div>', unsafe_allow_html=True)
    c.image(p, use_container_width=True)
    if img.caption:
        c.markdown(f'<div class="ma-caption">{img.caption}</div>', unsafe_allow_html=True)
    c.markdown("</div>", unsafe_allow_html=True)


def image_grid(images: Iterable[PlotImage], *, columns: int = 2) -> None:
    images = list(images)
    if not images:
        st.info("Gösterilecek görsel yok.")
        return

    for i in range(0, len(images), columns):
        cols = st.columns(columns)
        for j, img in enumerate(images[i : i + columns]):
            with cols[j]:
                image_card(img)

