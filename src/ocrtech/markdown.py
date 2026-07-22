"""Markdown rendering for parsed documents."""

from __future__ import annotations

from .schemas import Block, Document
from .tables import table_to_markdown


def render_document_markdown(document: Document, *, include_structural_roles: bool = True) -> str:
    tables = {table.table_id: table for table in document.tables}
    figures = {figure.figure_id: figure for figure in document.figures}
    sections: list[str] = []
    for page in sorted(document.pages, key=lambda item: item.page_index):
        if len(document.pages) > 1:
            sections.append(f"<!-- page {page.page_index + 1} -->")
        for block in sorted(page.blocks, key=lambda item: item.order):
            if not include_structural_roles and _is_structural_block(block):
                continue
            if block.block_type == "title":
                sections.append(f"# {block.text.strip()}")
            elif block.block_type == "table" and block.table_id in tables:
                table_md = table_to_markdown(tables[block.table_id])
                if table_md:
                    sections.append(table_md)
            elif block.block_type == "figure" and block.figure_id in figures:
                figure = figures[block.figure_id]
                alt = figure.caption or figure.summary or figure.figure_id
                if figure.image_path:
                    sections.append(f"![{alt}]({figure.image_path})")
                if figure.caption:
                    sections.append(figure.caption)
                if figure.summary:
                    sections.append(figure.summary)
            elif block.text.strip():
                sections.append(block.text.strip())
    return "\n\n".join(sections).strip() + "\n"


def _is_structural_block(block: Block) -> bool:
    role = (block.metadata or {}).get("structural_role")
    return isinstance(role, str) and role not in {"", "text"}
