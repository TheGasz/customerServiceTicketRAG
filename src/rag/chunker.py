"""Structure-aware Markdown SOP chunker preserving doc_id, product_model, and heading hierarchy."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class SOPChunk:
    """Structured SOP document chunk with product and section metadata."""

    chunk_id: str
    doc_id: str
    product_model: str
    category: str
    doc_title: str
    section_title: str
    subsection_title: str
    text: str
    lexical_text: str
    dense_passage_text: str

    def to_dict(self) -> Dict[str, Any]:
        """Serialize chunk to dictionary."""
        return asdict(self)


def parse_sop_markdown_file(file_path: Path) -> List[SOPChunk]:
    """Split a single SOP Markdown file by `##` and `###` headings with metadata.

    Creates both:
    - Consolidated procedure chunks per `##` section (combining Symptom + Repair Steps + Escalation)
    - Granular `###` sub-section chunks for fine-grained retrieval

    Args:
        file_path: Path to `.md` SOP file inside `data/kb_docs/`.

    Returns:
        List of `SOPChunk` objects extracted from the file.
    """
    raw_content = file_path.read_text(encoding="utf-8").strip()
    doc_id = file_path.name

    lines = raw_content.splitlines()
    doc_title = doc_id
    product_model = "UNKNOWN"
    category = "Hardware_POS"

    for line in lines[:8]:
        if line.startswith("# "):
            doc_title = line[2:].strip()
        elif line.lower().startswith("product model:"):
            product_model = line.split(":", 1)[1].strip().upper()
        elif line.lower().startswith("category:"):
            category = line.split(":", 1)[1].strip()

    # Split by ## level-2 sections
    sec_splits = re.split(r"(?m)^##\s+", raw_content)
    chunks: List[SOPChunk] = []
    chunk_counter = 1

    for sec_block in sec_splits[1:]:
        sec_lines = sec_block.strip().splitlines()
        if not sec_lines:
            continue
        section_title = sec_lines[0].strip()
        sec_body = "\n".join(sec_lines[1:]).strip()

        # Split by ### level-3 subsections inside this ## section
        sub_splits = re.split(r"(?m)^###\s+", sec_body)

        subsection_blocks: List[str] = []
        for sub_block in sub_splits[1:]:
            sub_lines = sub_block.strip().splitlines()
            if not sub_lines:
                continue
            subsection_title = sub_lines[0].strip()
            sub_body = "\n".join(sub_lines[1:]).strip()
            subsection_blocks.append(f"### {subsection_title}\n{sub_body}")

            # Extract any alphanumeric error code from section_title so dense passage
            # focuses on natural language semantics while BM25 indexes exact error codes
            clean_sec_for_dense = re.sub(r"\([A-Z0-9_-]+\)", "", section_title).strip()
            clean_sub_for_dense = re.sub(r"\b(E-\d{3,4}|ERR_[A-Z0-9_]+|RMA-\d{2}|SLA-GOLD|SSO-\d{3}|VA-\d{3}|OTA-\d{3})\b", "", sub_body)

            full_chunk_text = (
                f"[{doc_id} | {product_model}] ## {section_title} — ### {subsection_title}\n"
                f"{sub_body}"
            )
            chunks.append(
                SOPChunk(
                    chunk_id=f"{doc_id}::chunk-{chunk_counter:02d}",
                    doc_id=doc_id,
                    product_model=product_model,
                    category=category,
                    doc_title=doc_title,
                    section_title=section_title,
                    subsection_title=subsection_title,
                    text=full_chunk_text,
                    lexical_text=f"{product_model} {section_title} {subsection_title} {sub_body}",
                    dense_passage_text=f"passage: {clean_sec_for_dense}. {subsection_title}. {clean_sub_for_dense}",
                )
            )
            chunk_counter += 1

        # Also add a comprehensive ## section-level procedure chunk
        if subsection_blocks:
            combined_body = "\n\n".join(subsection_blocks)
            clean_sec_for_dense = re.sub(r"\([A-Z0-9_-]+\)", "", section_title).strip()
            clean_combined_dense = re.sub(
                r"\b(E-\d{3,4}|ERR_[A-Z0-9_]+|RMA-\d{2}|SLA-GOLD|SSO-\d{3}|VA-\d{3}|OTA-\d{3})\b",
                "",
                combined_body,
            )
            chunks.append(
                SOPChunk(
                    chunk_id=f"{doc_id}::chunk-{chunk_counter:02d}",
                    doc_id=doc_id,
                    product_model=product_model,
                    category=category,
                    doc_title=doc_title,
                    section_title=section_title,
                    subsection_title="Ringkasan Prosedur Lengkap",
                    text=f"[{doc_id} | {product_model}] ## {section_title}\n{combined_body}",
                    lexical_text=f"{product_model} {section_title} {combined_body}",
                    dense_passage_text=f"passage: {clean_sec_for_dense}. {clean_combined_dense}",
                )
            )
            chunk_counter += 1

    return chunks


def load_and_chunk_kb_docs(kb_docs_dir: str | Path = "data/kb_docs") -> List[SOPChunk]:
    """Load all 8 SOP Markdown files from `kb_docs_dir` and return structured chunks."""
    kb_path = Path(kb_docs_dir)
    if not kb_path.exists():
        raise FileNotFoundError(f"KB directory not found: {kb_path}")

    all_chunks: List[SOPChunk] = []
    for md_file in sorted(kb_path.glob("SOP-*.md")):
        all_chunks.extend(parse_sop_markdown_file(md_file))
    return all_chunks


if __name__ == "__main__":
    chunks = load_and_chunk_kb_docs()
    print(f"Total SOP chunks created: {len(chunks)}")
    for c in chunks[:4]:
        print(f"- {c.chunk_id} | product={c.product_model} | section={c.section_title}")
