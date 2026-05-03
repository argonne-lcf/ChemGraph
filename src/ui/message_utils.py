"""Message parsing and molecular-structure extraction helpers.

Every function in this module is **Streamlit-free** so it can be unit-tested
without a running Streamlit runtime.
"""

import ast
import json
import re
from typing import Any, Optional

from ase.data import chemical_symbols


# ---------------------------------------------------------------------------
# Content normalisation
# ---------------------------------------------------------------------------


def normalize_message_content(content: Any) -> str:
    """Convert varying message content payloads (str/list/dict) into plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
                else:
                    parts.append(str(item))
            else:
                parts.append(str(item))
        return "\n".join(p for p in parts if p)
    if isinstance(content, dict):
        text = content.get("text")
        if isinstance(text, str):
            return text
        return str(content)
    return str(content)


# ---------------------------------------------------------------------------
# Message extraction
# ---------------------------------------------------------------------------


def extract_messages_from_result(result: Any) -> list:
    """Extract messages from a result object, handling different formats."""
    if isinstance(result, list):
        return result
    elif isinstance(result, dict) and "messages" in result:
        # Copy the list so we never mutate the original stored in
        # conversation_history -- without this, worker messages would be
        # duplicated on every Streamlit rerun.
        messages = list(result["messages"])
        # For multi-agent workflows, also extract messages from worker_channel
        if "worker_channel" in result:
            worker_channel = result["worker_channel"]
            for _worker_id, worker_messages in worker_channel.items():
                if isinstance(worker_messages, list):
                    messages.extend(worker_messages)
        return messages
    else:
        return [result]


# ---------------------------------------------------------------------------
# Structure extraction
# ---------------------------------------------------------------------------


def extract_molecular_structure(message_content: str) -> Optional[dict]:
    """Return ``{atomic_numbers, positions}`` if structure data is embedded."""
    if not message_content:
        return None

    # Try JSON first
    try:
        if message_content.strip().startswith("{") and message_content.strip().endswith(
            "}"
        ):
            json_data = json.loads(message_content)

            structure_data = None
            if "answer" in json_data:
                structure_data = json_data["answer"]
            elif "numbers" in json_data and "positions" in json_data:
                structure_data = json_data
            elif "atomic_numbers" in json_data and "positions" in json_data:
                structure_data = json_data

            if (
                structure_data
                and "numbers" in structure_data
                and "positions" in structure_data
            ):
                return {
                    "atomic_numbers": structure_data["numbers"],
                    "positions": structure_data["positions"],
                }
            elif (
                structure_data
                and "atomic_numbers" in structure_data
                and "positions" in structure_data
            ):
                return {
                    "atomic_numbers": structure_data["atomic_numbers"],
                    "positions": structure_data["positions"],
                }
    except (json.JSONDecodeError, KeyError):
        pass

    # Plain-text format fallback
    lines = message_content.splitlines()
    atomic_numbers, positions = None, None

    for i, line in enumerate(lines):
        if "Atomic Numbers" in line:
            try:
                numbers_str = line.split(":")[1].strip()
                atomic_numbers = ast.literal_eval(numbers_str)
            except Exception:
                pass
        elif "Positions" in line:
            positions = []
            for sub in lines[i + 1 :]:
                sub = sub.strip()
                if sub.startswith("- [") and sub.endswith("]"):
                    try:
                        positions.append(ast.literal_eval(sub[2:]))
                    except Exception:
                        pass
                elif not sub.startswith("-") and positions:
                    break

    if (
        isinstance(atomic_numbers, list)
        and isinstance(positions, list)
        and len(atomic_numbers) == len(positions)
    ):
        return {"atomic_numbers": atomic_numbers, "positions": positions}

    return None


def find_structure_in_messages(messages: list) -> Optional[dict]:
    """Look through all messages to find structure data."""
    for message in messages:
        if hasattr(message, "content") or isinstance(message, dict):
            raw_content = (
                getattr(message, "content", "")
                if hasattr(message, "content")
                else message.get("content", "")
            )
            content = normalize_message_content(raw_content)
            structure = extract_molecular_structure(content)
            if structure:
                return structure
    return None


def has_structure_signal(
    messages: list, query_text: str = "", final_answer: str = ""
) -> bool:
    """Return True when the interaction appears to include structure artifacts."""
    structure_tools = {
        "smiles_to_coordinate_file",
        "run_ase",
        "file_to_atomsdata",
        "save_atomsdata_to_file",
    }
    structure_markers = (
        ".xyz",
        "final_structure",
        "atomic_numbers",
        "positions",
        "coordinate_file",
    )

    for message in messages:
        name = getattr(message, "name", None)
        content = getattr(message, "content", "")

        if isinstance(message, dict):
            name = message.get("name", name)
            content = message.get("content", content)

        if name in structure_tools:
            return True

        if isinstance(content, str):
            lowered = content.lower()
            if any(marker in lowered for marker in structure_markers):
                return True

    combined_text = f"{query_text}\n{final_answer}".lower()
    keyword_markers = (
        "geometry",
        "optimiz",
        "structure",
        "coordinates",
        "xyz",
    )
    return any(marker in combined_text for marker in keyword_markers)


# ---------------------------------------------------------------------------
# HTML report helpers
# ---------------------------------------------------------------------------


def find_html_filename(messages: list) -> Optional[str]:
    """Scan *messages* in reverse for the first ``*.html`` reference.

    Returns the matched substring (path or bare filename) or ``None``.
    """
    pattern = r"[\w./-]+\.html\b"

    for message in reversed(messages):
        raw_content = ""
        if hasattr(message, "content"):
            raw_content = getattr(message, "content", "")
        elif isinstance(message, dict):
            raw_content = message.get("content", "")
        elif isinstance(message, str):
            raw_content = message
        else:
            raw_content = str(message)
        content = normalize_message_content(raw_content)

        if content:
            match = re.search(pattern, content, flags=re.IGNORECASE)
            if match:
                return match.group(0)

    return None


def extract_xyz_from_report_html(html_content: str) -> Optional[dict]:
    """Decode base64-encoded XYZ data from an HTML report's ``atob()`` call."""
    import base64 as _b64

    match = re.search(r'atob\(["\']([A-Za-z0-9+/=]+)["\']\)', html_content)
    if not match:
        return None

    try:
        xyz_text = _b64.b64decode(match.group(1)).decode("utf-8")
    except Exception:
        return None

    lines = xyz_text.strip().splitlines()
    if len(lines) < 3:
        return None

    try:
        num_atoms = int(lines[0].strip())
    except ValueError:
        return None

    sym_to_num = {s: i for i, s in enumerate(chemical_symbols)}

    atomic_numbers: list[int] = []
    positions: list[list[float]] = []
    for line in lines[2 : 2 + num_atoms]:
        parts = line.strip().split()
        if len(parts) < 4:
            continue
        symbol = parts[0]
        anum = sym_to_num.get(symbol)
        if anum is None:
            return None
        try:
            pos = [float(parts[1]), float(parts[2]), float(parts[3])]
        except ValueError:
            return None
        atomic_numbers.append(anum)
        positions.append(pos)

    if len(atomic_numbers) != num_atoms:
        return None

    return {"atomic_numbers": atomic_numbers, "positions": positions}


def strip_viewer_from_report_html(html_content: str) -> str:
    """Remove the NGL 3D-viewer section from a ChemGraph HTML report.

    Strips the ``<div id="viewer">`` element, the NGL ``<script src>`` tag,
    and the inline ``<script>`` that initialises the NGL stage.  Keeps the
    toggle-section JS so collapsible sections still work.
    """
    # 1. External NGL script tag
    html_content = re.sub(
        r'<script\s+src="[^"]*ngl[^"]*"[^>]*>\s*</script>\s*',
        "",
        html_content,
        flags=re.IGNORECASE,
    )

    # 2. Viewer div
    html_content = re.sub(
        r'<div\s+id="viewer"[^>]*>\s*</div>\s*',
        "",
        html_content,
    )

    # 3. Inline NGL script block -- keep toggle helpers
    html_content = re.sub(
        r"<script>\s*function toggleSection.*?const stage = new NGL\.Stage.*?</script>",
        """<script>
        function toggleSection(sectionId) {
            const content = document.getElementById(sectionId);
            const header = content.previousElementSibling;
            content.classList.toggle('collapsed');
            header.classList.toggle('collapsed');
        }
        function toggleSubSection(sectionId) {
            const content = document.getElementById(sectionId);
            const header = content.previousElementSibling;
            if (content.style.display === 'none') {
                content.style.display = 'block';
                header.classList.remove('collapsed');
            } else {
                content.style.display = 'none';
                header.classList.add('collapsed');
            }
        }
        document.addEventListener('DOMContentLoaded', function() {
            const subSections = document.querySelectorAll('.sub-section-content');
            subSections.forEach(section => { section.style.display = 'block'; });
        });
        </script>""",
        html_content,
        flags=re.DOTALL,
    )

    # 4. "XYZ Molecule Viewer" heading
    html_content = re.sub(
        r"<h1>\s*XYZ Molecule Viewer\s*</h1>\s*",
        "",
        html_content,
    )

    return html_content


# ---------------------------------------------------------------------------
# IR spectrum detection
# ---------------------------------------------------------------------------


def is_infrared_requested(messages: list) -> bool:
    """Return True if any message mentions infrared / IR."""
    for message in messages:
        raw_content = ""
        if hasattr(message, "content"):
            raw_content = getattr(message, "content", "")
        elif isinstance(message, dict):
            raw_content = message.get("content", "")
        elif isinstance(message, str):
            raw_content = message
        else:
            raw_content = str(message)

        content = normalize_message_content(raw_content)
        lowered = content.lower()
        if content and (("infrared" in lowered) or re.search(r"\bir\b", lowered)):
            return True
    return False
