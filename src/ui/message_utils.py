"""Message parsing and molecular-structure extraction helpers.

Every function in this module is **Streamlit-free** so it can be unit-tested
without a running Streamlit runtime.
"""

import ast
import json
import re
from typing import Any, Literal, Optional

from ase.data import chemical_symbols


# ---------------------------------------------------------------------------
# Content normalisation
# ---------------------------------------------------------------------------


def normalize_message_content(content: Any) -> str:
    """Convert varying message content payloads into plain text.

    Parameters
    ----------
    content : Any
        Message content as text, list blocks, dictionary, or scalar.

    Returns
    -------
    str
        Normalized plain-text content.
    """
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


def normalize_latex_delimiters(text: str) -> str:
    """Convert common LLM math delimiters into Streamlit-renderable Markdown.

    Parameters
    ----------
    text : str
        Markdown text that may contain LLM-style math delimiters.

    Returns
    -------
    str
        Text with display and inline math delimiters normalized.
    """
    if not text:
        return ""

    text = _convert_square_bracket_math(text)

    return _convert_parenthetical_math_outside_display(text)


def _is_latex_square_delimiter(text: str, index: int) -> bool:
    """Return whether a bracket belongs to ``\\left``/``\\right``.

    Parameters
    ----------
    text : str
        Source text.
    index : int
        Bracket index to inspect.

    Returns
    -------
    bool
        ``True`` when the bracket is part of a LaTeX delimiter command.
    """
    prefix = text[max(0, index - 6) : index]
    return prefix.endswith(r"\left") or prefix.endswith(r"\right")


def _find_square_math_close(text: str, start: int) -> int | None:
    """Find the matching closing square bracket for display math.

    Parameters
    ----------
    text : str
        Source text.
    start : int
        Index of the opening square bracket.

    Returns
    -------
    int or None
        Closing bracket index, or ``None`` if unmatched.
    """
    depth = 1
    index = start + 1
    while index < len(text):
        if text.startswith(r"\left[", index):
            index += len(r"\left[")
            continue
        if text.startswith(r"\right]", index):
            index += len(r"\right]")
            continue

        char = text[index]
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                return index
        index += 1
    return None


def _convert_square_bracket_math(text: str) -> str:
    """Convert LLM-style ``[ TeX ]`` display math while preserving links.

    Parameters
    ----------
    text : str
        Source Markdown text.

    Returns
    -------
    str
        Text with display math converted to ``$$`` blocks.
    """
    chunks: list[str] = []
    index = 0
    while index < len(text):
        char = text[index]
        if char != "[" or _is_latex_square_delimiter(text, index):
            chunks.append(char)
            index += 1
            continue

        close_index = _find_square_math_close(text, index)
        if close_index is None:
            chunks.append(char)
            index += 1
            continue

        body = text[index + 1 : close_index].strip()
        has_latex_command = bool(re.search(r"\\[A-Za-z]+", body))
        is_markdown_link = close_index + 1 < len(text) and text[close_index + 1] == "("
        if body and has_latex_command and not is_markdown_link:
            chunks.append(f"$$\n{body}\n$$")
        else:
            chunks.append(text[index : close_index + 1])
        index = close_index + 1

    return "".join(chunks)


def _find_parenthesis_close(text: str, start: int) -> int | None:
    """Find the matching closing parenthesis.

    Parameters
    ----------
    text : str
        Source text.
    start : int
        Index of the opening parenthesis.

    Returns
    -------
    int or None
        Closing parenthesis index, or ``None`` if unmatched.
    """
    depth = 1
    index = start + 1
    while index < len(text):
        char = text[index]
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return index
        index += 1
    return None


def _convert_parenthetical_inline_math(text: str) -> str:
    """Convert parenthesized math-like text to inline math.

    Parameters
    ----------
    text : str
        Text outside display-math blocks.

    Returns
    -------
    str
        Text with math-like parenthetical content wrapped in ``$``.
    """
    chunks: list[str] = []
    index = 0
    while index < len(text):
        char = text[index]
        if char != "(":
            chunks.append(char)
            index += 1
            continue

        close_index = _find_parenthesis_close(text, index)
        if close_index is None:
            chunks.append(char)
            index += 1
            continue

        body = text[index + 1 : close_index].strip()
        has_latex = bool(re.search(r"\\[A-Za-z]+|\\\s|[_^{}]", body))
        if body and has_latex:
            chunks.append(f"${body}$")
        else:
            chunks.append(text[index : close_index + 1])
        index = close_index + 1

    return "".join(chunks)


def _convert_parenthetical_math_outside_display(text: str) -> str:
    """Convert parenthetical math outside display-math blocks.

    Parameters
    ----------
    text : str
        Markdown text that may contain display-math blocks.

    Returns
    -------
    str
        Text with inline parenthetical math normalized.
    """
    chunks: list[str] = []
    last_end = 0
    for match in re.finditer(r"(?s)\$\$.*?\$\$", text):
        chunks.append(_convert_parenthetical_inline_math(text[last_end : match.start()]))
        chunks.append(match.group(0))
        last_end = match.end()
    chunks.append(_convert_parenthetical_inline_math(text[last_end:]))
    return "".join(chunks)


def split_markdown_latex_blocks(
    text: str,
) -> list[tuple[Literal["markdown", "latex"], str]]:
    """Split text into Markdown and display-LaTeX blocks.

    Parameters
    ----------
    text : str
        Markdown text that may contain display math blocks.

    Returns
    -------
    list[tuple[Literal["markdown", "latex"], str]]
        Ordered render blocks for Streamlit.
    """
    normalized = normalize_latex_delimiters(text)
    if not normalized:
        return []

    parts: list[tuple[Literal["markdown", "latex"], str]] = []
    last_end = 0
    for match in re.finditer(r"(?s)\$\$\s*(.*?)\s*\$\$", normalized):
        markdown = normalized[last_end : match.start()].strip()
        if markdown:
            parts.append(("markdown", markdown))

        latex = match.group(1).strip()
        if latex:
            parts.append(("latex", latex))
        last_end = match.end()

    trailing = normalized[last_end:].strip()
    if trailing:
        parts.append(("markdown", trailing))

    return parts


# ---------------------------------------------------------------------------
# Message extraction
# ---------------------------------------------------------------------------


def extract_messages_from_result(result: Any) -> list:
    """Extract messages from a result object, handling different formats.

    Parameters
    ----------
    result : Any
        Agent result, state dictionary, message list, or scalar result.

    Returns
    -------
    list
        Extracted message-like objects.
    """
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
    """Return embedded molecular structure data if present.

    Parameters
    ----------
    message_content : str
        Message text that may contain JSON or plain-text structure data.

    Returns
    -------
    dict or None
        Dictionary with ``atomic_numbers`` and ``positions``, or ``None``.
    """
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
    """Look through messages in reverse to find the latest structure data.

    Parameters
    ----------
    messages : list
        Message-like objects or dictionaries to scan.

    Returns
    -------
    dict or None
        Latest embedded structure dictionary, or ``None``.
    """
    for message in reversed(messages):
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
    """Return True when an interaction appears to include structure artifacts.

    Parameters
    ----------
    messages : list
        Message-like objects or dictionaries to inspect.
    query_text : str, optional
        Original user query text.
    final_answer : str, optional
        Final assistant answer text.

    Returns
    -------
    bool
        ``True`` when structure-related tools, artifacts, or text are found.
    """
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

    Parameters
    ----------
    messages : list
        Message-like objects, dictionaries, or strings to scan.

    Returns
    -------
    str or None
        First HTML path/filename found from the end of the message list.
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
    """Decode base64-encoded XYZ data from an HTML report.

    Parameters
    ----------
    html_content : str
        HTML report content containing an ``atob()`` XYZ payload.

    Returns
    -------
    dict or None
        Structure dictionary with atomic numbers and positions, or ``None``.
    """
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

    Parameters
    ----------
    html_content : str
        Raw HTML report content.

    Returns
    -------
    str
        HTML content with the embedded molecule viewer removed.
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
    """Return True if any message mentions infrared or IR.

    Parameters
    ----------
    messages : list
        Message-like objects, dictionaries, or strings to inspect.

    Returns
    -------
    bool
        ``True`` when infrared-related wording is found.
    """
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
