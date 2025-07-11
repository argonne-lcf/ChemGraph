import py3Dmol
import chainlit as cl
import os
import uuid

# Ensure the public directory exists
if not os.path.exists("public"):
    os.makedirs("public")


def mol_viewer(xyz: str) -> str:
    """Renders a 3D molecule viewer and returns it as an iframe HTML string."""
    view = py3Dmol.view(width=600, height=400)
    view.addModel(xyz, "xyz")
    view.setStyle({"stick": {}})
    view.zoomTo()

    # Generate a unique filename
    filename = f"mol_viewer_{uuid.uuid4()}.html"
    filepath = os.path.join("public", filename)

    # Write the viewer to an HTML file
    view.write_html(filepath)

    # Return an iframe pointing to the static file
    iframe_html = f'<iframe src="/public/{filename}" width="620" height="420" style="border:none;"></iframe>'
    return iframe_html


def log_tail(log: str) -> str:
    """Shows a collapsible panel with the last 100 lines of a log as an HTML string."""
    lines = log.splitlines()
    # Basic HTML escaping for log content
    last_100_lines = (
        "\n".join(lines[-100:])
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    details_html = f"""
<details>
  <summary>View Full Log</summary>
  <pre><code>{last_100_lines}</code></pre>
</details>
"""
    return details_html
