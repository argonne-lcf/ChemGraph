"""Read-only package resources exposed through the Deep Agents file protocol."""

from importlib import resources
from pathlib import PurePosixPath

from deepagents.backends.protocol import (
    BackendProtocol,
    DeleteResult,
    EditResult,
    FileDownloadResponse,
    FileUploadResponse,
    GlobResult,
    LsResult,
    ReadResult,
    WriteResult,
)
from deepagents.backends.utils import (
    compile_recursive_glob,
    create_file_data,
    grep_matches_from_files,
    slice_read_response,
)
from deepagents.middleware.skills import _parse_skill_metadata


class BundledSkillsBackend(BackendProtocol):
    """Serve a snapshot of skill resources without filesystem or graph state.

    Async operations inherit the protocol's sync adapters. No local host path
    is exposed: resources may come from an unpacked wheel or a zip importer.
    """

    def __init__(self):
        root = resources.files("chemgraph.skills")
        self._files: dict[str, bytes] = {}
        skill_names = set()

        def collect(directory, prefix):
            for item in sorted(directory.iterdir(), key=lambda item: item.name):
                if item.name.startswith(".") or item.name == "__pycache__":
                    continue
                path = f"{prefix}/{item.name}"
                if item.is_dir():
                    collect(item, path)
                else:
                    self._files[path] = item.read_bytes()

        for directory in sorted(root.iterdir(), key=lambda item: item.name):
            if not directory.is_dir() or directory.name.startswith(("_", ".")):
                continue
            skill_path = f"/{directory.name}/SKILL.md"
            collect(directory, f"/{directory.name}")
            content = self._files.get(skill_path)
            if (
                content is None
                or _parse_skill_metadata(
                    content.decode("utf-8"),
                    skill_path,
                    directory.name,
                )
                is None
            ):
                raise ValueError(f"Invalid bundled skill: {skill_path}")
            skill_names.add(directory.name)
        missing = {"chemgraph", "pbs-hpc"} - skill_names
        if missing:
            raise ValueError(f"Missing bundled skills: {', '.join(sorted(missing))}")

        self._text_files = {
            path: create_file_data(content.decode("utf-8"))
            for path, content in self._files.items()
            if self._is_text(content)
        }

    @staticmethod
    def _is_text(content):
        try:
            content.decode("utf-8")
            return True
        except UnicodeDecodeError:
            return False

    @staticmethod
    def _valid_path(path):
        return path.startswith("/") and ".." not in PurePosixPath(path).parts

    def _info(self, path):
        return {"path": path, "is_dir": False, "size": len(self._files[path])}

    def ls(self, path):
        if not self._valid_path(path):
            return LsResult(error="invalid_path")
        prefix = path.rstrip("/") + "/"
        entries = {}
        for filename in self._files:
            if not filename.startswith(prefix):
                continue
            relative = filename[len(prefix) :]
            if "/" in relative:
                child = prefix + relative.split("/", 1)[0] + "/"
                entries[child] = {"path": child, "is_dir": True}
            else:
                entries[filename] = self._info(filename)
        if not entries and path != "/":
            return LsResult(error="path_not_found")
        return LsResult(entries=[entries[key] for key in sorted(entries)])

    def read(self, file_path, offset=0, limit=2000):
        if not self._valid_path(file_path):
            return ReadResult(error="invalid_path")
        if file_path not in self._files:
            return ReadResult(error="file_not_found")
        if file_path not in self._text_files:
            return ReadResult(error="Binary resource; use download_files.")
        return slice_read_response(self._text_files[file_path], offset, limit)

    def glob(self, pattern, path=None):
        prefix = (path or "/").rstrip("/") + "/"
        if not self._valid_path(prefix):
            return GlobResult(error="invalid_path")
        matches = compile_recursive_glob(pattern)
        return GlobResult(
            matches=[
                self._info(filename)
                for filename in sorted(self._files)
                if filename.startswith(prefix) and matches(filename[len(prefix) :])
            ]
        )

    def grep(self, pattern, path=None, glob=None, *, max_count=None):
        return grep_matches_from_files(
            self._text_files,
            pattern,
            path or "/",
            glob,
            max_count=max_count,
        )

    def download_files(self, paths):
        return [
            FileDownloadResponse(
                path=path,
                content=self._files.get(path),
                error=(
                    "invalid_path"
                    if not self._valid_path(path)
                    else "file_not_found"
                    if path not in self._files
                    else None
                ),
            )
            for path in paths
        ]

    def write(self, file_path, content):
        return WriteResult(error="Bundled skills are read-only.")

    def edit(self, file_path, old_string, new_string, replace_all=False):
        return EditResult(error="Bundled skills are read-only.")

    def delete(self, file_path):
        return DeleteResult(error="Bundled skills are read-only.")

    def upload_files(self, files):
        return [
            FileUploadResponse(path=path, error="permission_denied")
            for path, _ in files
        ]
