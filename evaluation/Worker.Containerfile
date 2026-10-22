FROM docker.io/library/python:3.11-slim-bookworm AS wheel
WORKDIR /build
COPY pyproject.toml README.md LICENSE ./
COPY src/ ./src/
COPY build.lock ./
RUN pip install --no-cache-dir -r build.lock && python -m build --wheel --no-isolation

FROM docker.io/library/python:3.11-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
COPY requirements.lock /opt/requirements.lock
RUN pip install --no-cache-dir -r /opt/requirements.lock
COPY --from=wheel /build/dist/*.whl /tmp/wheels/
RUN pip install --no-deps /tmp/wheels/*.whl && rm -rf /tmp/wheels && pip check
ENV XDG_CACHE_HOME=/opt/model-cache PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
COPY prepare_image.py /opt/prepare_image.py
RUN OMP_NUM_THREADS=4 python /opt/prepare_image.py && chmod -R a+rX /opt/model-cache && useradd --uid 1000 --create-home worker && mkdir /workspace && chown 1000:1000 /workspace
USER 1000:1000
WORKDIR /workspace
ENTRYPOINT ["python", "-m", "chemgraph.eval.worker"]
