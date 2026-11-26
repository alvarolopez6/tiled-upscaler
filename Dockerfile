FROM nvidia/cuda:13.2.1-runtime-ubuntu22.04

COPY --from=ghcr.io/astral-sh/uv:0.13.8 /uv /uvx /bin/

ENV DEBIAN_FRONTEND=noninteractive \
    HF_HOME=/models \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_INSTALL_DIR=/opt/uv/python \
    PATH="/app/.venv/bin:${PATH}"

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY .python-version pyproject.toml uv.lock README.md ./
COPY src ./src

RUN uv sync --frozen --no-dev

ENTRYPOINT ["tiled-upscaler"]
CMD ["--help"]
