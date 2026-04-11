FROM --platform=linux/amd64 ghcr.io/astral-sh/uv:python3.12-trixie

RUN adduser agentbeats
USER agentbeats
RUN mkdir -p /home/agentbeats/.cache/uv
WORKDIR /home/agentbeats/app

COPY pyproject.toml uv.lock README.md ./
COPY src src

RUN \
    --mount=type=cache,target=/home/agentbeats/.cache/uv,uid=1000 \
    uv sync --no-editable --locked

ENTRYPOINT ["uv", "run", "src/server.py"]
CMD ["--host", "0.0.0.0", "--port", "9009"]
EXPOSE 9009
