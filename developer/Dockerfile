FROM node:22-bookworm-slim AS node
FROM python:3.12-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends git curl ripgrep libatomic1 libstdc++6 && rm -rf /var/lib/apt/lists/*
COPY --from=node /usr/local/bin/node /usr/local/bin/node
COPY --from=node /usr/local/lib/node_modules /usr/local/lib/node_modules
RUN ln -s ../lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm && ln -s ../lib/node_modules/npm/bin/npx-cli.js /usr/local/bin/npx
WORKDIR /opt/agent
COPY developer/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && pip freeze > installed-requirements.txt && useradd -u 10001 -M agent
COPY developer/server.py developer/supervisor.py developer/test_tools.py ./
USER 10001:10001
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
CMD ["python","-m","uvicorn","server:app","--host","0.0.0.0","--port","8000"]
