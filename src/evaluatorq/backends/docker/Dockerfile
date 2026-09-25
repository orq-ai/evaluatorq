# Coding agents are Node programs. A task image without Node must install it before the npm install line.
FROM node:22-slim
ARG CLAUDE_CODE_VERSION=2.1.282
ARG CODEX_VERSION=0.157.0
ARG OPENCODE_VERSION=1.18.32
ARG ORQ_CLI_VERSION=11.0.0
RUN apt-get update && apt-get install -y --no-install-recommends git ripgrep ca-certificates \
 && rm -rf /var/lib/apt/lists/*
RUN npm install -g @anthropic-ai/claude-code@${CLAUDE_CODE_VERSION} @openai/codex@${CODEX_VERSION} \
      opencode-ai@${OPENCODE_VERSION} @orq-ai/cli@${ORQ_CLI_VERSION}
COPY entrypoint.sh /usr/local/bin/evq-entrypoint
RUN chmod 0755 /usr/local/bin/evq-entrypoint
RUN chmod 0666 /etc/passwd && useradd -m -u 1000 agent
USER agent
