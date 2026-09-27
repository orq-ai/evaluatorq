#!/bin/sh
# Runs at the front of every `docker exec`. Gives the runtime uid (the host user's, via --user) a passwd
# entry, because Node's os.userInfo() throws for a uid that has none. Idempotent; then runs the agent.
if ! id -un >/dev/null 2>&1; then
  echo "evq:x:$(id -u):$(id -g)::${HOME:-/evq-home}:/bin/sh" >> /etc/passwd
fi
exec "$@"
