#!/usr/bin/env sh
# Nightly backup of the Sapphirus Assistant (run from the compose directory, e.g. via cron at 02:00):
#   0 2 * * * cd /opt/sapphirus && ./deploy/backup.sh >> logs/backup.log 2>&1
# Backs up: PostgreSQL (users, tokens, documents registry, chats, metrics, jobs), the data volume (vector index,
# uploaded originals, archive) and the knowledge-base source files. Keeps 14 days locally - copy the folder to
# off-site storage (S3 / Azure Blob) as well.
set -eu
STAMP=$(date +%Y%m%d-%H%M%S)
DEST=${BACKUP_DIR:-./backups}/$STAMP
mkdir -p "$DEST"
docker compose exec -T db pg_dump -U sapphirus -d sapphirus_assistant --format=custom > "$DEST/postgres.dump"
docker compose run --rm --no-deps -v "$(pwd)/$DEST:/backup" --entrypoint sh api \
  -c "tar czf /backup/appdata.tgz -C /app data"
tar czf "$DEST/knowledge.tgz" knowledge_bases datasets
find "${BACKUP_DIR:-./backups}" -mindepth 1 -maxdepth 1 -type d -mtime +14 -exec rm -rf {} +
echo "Backup written to $DEST"
# Restore: pg_restore -U sapphirus -d sapphirus_assistant --clean postgres.dump ; untar appdata.tgz into the volume.
# The vector index can also be rebuilt from the source files at any time: python -m ingestion.cli reindex
