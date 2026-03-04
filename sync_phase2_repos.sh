#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${ROOT_DIR}/data"
LIST_FILE="${ROOT_DIR}/phase2_repos.txt"

if [[ ! -f "${LIST_FILE}" ]]; then
  echo "Lista não encontrada: ${LIST_FILE}"
  exit 1
fi

mkdir -p "${DATA_DIR}"

while IFS= read -r repo; do
  [[ -z "${repo}" ]] && continue
  [[ "${repo}" =~ ^# ]] && continue

  owner="${repo%%/*}"
  name="${repo##*/}"
  target="${DATA_DIR}/${name}"

  if [[ -d "${target}/.git" ]]; then
    echo "[update] ${repo} -> ${target}"
    git -C "${target}" fetch --all --prune
    git -C "${target}" pull --ff-only
  else
    echo "[clone] ${repo} -> ${target}"
    git clone --depth=1 "https://github.com/${repo}.git" "${target}"
  fi

done < "${LIST_FILE}"

echo "Concluído: repositórios da Fase 2 sincronizados."
