#!/usr/bin/env bash
# Wrapper simples para Linux/macOS do instalador Python (instalar.py).
# Repassa todos os argumentos recebidos (ex.: --abrir, --sem-dataset, --zip-local).
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Verificando Python 3 no sistema..."
if ! command -v python3 >/dev/null 2>&1; then
  echo "Erro: python3 não encontrado no PATH. Instale o Python 3.10+ e tente novamente." >&2
  exit 1
fi

echo "Delegando instalação para $DIR/instalar.py..."
exec python3 "$DIR/instalar.py" "$@"
