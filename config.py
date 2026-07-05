#!/usr/bin/env python3
"""Configuração centralizada do PROSPECTOR-AI (lida de variáveis de ambiente).

Em produção (`APP_ENV=production`) a aplicação se RECUSA a subir com configuração
insegura (chave padrão, CORS liberado) — ver `validar_producao()`. Em
desenvolvimento, usa defaults convenientes apenas emitindo avisos.
"""

from __future__ import annotations

import os
from typing import List

VERSION = "5.0"

DEV_API_KEY = "dev-key-12345"


def _bool_env(nome: str, default: bool) -> bool:
    val = os.environ.get(nome)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on", "sim")


# Ambiente
APP_ENV = os.environ.get("APP_ENV", "development").strip().lower()
IS_PRODUCTION = APP_ENV in ("production", "prod")

# Segurança / credenciais
_api_key_env = os.environ.get("API_KEY", "").strip()
# Em dev, cai no key de desenvolvimento para conveniência; em prod, exige a real.
API_KEY = _api_key_env or ("" if IS_PRODUCTION else DEV_API_KEY)

SUPABASE_URL = os.environ.get("SUPABASE_URL", "").strip().rstrip("/")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "").strip()

# Comportamento
USAR_IA = _bool_env("USAR_IA", True) and bool(os.environ.get("OPENAI_API_KEY", "").strip())
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").strip().upper()

# CORS — lista separada por vírgula. Vazio => '*' (apenas dev).
# Navegadores rejeitam allow_origins='*' junto com credentials, então só
# habilitamos credentials quando há origens explícitas.
_origins_raw = os.environ.get("CORS_ORIGINS", "").strip()
CORS_ORIGINS: List[str] = [o.strip() for o in _origins_raw.split(",") if o.strip()] or ["*"]
CORS_ALLOW_CREDENTIALS = CORS_ORIGINS != ["*"]


def validar_producao() -> List[str]:
    """Lista de problemas de configuração que impedem operação segura em produção.
    Vazia => configuração OK."""
    problemas: List[str] = []
    if not IS_PRODUCTION:
        return problemas
    if not API_KEY or API_KEY == DEV_API_KEY:
        problemas.append("API_KEY ausente ou igual à chave de desenvolvimento.")
    if CORS_ORIGINS == ["*"]:
        problemas.append("CORS_ORIGINS não definido — origens liberadas para '*'.")
    if not SUPABASE_KEY or not SUPABASE_URL:
        problemas.append("SUPABASE_URL/KEY ausentes — resultados não serão persistidos.")
    return problemas
