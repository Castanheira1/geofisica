import os

# Ambiente de teste determinístico: modo dev, chave conhecida, IA desligada.
os.environ.setdefault("APP_ENV", "development")
os.environ.setdefault("API_KEY", "test-key")
os.environ.pop("OPENAI_API_KEY", None)

# Evita qualquer tentativa de rede ao GeoSGB durante os testes: marca o cache de
# camadas como "consultado e vazio".
import geological_layers  # noqa: E402

geological_layers.GeoSGBClient._typenames_cache = set()
