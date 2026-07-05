# PROSPECTOR-AI API

Sistema de **triagem/priorização** de prospecção mineral — Província de Carajás
(alvos IOCG, ouro orogênico e ferro BIF/jaspilito).

> **Escopo honesto.** Esta é uma ferramenta de **priorização exploratória**: combina
> magnetometria (anomalia residual), contexto geológico/geoquímico e regras
> metalogênicas para **ranquear alvos com incerteza explícita**. Não é um oráculo e
> **não decide onde furar** — furo exige validação multi-método (mag + gravimetria +
> geoquímica + estrutura). Ver `RELATORIO_TECNICO.md` para fundamentação e citações.

## Arquitetura

```
Sensor (Teensy) → Raspberry Pi → API (FastAPI) → Supabase ← Dashboard
                                      │
                          GeoSGB/CPRM (WFS) + regras metalogênicas
```

## Endpoints

- `GET /` — dashboard (ou status JSON se `dashboard.html` ausente)
- `GET /health` — health check (versão, env, conexões)
- `POST /api/v1/readings` — processa uma leitura do sensor (autenticado)
- `GET /api/v1/results` — lista resultados persistidos (autenticado)
- `GET /api/v1/hotspots` — clusters de anomalias persistentes (autenticado)

Autenticação: header `Authorization: Bearer <API_KEY>`.

## Variáveis de ambiente

| Variável | Obrigatória | Descrição |
|---|---|---|
| `APP_ENV` | — | `production` ativa validação fail-fast (default: `development`) |
| `API_KEY` | **sim em prod** | Token Bearer da API. Em prod não pode ser vazio nem o default de dev |
| `CORS_ORIGINS` | **sim em prod** | Origens permitidas (separadas por vírgula). Vazio = `*` (só dev) |
| `SUPABASE_URL` | recomendado | URL do projeto Supabase (persistência) |
| `SUPABASE_KEY` | recomendado | Chave de serviço do Supabase |
| `OPENAI_API_KEY` | opcional | Habilita o parecer textual por IA; sem ela a IA é desativada |
| `USAR_IA` | opcional | `true`/`false` (default `true`, mas só ativa com `OPENAI_API_KEY`) |
| `LOG_LEVEL` | opcional | `INFO` (default), `DEBUG`, etc. |
| `PORT` | — | Porta (Render define automaticamente) |

Em **produção** a aplicação **recusa subir** se `API_KEY` for o default de dev ou se
`CORS_ORIGINS` estiver liberado para `*` (ver `config.validar_producao`).

## Rodar localmente

```bash
pip install -r requirements-dev.txt
export API_KEY=dev-key-12345        # ou o seu
uvicorn api:app --reload
```

## Testes

```bash
pytest
```

CI roda `py_compile` + `pytest` em cada push/PR (`.github/workflows/ci.yml`).

## Deploy no Render

1. Conecte o repositório ao Render.com.
2. Configure as variáveis de ambiente acima (no mínimo `APP_ENV=production`,
   `API_KEY`, `CORS_ORIGINS`, `SUPABASE_URL`, `SUPABASE_KEY`).
3. O `Procfile` inicia `uvicorn api:app`.

> **Estado em memória:** o buffer de micronivelamento e a memória espacial vivem no
> processo. Rode **um worker** (ou externalize o estado para banco/Redis) até a Fase 2
> do roadmap — ver `RELATORIO_TECNICO.md` §6. Com múltiplos workers, cada um teria
> estado próprio.

## Limitações conhecidas / roadmap

Documentadas em `RELATORIO_TECNICO.md` §6:
- **Fase 2:** adotar Harmonica/Verde/SimPEG; integrar gravimetria + EM/MT.
- **Fase 3:** ML defensável com depósitos **reais** (GeoSGB) como rótulos, PU learning
  e validação cruzada espacial — aposentando o classificador sintético atual.
- Nomes reais das camadas do GeoSGB ainda **não verificados**; o cliente os descobre em
  runtime via GetCapabilities.
