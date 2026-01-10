# PROSPECTOR-AI API

Sistema de Prospecção Mineral - Província de Carajás

## Deploy no Render

1. Fork/clone este repositório
2. Conecte ao Render.com
3. Configure as variáveis de ambiente:
   - `SUPABASE_URL`
   - `SUPABASE_KEY`
   - `ANTHROPIC_API_KEY` (opcional)

## Endpoints

- `GET /` - Status da API
- `GET /health` - Health check
- `POST /reading` - Recebe leitura do sensor
- `GET /readings` - Lista leituras
- `GET /results` - Lista resultados processados
- `GET /dashboard` - Dashboard visual

## Arquitetura

```
Sensor (Teensy) → Raspberry Pi → Supabase ← API (Render) ← Dashboard
```
