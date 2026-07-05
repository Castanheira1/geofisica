# Relatório Técnico — Estado-da-arte e Plano de Correção do PROSPECTOR-AI

**Província Mineral de Carajás (PA) — alvos IOCG, ouro orogênico e ferro (BIF/jaspilito)**

Data: 2026-06-16. Documento gerado a partir de pesquisa multi-fonte (serviços geológicos
nacionais, literatura revisada por pares e documentação oficial de software). Cada
recomendação está ancorada em fonte citada. Onde uma afirmação não pôde ser verificada no
texto primário (muitos editores retornaram HTTP 403 a coleta automatizada), está marcada
como **[verificação a nível de resumo]**.

> **Aviso de honestidade.** Este documento **não** afirma que o sistema é "o software mais
> avançado do mundo". Essa seria a mesma sobre-promessa (over-claim) que o próprio relatório
> identifica como defeito. O objetivo é o oposto: **ancorar o sistema no estado-da-arte real**
> usado por serviços geológicos e pela indústria, e **remover o que é pseudociência**, tratando
> incerteza de forma honesta. Um sistema de triagem defensável vale mais que um "oráculo"
> super-confiante.

---

## Sumário executivo — o que estava errado e o que muda

| # | Problema no código atual | Veredito | Correção |
|---|---|---|---|
| 1 | Classificação de depósito por **nT absoluto** (`26000–28500 nT = IOCG`) | **Incorreto** (geofísica) | Usar **anomalia residual** (pós-IGRF) e produtos direção-independentes (sinal analítico), não campo total |
| 2 | **Rede neural treinada em rótulos sintéticos** auto-gerados | **Inválido** (circular) | Remover como "preditor"; usar só como heurística declarada; ML real exige depósitos conhecidos como rótulos |
| 3 | Decisão `PERFURAR_IMEDIATO` com `confianca=0.98` | **Over-claim perigoso** | Saída = **favorabilidade relativa + incerteza explícita**, nunca veredito de furo |
| 4 | `typeName` do GeoSGB **chutados**, erros engolidos | **Frágil** | **Descoberta em runtime** (GetCapabilities) + falha explícita + medir completude do dado |
| 5 | `WorldModel` recriado por requisição (buffer/memória resetam) | **Quebrado em produção** | Serviço de estado persistente (documentado) |
| 6 | Só Cu/Au; um único "tipo" de alvo | **Incompleto** | Pathfinders completos + **separar tipos de alvo** (IOCG-magnetita / IOCG-hematita / ouro orogênico / BIF) |

---

## 1. Processamento de campo potencial (magnetometria) — workflow correto

**1.1 — O campo total (TMI) é dominado pelo IGRF e não é interpretável diretamente.** A
anomalia magnética é o que sobra após subtrair o IGRF (campo principal da Terra, da ordem de
25.000–60.000 nT, que varia com latitude, altitude e época) e corrigir a variação diurna. Só a
**anomalia residual** representa as rochas. — *Aeromagnetic survey* (definição corroborada);
prática da Geoscience Australia (grade TMI 2019): dados são "despiked", o IGRF é removido e
aplica-se correção diurna "para que a resposta seja apenas a das rochas no solo".
https://en.wikipedia.org/wiki/Aeromagnetic_survey ·
https://ecat.ga.gov.au/geonetwork/srv/api/records/7c38a2b2-28e4-4c79-b0ce-517d9861e20d **[verificação a nível de resumo]**

**1.2 — Por isso o limiar "26000–28500 nT = IOCG" do código é incorreto.** Esses valores são
uma impressão digital de **latitude/época do IGRF** (consistente com Carajás, baixa latitude
magnética), **não** uma assinatura de tipo de rocha. A mesma rocha mediria nT completamente
diferente em outra latitude ou ano. A discriminação de alvo se faz pela **amplitude da anomalia
relativa ao background local, comprimento de onda, forma, gradientes e contexto geológico** —
nunca pelo valor escalar bruto. — mesmas fontes; projeção do campo total só é válida para
anomalias < ~5000 nT (corpos de magnetita maciça violam isso).
https://www.nature.com/articles/s41598-020-68494-1

**1.3 — Carajás (~−6°, perto do equador magnético): a Redução ao Polo (RTP) é instável.** Abaixo
de ~15° de inclinação a RTP convencional "não tem uso prático" (estrias N–S, perda de detalhe
E–W). A solução é o **sinal analítico (gradiente total)**, cuja amplitude pica sobre as bordas
das fontes e é **independente da direção de magnetização** — apropriado para baixas latitudes.
— Silva (1986), GEOPHYSICS (RTP como problema inverso, baixa latitude)
https://library.seg.org/doi/10.1190/1.1442096 ; *Analytic Signal vs. Reduction to Pole*
https://www.earthdoc.org/content/journals/10.1071/EG03257 **[verificação a nível de resumo]**

**1.4 — Produtos padrão para detecção de borda/alvo e profundidade:**
- **Sinal analítico / gradiente total** — bordas, direção-independente (ideal p/ Carajás).
- **Tilt derivative** — realça anomalias fracas/profundas; contorno zero marca bordas.
- **Gradiente horizontal total (THG)** — máximos delineiam contatos.
- **Derivada vertical / continuação para cima** — separa fontes rasas vs profundas; regional–residual.
- **Deconvolução de Euler** — profundidade da fonte com índice estrutural (SI). Reid et al.
  (1990), GEOPHYSICS 55:80–91. https://library.seg.org/doi/abs/10.1190/1.1442774
- **Worms (multiscale edge detection)** — Hornby, Boschetti & Horowitz (1999), GJI 137:175–196.
  https://academic.oup.com/gji/article/137/1/175/700677
- Texto-base: **Blakely (1995)**, *Potential Theory in Gravity and Magnetic Applications*, Cambridge UP.

---

## 2. Mineral Prospectivity Mapping (MPM) e uso correto de ML

**2.1 — Métodos consagrados.** *Knowledge-driven*: Weights-of-Evidence (Bayesiano, log-odds;
Agterberg & Bonham-Carter), fuzzy logic e índice-overlay/AHP (quando há poucos/nenhum depósito).
*Data-driven*: regressão logística, random forest, gradient boosting, SVM, redes neurais.
— Rodriguez-Galiano/Carranza et al., *Ore Geology Reviews*
https://www.sciencedirect.com/science/article/abs/pii/S0169136815000037 **[resumo]**

**2.2 — WofE assume independência condicional (IC), violada na prática.** Camadas
geo/geofísicas/geoquímicas são correlacionadas → posteriores enviesados (inflados). Testar IC
(Thiart, Bonham-Carter & Agterberg) ou usar **regressão logística**, que não exige IC conjunta.
https://link.springer.com/content/pdf/10.1007/s11707-016-0595-y.pdf **[resumo]**

**2.3 — MPM é um problema Positive-Unlabeled (PU): célula "sem depósito" é não-rotulada, não
estéril.** Tratar não-rotulado como negativo verdadeiro é erro de princípio e gera *label
crossover* (prospectos verdadeiros rotulados como estéreis). Usar PU learning / one-class,
amostragem informada de negativos, custo-sensível. — Yang/Zuo et al., *Computers & Geosciences*
https://www.sciencedirect.com/science/article/abs/pii/S0098300420306397 ;
https://www.sciencedirect.com/science/article/pii/S0169136824004037

**2.4 — Validação cruzada ESPACIAL é obrigatória.** Autocorrelação espacial torna o CV aleatório
otimista demais e favorece modelos superajustados. Usar blocos/k-means espacial, leave-pair-out
espacial, CV com buffer. — https://link.springer.com/article/10.1007/s10618-018-00607-x ;
https://www.nature.com/articles/s41467-020-18321-y

**2.5 — POR QUE treinar rede neural em rótulos sintéticos auto-gerados é inválido.** Um modelo
supervisionado é limitado pela informação dos rótulos; se os rótulos **são** a saída de uma
heurística sobre as mesmas camadas de evidência, o modelo só aprende a reproduzir a heurística —
tautologia. É **label leakage** / **shortcut learning** ("Clever Hans") na forma mais extrema: o
alvo é derivável das features, a "acurácia" é artefato e nada generaliza. O remédio do campo é
**ground truth independente** (depósitos/ocorrências reais conhecidos).
— *Class Label Representativeness in ML-Based MPM*, *Natural Resources Research* (2025)
https://link.springer.com/article/10.1007/s11053-025-10468-z ;
*Leakage (machine learning)* https://en.wikipedia.org/wiki/Leakage_(machine_learning) **[resumo]** ;
Yousefi, Carranza, Kreuzer et al. (2021), *J. Geochemical Exploration*
https://researchonline.jcu.edu.au/70094/

**2.6 — Reportar honestamente.** Saída = **mapa contínuo de favorabilidade + incerteza**, não
veredito pontual. Performance por **success-rate curve** (% de depósitos conhecidos capturados vs
% de área marcada), ROC/AUC, P–A plot, e eficiência de captura/custo. Agterberg & Bonham-Carter
(2005) alertam: método mais "flexível" pode ter success-ratio maior sem mapa melhor — score alto
não se justifica sozinho. https://link.springer.com/article/10.1007/s11053-005-4674-0

---

## 3. Software open-source de referência (pipeline-alvo)

**Pipeline recomendado (tudo Python/open-source):**
`Harmonica` (processamento campo potencial) → `Verde` (gridding + split espacial) →
`SimPEG` (inversão magnética susceptibilidade + MVI) → `GemPy`/`LoopStructural`+`map2loop`
(geologia 3D implícita), com `geoh5py` (.geoh5) e `QGIS` de interoperabilidade.

- **Harmonica** (Fatiando a Terra) — RTP, continuação para cima, derivadas, **deconvolução de
  Euler** (`harmonica.EulerDeconvolution`), fontes equivalentes. BSD-3.
  https://www.fatiando.org/harmonica/ · https://github.com/fatiando/harmonica
- **Verde** — gridding estilo scikit-learn com **split espacial** (CV espacial).
  https://www.fatiando.org/verde/
- **SimPEG** — inversão de susceptibilidade **e MVI** (magnetização vetorial p/ remanência) +
  inversão conjunta grav+mag (PGI). MIT. https://simpeg.xyz · https://github.com/simpeg/simpeg
- **pyGIMLi** — inversão geofísica (forte em ERT/IP/sísmica). https://www.pygimli.org
- **GemPy** — modelagem geológica 3D implícita com incerteza Bayesiana.
  https://github.com/cgre-aachen/gempy
- **Loop3D**: `LoopStructural`, `map2loop`, `map2model` (este último em C++).
  https://github.com/Loop3D/LoopStructural · https://loop3d.org/map2loop/
- **geoh5py** (Mira Geoscience) — formato aberto `.geoh5` (HDF5).
  https://github.com/MiraGeoscience/geoh5py
- **QGIS** — GIS desktop padrão. https://qgis.org
- **GPlates** — só contexto tectônico regional, **não** é ferramenta de geofísica de depósito.

**Contexto comercial (proprietário/pago):** Oasis montaj/Geosoft (Seequent), Leapfrog Geo,
SKUA-GOCAD (AspenTech), Micromine, Datamine.

---

## 4. Critérios de exploração IOCG / Carajás (assinaturas reais)

**4.1 — IOCG: óxidos de Fe (magnetita e/ou hematita) com Cu–Au, enriquecidos em REE-P-U-Co-Ag;
alteração Na–Ca → K–Fe → hematítica.** Williams et al. (2005), *Econ. Geol. 100th Anniv. Vol.*
https://www.unige.ch/sciences/terre/research/Groups/mineral_resources/archive/pub_archive/williams_ironoxidecoppergold_seg05.pdf
; Barton (2014), *Treatise on Geochemistry*
https://www.geo.arizona.edu/~mdbarton/MDB_papers_pdf/Barton%5B14_IOCGSystems_ToG2-Ch20.pdf

**4.2 — Assinatura geofísica depende da mineralogia magnética.** Magnetita-dominante → **máximo
magnético + máximo gravimétrico** (Salobo, susceptibilidade até ~7,5 SI). Hematita-rica → denso
mas **magneticamente discreto**: um **máximo gravimétrico com mínimo magnético** pode vetorar um
IOCG oculto. — Goodwin et al. (UQ-SMI)
https://smi.uq.edu.au/files/44070/D206_Goodwin_Alteration%20Proxies.pdf ; Cristalino "potencial
oculto" (corpos de sulfeto maciço fracamente magnéticos podem ter sido perdidos), Tavares et al.
(2020), *Ore Geology Reviews*
https://www.sciencedirect.com/science/article/abs/pii/S016913681930695X **[resumo]**

**4.3 — Controle estrutural real (corredores de cisalhamento):** **Cinzento (norte)** — Salobo,
Igarapé Bahia/Alemão, GT-46, Grota Funda; **Carajás (sul / Southern Copper Belt)** — Sossego,
Cristalino, Alvo 118. — síntese *J. South Am. Earth Sci.* (2023)
https://www.sciencedirect.com/science/article/abs/pii/S0895981123000846 ; Sossego: Monteiro,
Xavier et al. (2008), *Ore Geology Reviews*
https://www.sciencedirect.com/science/article/pii/S0169136808000048

**4.4 — Pathfinders geoquímicos:**
- **IOCG:** Cu–Au–Fe–**REE–U–Co–Ag** (± Mo, Bi, Te, P).
  https://eartharxiv.org/repository/object/2413/download/5851/
- **Ouro orogênico:** **Au–As–Sb–W–Te** (± Bi, Mo). Groves et al. (1998, 2003)
  https://www.sciencedirect.com/science/article/abs/pii/S0169136897000127

**4.5 — Sob cobertura, gravimetria e MT (resistividade) tornam-se primários** porque a
magnetometria sozinha não resolve alvos hematíticos/profundos.
https://www.tandfonline.com/doi/full/10.1080/08123985.2024.2378132

**4.6 — Ferro BIF/jaspilito (S11D, N4–N5) é alvo DISTINTO** — anomalia magnética/gravimétrica
**forte, contínua e estratiforme**, não Cu–Au. Deve ser tratado separadamente.
*Brazilian J. Geology* (2020) https://www.scielo.br/j/bjgeo/a/NpgzbY789CYwWBJ5p3tH3Hv/?lang=en

> **Nota:** Serra Pelada é atípico (Au–Pd–Pt, hospedado em sedimento), não ouro orogênico clássico.

---

## 5. Portais de dados e padrões de interoperabilidade

**Status:** a frente de pesquisa dedicada aos portais nacionais e aos **nomes reais das camadas
do GeoSGB/CPRM** foi interrompida por limite de sessão antes de produzir resultado verificado, e
o egress de rede deste ambiente bloqueia `geosgb.sgb.gov.br` / `geosgb.cprm.gov.br` — portanto
**os `typeName` reais do GeoSGB permanecem NÃO verificados**.

**Consequência para o código:** em vez de **chutar** nomes de camada (o que faz hoje, retornando
vazio em silêncio), o cliente deve **descobrir as camadas em runtime via `GetCapabilities`** e
**falhar de forma explícita** quando não encontrar — ver correção em `geological_layers.py`.

**Padrões a adotar:** GeoSciML e EarthResourceML (vocabulários CGI) para geologia/recursos
minerais; OGC **WFS/WCS/WMS** para serviços; OneGeology como agregador. Portais a integrar
(quando o egress permitir): CPRM/SGB GeoSGB (Brasil), Geoscience Australia, USGS Mineral
Resources, GSC/NRCan GDR, BGS GeoIndex, BGR, SGU, SERNAGEOMIN, GSI/Bhukosh, AIST/GSJ, China
Geological Survey.

---

## 6. Plano de correção (priorizado)

**Fase 1 — Remover pseudociência e parar de mentir sobre confiança (feito neste PR):**
1. Novo módulo `mag_anomaly.py`: anomalia residual + sinal analítico + tilt + THG (numpy/scipy),
   com docstrings citando fontes. Substitui o raciocínio por nT absoluto.
2. Reescrever o scoring (`world_model.py`): favorabilidade **relativa** por tipo de alvo
   (IOCG-magnetita / IOCG-hematita / ouro orogênico / BIF) usando anomalia, **não** nT absoluto;
   **incerteza explícita** ligada à completude do dado; decisões deixam de ser "PERFURAR 0.98".
3. Avisos de validade científica em `geo_neural_synapse_v3.py` (rede sintética) e no campo de
   "confiança" — métrica circular não é apresentada como preditiva.
4. `geological_layers.py`: descoberta de camadas em runtime + falha explícita + métrica de
   completude do dado alimentando a incerteza.

**Fase 2 — Ancorar no pipeline de referência (requer dados/decisão):**
5. Adotar **Harmonica** (RTP/derivadas/Euler) e **Verde** (gridding + split espacial).
6. Inversão com **SimPEG** (susceptibilidade + MVI) onde houver malha de dados.
7. Integrar **gravimetria + EM/IP/MT** (a literatura de Carajás mostra que magnetometria sozinha
   é ambígua) — depende de fontes de dado.

**Fase 3 — ML defensável (requer ground truth):**
8. Montar conjunto de **depósitos/ocorrências reais conhecidos** (GeoSGB recursos minerais) como
   positivos; **PU learning**; **CV espacial**; saída = mapa de favorabilidade + incerteza;
   validação por success-rate/P–A. Aposentar a "rede neural sintética".

---

## Limitações desta pesquisa

- Vários editores (ScienceDirect, Springer, SEG, OUP, GA eCat) retornaram **HTTP 403** a coleta
  automatizada; itens assim estão marcados **[resumo]** e foram corroborados por múltiplas fontes.
- Nenhum paper aplicando **RTP especificamente a Carajás** foi recuperado; o vínculo da latitude
  é fisicamente sólido mas não fixado a uma citação única.
- **Nomes de camadas do GeoSGB NÃO verificados** (limite de sessão + egress bloqueado).
- O argumento "rótulo sintético = circular" (§2.5) é dedução sólida a partir da literatura de
  *label leakage* + qualidade de rótulo em MPM, não uma frase única citada literalmente.
</content>
