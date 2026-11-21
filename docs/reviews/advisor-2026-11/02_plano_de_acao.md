# Plano de ação para cumprir as recomendações da revisão

Base: `agentic-eval/` (documentos 00, 01 e 02) e `01_sintese_didatica.md`. Códigos: **G/M** = problema, **H** = insight, **L** = melhoria barata, **E** = experimento da proposta.
Custo: **[T]** só texto, **[R]** reanálise do que já existe, **[E]** experimento novo (GPU), **[L]** leitura. "Nós" = Vitor e este assistente; "GPU" = máquina maidai (3090 e 3080 Ti).
Nada abaixo foi executado ainda. As estimativas de tempo são aproximadas e não incluem a espera por revisão do jurista nem por resposta do orientador.

---

## 0. Princípios de ordem

1. **Consertar a régua antes de medir** (Fase 0): nenhuma comparação posterior vale se a métrica oscila mais do que o efeito.
2. **Começar por reanálise**: o que não exige GPU nova roda primeiro e já diz se a tese se sustenta.
3. **Uma decisão de ramo** (A, B ou C) depois da Fase 1; as Fases 2 e 3 servem os três ramos.
4. **Pausar o que usa a métrica antiga.** As suítes de `scripts/quant_suite.py` e a rodada de `scripts/baselines.py` estão na acurácia de letra; seus números só são aproveitáveis pela NLL neutra e pelo `law_logp`.

## 1. Visão geral

| Fase | Objetivo | Tarefas | Custo principal | Depende de | Responsável |
|---|---|---|---|---|---|
| **P** | Higiene e pedidos do orientador | P1 a P5 | [T] [L] | nada | nós |
| **0** | Arnês de avaliação confiável | E0.1 a E0.4 | [E] pouco | P3 | nós + GPU |
| **1** | O teto do ganho de domínio | E1.1 a E1.3 | [R] depois [E] | Fase 0 (E1.1 pode começar antes) | nós + GPU |
| **2** | O que a localização causal acha | E2.1 a E2.5 | [R] e [E] | Fase 0 | nós + GPU |
| **3** | Sensibilidade medida por unidade | E3.1 a E3.5 | [E] grande | Fases 0 e 1 | nós + GPU |
| **4** | Generalização | modelos, quantizadores | [E] grande | Fases 1 a 3 | nós + GPU |
| **5** | TaCQ no mesmo arnês (opcional) | F5 | [E] com gradiente | ramo B | nós + GPU |
| **M** | Manuscrito | M1 a M6 | [T] | resultados de cada fase | nós |

Ponto de decisão: **fim da Fase 1** (escolha do ramo).

---

## 2. Fase P: higiene e pedidos (começar já, sem GPU)

- [ ] **P1. Suspender as suítes em andamento na maidai** (`quant_suite`, `baselines`) e guardar os JSONs atuais (`outputs/suiteA.json`, `baselines_4b.json`). Motivo: usam acurácia de letra. *Conferir o estado na maidai (a resolução de nome falhou nesta sessão).* [T]
- [ ] **P2. Responder os 9 pedidos da revisão** (`00_LEIA-ME`, seção 5). Seis já têm resposta no repositório (tabela da seção 8 da síntese); faltam: logits por questão (não salvos, exige E0.2), NLL neutra da 3080 Ti (não registrada), tabelas ΔL por linha do 0.8B (conferir em `outputs/sensitivity.json`), planilhas e `probe.json` dos dois circuitos (já existem, só enviar) e **venue e prazo**. [T]
- [ ] **P3. Salvar logits por questão** nas avaliações: guardar as quatro NLL por letra (`option_nll`) que `evaluate_mmlu_rows` já calcula, e não só acerto/erro. É pré-requisito de E0.2, KL, flips e acurácia sem viés. Alterar `measure()` em `scripts/baselines.py` e `scripts/quant_suite.py`. [E pequeno: código]
- [ ] **P4. Leitura dos vizinhos de 2025–2026** (`referencias.csv`), nesta ordem: arXiv:2609.26241, 2611.11079, 2607.00908 (podem já reportar um teto); depois 2606.19558, 2610.00694 (métricas de avaliação); 2504.07389 (TaCQ) e 2306.02272 (OWQ) na íntegra; por fim 2511.06516, 2607.17733, 2609.04098, 2607.02893. Registrar, para cada um, o que reporta e onde nosso trabalho se diferencia. Critério: tabela de posicionamento refeita. [L]
- [ ] **P5. Decisões a tomar com o orientador:** venue e prazo; se o 9B entra; quem revisa a planilha jurídica; licença do conjunto de exames. [T]

## 3. Fase 0: arnês de avaliação confiável

**Critério de decisão da fase:** existe uma métrica primária cujo desvio entre realizações é pelo menos 3 vezes menor que o efeito do int4 uniforme. Se não existir em 4 bits, o programa passa para 3 bits.

- [ ] **E0.4. Teste de identidade** (primeiro, porque é barato e pode explicar a queda de 9 a 13 pontos com proteção aleatória sob GPTQ): proteger um conjunto vazio pelo caminho "protegido" reproduz o uniforme bit a bit, em RTN e GPTQ, para unidades contíguas e para colunas espalhadas. Estender `tests/test_protect.py` e `tests/test_gptq.py`. Tratar também M1: manter os blocos originais de 64 colunas ao excluir colunas espalhadas (hoje são refeitos). **Aceite:** teste passa; se falhar, corrigir antes de qualquer número de GPTQ com proteção. [R]
- [ ] **E0.1. Mesmos pesos, duas GPUs:** salvar o state dict quantizado (GPTQ int4) na 3090, carregá-lo na 3080 Ti e vice-versa; avaliar as 600 questões; repetir para o fp16. O fp16 já dá 43,8% nas duas; falta o caso quantizado. **Aceite:** mesma acurácia ±1 questão para o mesmo arquivo. Se não, o problema é da avaliação. [E minutos]
- [ ] **E0.2. Letras previstas e acurácia sem viés:** com os logits de P3, histograma de letras previstas e do gabarito por condição; acurácia depois de subtrair o log-prior por letra (estimado nas ~1.600 questões do OAB fora das 600); acurácia média sobre as 4 rotações das alternativas. **Aceite:** tabela por condição; 27,2% explicado ou não. [E, reanálise se os logits existirem]
- [ ] **E0.3. Métricas e realizações:**
  - métrica primária: KL(p_fp16 ‖ p_q) por token em texto de domínio fora da calibração (e sobre as quatro letras nas múltiplas escolhas);
  - secundárias: NLL, diferença de logit em fatos, flips (melhorou/piorou), acurácia com rotação;
  - realizações: no GPTQ, R ≥ 5 sorteios de calibração com semente registrada; no RTN, R ≥ 5 deslocamentos aleatórios dos blocos de 64 colunas;
  - análise: decomposição da variância (questões e realizações), comparação pareada por realização;
  - usar as ~2.000 questões do OAB fora da calibração (erro-padrão por modelo cai de ~2,0 para ~1,1).
  Acrescentar `--reps` e a métrica KL a `scripts/baselines.py` e `quant_suite.py` (a suíte `protect` já tem réplicas de calibração). Tornar a acumulação da Hessiana determinística (fp64, sem TF32). **Aceite:** desvio entre realizações medido e reportado. [E]

## 4. Fase 1: o teto do ganho de domínio

- [ ] **E1.1. Confiabilidade e correlação corrigida** [R]: com as tabelas ΔL do 0.8B (64 linhas por domínio), dividir as linhas em duas metades muitas vezes, aplicar Spearman–Brown para a confiabilidade de cada domínio e formato e calcular ρ = r_xy / √(r_xx·r_yy) entre pares de domínios, com bootstrap sobre linhas. **Aceite:** tabela de ρ bruta e corrigida. *Pode começar já, sem GPU, assim que as linhas por camada estiverem acessíveis.*
- [ ] **E1.2. Plano-oráculo × plano genérico com piso de ruído:** para o domínio-alvo T e orçamentos de +1, +2, +5 e +10% de bytes: plano_T (tabela de T, metade A), plano_G (média dos outros domínios) e plano_T′ (tabela de T, metade B); avaliar em linhas de T fora de qualquer tabela. Ganho G = [dano(G) − dano(T)] / dano(uniforme); piso G₀ com T′ no lugar de G; teto = G − G₀. Primeiro passo [R] (ganho previsto pela aditividade); segundo [E]. Reaproveitar `scripts/alloc_e2e.py` e `compress/sensitivity.py`. [R → E]
- [ ] **E1.3. Onde o teto aparece:** repetir a tabela medida em int3 e int2; acrescentar domínios distantes (texto jurídico em português: enunciados do OAB fora da avaliação e legislação; texto geral em português; código); repetir no 4B. **Aceite:** tabela teto(bits, distância) com intervalos, que escolhe o ramo. [E]
- **Decisão do ramo (A, B ou C)** com o orientador. Registrar no ADR (nova decisão D11).

## 5. Fase 2: o que a localização causal acha

- [ ] **E2.1. Circuito jurídico × siglas gerais** [R]: necessidade e suficiência cruzadas, Jaccard em vários k, bootstrap sobre pares; contar primeiros tokens distintos nos 26 fatos jurídicos e a fração de pares cuja resposta começa com a inicial da sigla. Os dois `probe.json` existem. **Aceite:** tabela 2×2 com intervalos; necessidade cruzada > 0,7 ⇒ chamar de "circuito de expansão de sigla".
- [ ] **E2.2. Intervalos por bootstrap sobre pares** nas Tabelas 7, 8, 15 e 16 [R] (M4, M8).
- [ ] **E2.3. Criticidade genérica dos circuitos** [E]: quantizar só as colunas de cada circuito (k = 16, 32, 64) e medir KL em texto neutro e em texto do domínio; comparar com aleatórios de mesma classe, profundidade e dose; razão de especificidade KL(domínio)/KL(neutro).
- [ ] **E2.4. Proteção com distribuição nula** [E]: em RTN e GPTQ, 4 e 3 bits, dois tamanhos: circuito, dose estilo OWQ, dose × suficiência e 30 conjuntos aleatórios de mesma classe e profundidade; R ≥ 5; métrica primária da Fase 0; p de permutação. (Resolve G3 e G4.)
- [ ] **E2.5. Pares entre domínios** [E, opcional]: mesmo template, sujeito jurídico contra sujeito não jurídico.
- [ ] **M5 (reanálise barata, junto):** D relativo = D / energia de saída da camada (`(W² · V).sum()`), Spearman contra ΔL, a partir de `stats.pt` e das tabelas existentes. Se recuperar a ordenação, a "tabela medida" deixa de ser necessária. [R]

## 6. Fase 3: sensibilidade medida por unidade

- [ ] **E3.1. Varredura:** para cada unidade u (cabeça ou grupo de 64 neurônios), s_u(d) = E_x KL(p_fp16 ‖ p_{Q_u}), com um hook que subtrai E_u·h_u da saída (uma unidade diferente por sequência do lote; reaproveita a varredura em lote de `compress/circuit.py`). 0.8B: 1.680 unidades, int3, 64 linhas por domínio. Remédios para o sinal pequeno: KL em vez de diferença de NLL, int3 ou int2, varredura hierárquica (camadas primeiro). **Aceite:** confiabilidade split-half da tabela por unidade reportada.
- [ ] **E3.2. Aditividade por unidade:** planos aleatórios de 8, 32 e 128 unidades, soma dos efeitos individuais contra efeito conjunto, em texto igual e separado.
- [ ] **E3.3. Comparação em bytes iguais** (R ≥ 5, métrica primária): aleatório, D por camada, dose por unidade (OWQ-style), camadas medidas, circuito, dose × suficiência, dose calibrada (escala medida da camada × dose da unidade), **unidades medidas**, e opcionalmente Fisher diagonal por unidade. **Aceite:** "unidades medidas" supera "camadas medidas" e "dose" em ao menos dois orçamentos, com intervalo sobre realizações que exclui zero; senão, a peça construtiva passa a ser a "dose calibrada" ou sai do paper.
- [ ] **E3.4. Versão por domínio** (plano por unidades de T contra genérico, com piso de ruído).
- [ ] **E3.5. Custo:** tempo e memória por seletor no 0.8B e no 4B.

## 7. Fases 4 e 5

- [ ] **Fase 4:** 4B e 9B da mesma família (o 9B não coube na GPU compartilhada; depende de liberar a 3090) e ao menos um transformer convencional de 1 a 4 bilhões de outra família; RTN e GPTQ com clipping em 4, 3 e 2 bits; repetir Fase 1, E2.4 e E3.3. **Aceite:** sinal do teto e ordem dos seletores mantidos em ao menos 2 de 3 modelos.
- [ ] **Fase 5 (só ramo B):** código público do TaCQ (`https://github.com/The-Inscrutable-X/TACQ`) no mesmo arnês, com R realizações e controles de mesmo tamanho.

---

## 8. Manuscrito (reescrita em paralelo às fases)

Mapeamento da lista "Retirar / Corrigir / Acrescentar / Decidir" do `00_LEIA-ME` para tarefas:

| Tarefa | O que | Ligado a | Quando |
|---|---|---|---|
| **M1 [T]** | Retirar: título atual; "current method" para a proteção por circuito; frase "about a third"; "±6 accuracy points" sem derivação; as suítes em execução citadas na Seção 9.4; as Seções 12 e A (vão para o repositório); persona do corpo principal (apêndice) | G2, G7 | já (pode ser feito antes dos novos resultados) |
| **M2 [T]** | Corrigir: legenda da Tabela 11 e a explicação por "rounding noise" (G1); "the only protected set with a significant gain" e "do not give this gain" (G3); "indistinguishable from uniform" sob GPTQ (G4); "domain of knowledge" → "relation" (G5); "no law specificity" → "não detectável com a margem" (M2); "almost disjoint" (M4); "validated against" → "confere no erro por matriz e na NLL" (M6); referência [4] sem autores e GPTQ ausente (G6); posição da Tabela 9, 42,4% (800 q.) vs 43,8% (600 q.), redação de E12 na Tabela 13 | G1 a G6, M2, M4, M6 | já |
| **M3 [T]** | Trabalhos relacionados, a partir de `referencias.csv` e da tabela da seção 6.3 da proposta; renomear dose/`outlier_all` como "OWQ-style"; dizer que o clipping é opção existente do GPTQ/AWQ | G6, L10 | depois de P4 |
| **M4 [R/E]** | Acrescentar: métrica primária contínua, várias realizações, distribuição nula sobre conjuntos; intervalos nas Tabelas 7, 8, 15, 16; figuras de dados (teto, histograma dos 30 aleatórios, dano por orçamento, ΔNLL × acurácia, aditividade) | Fases 0 a 3 | à medida que saem |
| **M5 [T]** | Licença do conjunto de exames; revisão da planilha de fatos por jurista | M7 | antes da submissão |
| **M6 [T]** | Reestruturar em 9 páginas (introdução com a figura do teto; relacionados; protocolo; teto; localização; sensibilidade por unidade; "como um ganho aparece e some"; limitações), formato da conferência-alvo, checklist, autoria | G7 | depois da decisão de ramo e de venue |

Também atualizar: `docs/adr/0001-...md` (novas decisões e experimentos), `README.md` (Findings) e `docs/relatorio_orientador` (ainda sem os resultados da 4B).

## 9. Sequência sugerida

| Etapa | Itens | Pré-requisito | Resultado |
|---|---|---|---|
| 1 | P1, P2, P3, M1, M2 | nenhum | texto sem as afirmações indefensáveis; logits passam a ser salvos |
| 2 | E0.4, E0.1, E0.2 (paralelo a P4) | P3 | arnês diagnosticado: sabemos se G1 é falha de controle ou ruído real |
| 3 | E1.1, E1.2 (previsto), E2.1, E2.2, M5 (D relativo) | tabelas e `probe.json` existentes | resposta barata à pergunta "a sensibilidade depende do domínio?" |
| 4 | E0.3 | etapa 2 | métrica primária e R realizações |
| 5 | E1.2 (medido), E1.3 | etapa 4 | **teto por bits e distância ⇒ decisão do ramo** |
| 6 | E2.3, E2.4, M3, M4 (parcial) | etapa 5 | resultado negativo fechado, com distribuição nula |
| 7 | E3.1 a E3.5 | etapa 5 | peça construtiva |
| 8 | Fases 4 e 5, M6 | etapas 5 a 7 | paper completo |

As etapas 1 a 3 não exigem quantização nova além da já feita e bastam para saber se a tese se sustenta (como diz o `00_LEIA-ME`).

## 10. Riscos e planos B

| Risco | Sinal | Plano B |
|---|---|---|
| Um trabalho de 2026 já reporta um teto equivalente | leitura de P4 | o diferencial passa a ser o piso de ruído, a unidade e os controles causais |
| Sem métrica estável em 4 bits | desvio entre realizações do tamanho do efeito | operar em 3 bits e reportar 4 bits como "não mensurável neste arnês" |
| Efeito por unidade abaixo do ruído | split-half baixa em E3.1 | hierarquia por camada, int2, blocos de unidades, "dose calibrada" |
| Não aditividade por unidade | razão longe de 1 em E3.2 | alocar por blocos; reportar como limite |
| Teto alto e nenhum seletor o alcança | Fase 3 muito abaixo do oráculo | vira resultado; motiva a referência com gradiente (Fase 5) |
| Erro de implementação das colunas protegidas | E0.4 falha | corrigir antes de qualquer número de GPTQ com proteção |
| GPU compartilhada (3090 e 3080 Ti) limita o 9B e as réplicas | falta de memória | priorizar 4B; réplicas em fila; reaproveitar logits salvos |

## 11. Critérios de pronto, por entrega

- **Arnês:** mesma acurácia ±1 questão para o mesmo arquivo de pesos; tabela por condição com NLL, KL, flips e histograma de letras; frase sobre ruído cita variância estimada com R ≥ 5.
- **Teto:** curva teto(bits) por distância de domínio, com faixa do piso de ruído, e tabela de ρ corrigida.
- **Negativo:** p de permutação contra ≥ 30 conjuntos; tabela jurídico × siglas gerais com intervalos.
- **Construtivo:** figura de dano recuperado por orçamento, com intervalo sobre realizações, e tabela de custo.
- **Texto:** nenhuma frase expressa ganho como fração de uma perda de acurácia de letra com uma realização; "significant" só aparece associada ao teste de permutação; GPTQ, AWQ, OWQ, SpQR, TaCQ, Dutta et al., Hase et al. e ao menos duas referências de patching citados e diferenciados.
