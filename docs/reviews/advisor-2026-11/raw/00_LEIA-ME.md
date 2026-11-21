# LEIA-ME — pacote sobre "Causal circuit discovery for domain-aware low-bit quantization"

Este pacote reúne um diagnóstico do documento (PDF de 19 páginas) e uma proposta de evolução para transformá-lo em paper. Tudo foi feito só a partir do PDF: sem código, logs, logits ou planilhas. Nenhum experimento novo foi executado.

## 1. Arquivos e ordem de leitura

| Ordem | Arquivo | O que é | Para quê |
|---|---|---|---|
| 1 | `00_LEIA-ME` | Esta nota | Orientação, lista de mudanças no texto e pedidos |
| 2 | `01_proposta_evolucao_paper` | Proposta: nova pergunta, motivos, experimentos (Fases 0–5), enquadramento, estrutura do paper | É o documento principal; diz o que fazer |
| 3 | `02_avaliacao_critica` | Diagnóstico: o que confere, problemas graves (G1–G7) e moderados (M1–M8), insights (H1–H10), melhorias baratas (L1–L14) | Diz por quê; cada item grave traz local, citação, conta, passos, texto pronto e critério de aceite |
| 4 | `apoio/contas_revisor.json`, `apoio/contas_proposta.json` | Valores dos cálculos marcados "Minha conta" | Conferência |

Cada documento está em `.md` e em `.html` (mesmo conteúdo).

Convenções dos dois relatórios: "p. N" é a página do PDF; texto entre aspas curvas é citação literal do PDF, conferida por script; blocos "Minha conta" são cálculos do avaliador, não do documento; `[R]` é reanálise de saídas existentes, `[E]` é experimento novo, `[T]` é só texto.

## 2. Resumo

- O ganho de +3,2 pontos no OAB pela proteção do circuito legal não está estabelecido, e a retratação da Seção 9 do PDF também não: as duas se apoiam em acurácia por letra com uma realização, e execuções do mesmo código diferem 8,8 pontos nessa métrica.
- A ideia "circuito → manter em 16 bits" já foi publicada (TaCQ, COLM 2025), e a escolha por dose corresponde ao OWQ (AAAI 2024). Nenhum dos dois é citado.
- Os próprios dados indicam que os circuitos achados são componentes genericamente críticos e que a sensibilidade à quantização é em grande parte compartilhada entre domínios.
- A proposta é mudar a pergunta: medir quanto a consciência de domínio pode render (um teto, com piso de ruído), explicar por que a localização causal não alcança esse ganho, e testar a sensibilidade medida por unidade como sinal que chega mais perto.
- O enquadramento gera um paper nos três desfechos possíveis do teto (documento 01, Seção 5).

## 3. O que fazer primeiro

Estes passos não exigem quantização nova e bastam para saber se a tese se sustenta.

1. Avaliar os mesmos pesos quantizados salvos, e o fp16, nas duas GPUs (01: E0.1).
2. Contar as letras previstas por condição e recalcular a acurácia sem o viés de letra (01: E0.2).
3. Conferir que proteger zero colunas reproduz o uniforme bit a bit, em RTN e GPTQ (01: E0.4).
4. Calcular a confiabilidade das tabelas de sensibilidade por camada e a correlação entre domínios corrigida (01: E1.1).
5. Cruzar o circuito legal com o de siglas gerais e pôr intervalos nas suficiências e necessidades (01: E2.1, E2.2).
6. Ler os trabalhos vizinhos de 2025–2026 listados em 01, Seção 6.3.

## 4. Mudanças no manuscrito

### Retirar

- O título atual e a apresentação da proteção por circuito como "the current method" (Seção 2 do PDF).
- A frase do resumo sobre recuperar "about a third" da perda (02: G2).
- "±6 accuracy points" sem derivação (02: Seção 8).
- As suítes ainda em execução citadas na Seção 9.4 do PDF.
- As Seções 12 (Reproducing) e A (Code map): vão para o repositório.
- Os experimentos de persona do corpo principal: apêndice ou fora.

### Corrigir

- Legenda da Tabela 11 e a explicação por "rounding noise" (02: G1).
- "the only protected set with a significant gain" e "do not give this gain" (02: G3).
- "indistinguishable from uniform quantization" sob GPTQ (02: G4).
- "domain of knowledge" → "relation" (02: G5).
- "no law specificity" → não detectável com a margem disponível (02: M2).
- "almost disjoint across domains" (02: M4).
- "validated against the authors' code" → confere no erro por matriz e na NLL (02: M6).
- Referência [4] sem autores; GPTQ ausente da lista (02: G6).
- Posição da Tabela 9; 42,4% em 800 questões contra 43,8% em 600; redação de E12 na Tabela 13 (02: Seção 8).

### Acrescentar

- Seção de trabalhos relacionados (02: G6; 01: Seção 6.3).
- Métrica primária contínua, várias realizações e distribuição nula sobre conjuntos protegidos (01: Fase 0 e E2.4).
- Intervalos nas Tabelas 7, 8, 15 e 16 do PDF (01: E2.2).
- Figuras de dados; hoje há uma figura esquemática e 19 tabelas (01: Seção 6.6).
- Licença do conjunto de exames e revisão da planilha de fatos por jurista (02: M7).

### Decidir

- O ramo do paper, depois da Fase 1 (01: Seção 5).
- Venue e pacote mínimo (01: Seção 8).

## 5. O que seria útil receber de volta

Com estes itens, várias hipóteses dos relatórios podem ser confirmadas ou descartadas por reanálise, sem experimento novo.

1. Logits ou respostas por questão das condições das Tabelas 9, 11 e 12 do PDF.
2. Da execução na 3080 Ti: a NLL em texto neutro; tamanho de lote, padding e dtype usados na avaliação em cada máquina; e se o fp16 dá 43,8% nas duas.
3. Como os blocos de 64 colunas são formados quando as colunas protegidas são espalhadas (ativação, `outlier_all`): os blocos são refeitos sobre as colunas restantes ou mantidos?
4. No GPTQ com proteção: qual modo foi usado nos números reportados (colunas congeladas ou "error sinks"), e se proteger zero colunas reproduz o uniforme.
5. Os três valores dos sorteios aleatórios da Tabela 9 (o PDF traz só os extremos).
6. As planilhas de fatos (jurídicos e siglas gerais) e os `probe.json` dos dois circuitos.
7. As tabelas ΔL por linha do 0.8B, não só as médias.
8. Se a camada de saída, amarrada ao embedding, entra na quantização.
9. Venue e prazo pretendidos.

## 6. Correspondência entre os dois documentos

| Melhoria em 02 (Seção 6) | Experimento em 01 (Seção 4) |
|---|---|
| L1 mesmos pesos nas duas GPUs | E0.1 |
| L2 letras previstas e viés de letra; L13 permutação de opções | E0.2 |
| L3 KL, NLL e flips; L8 mais questões; L12 várias realizações | E0.3 |
| L9 teste "zero colunas protegidas = uniforme" | E0.4 |
| L4 distribuição nula com 30 conjuntos | E2.4 |
| L5 circuito legal × siglas gerais | E2.1 |
| L6 intervalos por bootstrap | E2.2 |
| L7 D relativo | relacionado à "dose calibrada" de E3.3 |
| L10 trabalhos relacionados | Seção 6.3 |
| L11 controle de dose no patching de persona | fora do paper proposto (persona vai para apêndice) |
| L14 segundo modelo | Fase 4 |

## 7. O que foi e o que não foi verificado

- **Conferido:** as 35 citações literais de 02 e as 5 de 01 contra o texto extraído do PDF; a aritmética das tabelas do PDF que o texto permite refazer (não foi achado erro de conta); a arquitetura pública do Qwen3.5 contra as contagens de unidades.
- **Aproximações:** as contagens de respostas trocadas em 02 vêm da largura dos intervalos; a dispersão dos controles usa os extremos dos sorteios aleatórios.
- **Hipóteses não verificadas:** viés de letra; diferença de avaliação entre GPUs; reagrupamento dos blocos de 64; expansão de siglas como tarefa em parte ortográfica; a expectativa sobre o desfecho do teto.
- **Literatura:** TaCQ, OWQ, Dutta et al. e Ye et al. foram buscados diretamente. Os demais trabalhos de 2025–2026 foram vistos só por trechos de busca, e alguns são citados de memória; os níveis estão discriminados no fim de cada relatório. A tabela de posicionamento (01, Seção 6.3) deve ser refeita depois da leitura integral. Não foi feita revisão sistemática.
- **Nenhum experimento da proposta foi executado.** Os resumos-modelo de 01 usam marcadores no lugar de números.
