# Organização do repositório: diagnóstico e proposta

Escopo: estrutura de pastas, empacotamento, documentação, dados, resultados e o uso do Claude Code (harness).
Estado analisado: árvore de trabalho de 2026-11-20, com as suas mudanças de pasta ainda sem commit.
Nada foi movido por esta análise, exceto três correções de referência em `CLAUDE.md` e `agents/Rules/coding-standards.md` (seção 6).

---

## 1. Diagnóstico: por que está confuso

### 1.1 Raiz e documentação
| Achado | Evidência | Efeito |
|---|---|---|
| Quatro documentos longos na raiz com papéis misturados | `README.md` (286 linhas: instalação, pipeline, achados, limitações), `REQ.md` (270), `TASK.md` (354), `ARCHITECTURE.md` (97) | Não se sabe qual é o ponto de entrada nem o que está atual. REQ, TASK e ARCHITECTURE são histórico do início do projeto |
| `.gitignore` contradiz o repositório | `TASK.md` está listado como ignorado e está rastreado; o comentário de LaTeX aponta `docs/*.aux`, mas o paper agora está em `docs/draft/` | Arquivos auxiliares do LaTeX podem ser commitados; a regra de `TASK.md` não faz nada |
| `docs/` mistura quatro tipos de conteúdo | paper (`draft/`), decisões (`adr/`), dados de entrada (`controles/`, `revisao_juridica/`), revisão do orientador | Entradas lidas pelo código (CSV de fatos) vivem em `docs/` |
| O ADR único cresceu | 242 linhas, E1 a E23, D1 a D10, tudo em um arquivo | Difícil achar uma decisão ou um experimento; difícil revisar mudanças |
| O nome `draft` é ambíguo | `docs/draft/` guarda o paper e o relatório do orientador | Não diz se é rascunho ou versão atual |

### 1.2 Código
| Achado | Evidência | Efeito |
|---|---|---|
| Sem empacotamento | não há `pyproject.toml`; só `requirements.txt` | Todo comando precisa de `PYTHONPATH=.`; `quant_suite.py` usa `sys.path.insert` para importar `baselines.py` |
| `compress/` mistura três coisas | biblioteca (`quant`, `gptq`, `stats`, `serialize`, `allocate`), pesquisa (`circuit`, `patching`, `pair_patching`, `contrast`, `sensitivity`, `protect`) e dados (`oab`, `facts`, `mmlu`) em 17 arquivos no mesmo nível | Não se vê o que é núcleo e o que é experimento |
| `scripts/` é plano e de várias gerações | 17 scripts: persona (legado), planner por diagnóstico (legado), circuitos, proteção, baselines, suíte, análise | Não se sabe qual script pertence a qual experimento (E-ID) |
| Preparação duplicada entre scripts | `class_matched_top` em 3 scripts; `evaluate()` em 4; carga de sonda/unidades/estatísticas e `measure()` em 3 | Corrigir um erro exige tocar vários arquivos (ver o plano da Fase 0) |
| `experiments/` tem um único arquivo | `run_smollm.py` | Pasta sem papel claro; compete com `scripts/` |
| Sem lint, formatação nem CI | nenhum `ruff`, `pre-commit` ou workflow | O padrão de código depende de lembrar |
| Testes cobrem só a biblioteca | 156 testes em `tests/`; scripts sem teste | A parte que gera os números não é testada |

### 1.3 Dados e resultados
| Achado | Efeito |
|---|---|
| `outputs/` (78 MB) é ignorado e sem manifesto | Não se sabe qual JSON veio de qual commit, seed, GPU ou comando; parte só existe na maidai |
| O cache do Hugging Face está dentro do repositório (`.cache/`, 162 MB, ignorado) | Ocupa espaço do projeto e confunde a árvore |
| Dados pequenos de entrada em dois lugares | `data/persona/` (novo) e `docs/revisao_juridica`, `docs/controles` |
| `graphify-out/` na raiz | Saída de ferramenta externa no mesmo nível do código |

### 1.4 Harness (uso do Claude Code)
| Achado | Evidência | Efeito |
|---|---|---|
| As regras estão em `agents/Rules/`, que o Claude Code não carrega | O Claude Code lê `CLAUDE.md`, `.claude/rules/`, `.claude/skills/`, `.claude/agents/` e `.claude/settings.json` | O padrão de código só vale se alguém o ler; hoje o `CLAUDE.md` precisa apontar para ele |
| `agents/Design/` está vazia | | Pasta sem conteúdo nem convenção |
| Não existe `.claude/` | | Nenhuma permissão, hook, skill ou subagente do projeto |
| O `coding-standards.md` tem cerca de 190 linhas e tudo vale para qualquer tarefa | | Custo de contexto em toda sessão, mesmo em tarefas que não tocam código |
| Verificação depende de lembrar | | O guia da Anthropic pede um passa-ou-falha executável; hoje é instrução em texto |

---

## 2. Boas práticas consultadas

| Tema | Fonte | O que adotamos |
|---|---|---|
| Layout `src` e empacotamento | PyPA, "src layout vs flat layout" (https://packaging.python.org/discussions/src-layout-vs-flat-layout/) | `pyproject.toml` e instalação editável; o layout `src` obriga a instalar o pacote antes de rodar e evita importar a cópia local por engano. *Li só o resumo da busca, não a página inteira* |
| Estrutura de projeto de ciência de dados | Cookiecutter Data Science v2 (https://cookiecutter-data-science.drivendata.org/) | `pyproject.toml`, pacote de código separado, dados e resultados fora do código. *Li só a descrição da busca; as "opinions" da página não foram lidas* |
| Memória do projeto (`CLAUDE.md`, rules) | Claude Code, "Memory" (https://code.claude.com/docs/en/memory) | Menos de 200 linhas; regras por caminho em `.claude/rules/*.md` com `paths:`; `CLAUDE.local.md` no `.gitignore` |
| Skills | Claude Code, "Skills" (https://code.claude.com/docs/en/skills) | Procedimentos em `.claude/skills/<nome>/SKILL.md` (menos de 500 linhas); `disable-model-invocation: true` para o que tem efeito externo |
| Hooks | Claude Code, "Hooks guide" (https://code.claude.com/docs/en/hooks-guide) | `Stop` e `PreToolUse` com saída 2 para impor o que não pode falhar; hooks não são fronteira de segurança |
| Subagentes | Claude Code, "Subagents" (https://code.claude.com/docs/en/sub-agents) | Revisor independente com ferramentas só de leitura, em `.claude/agents/` |
| Configurações e permissões | Claude Code, "Settings" e "Permissions" (https://code.claude.com/docs/en/settings, https://code.claude.com/docs/en/permissions) | `.claude/settings.json` no git; `.claude/settings.local.json` fora; ordem de avaliação deny, ask, allow |
| Fluxo e verificação | Claude Code, "Best practices" (https://code.claude.com/docs/en/best-practices) | Verificação executável, explorar antes de planejar, `CLAUDE.md` curto |

Os resumos das páginas do Claude Code vieram de um subagente que leu cerca de 40 mil caracteres de cada página; o restante veio resumido por modelo. Itens marcados como "do resumo" no relatório dele (limites de tamanho de subdiretórios, caminhos de memória de subagente, detalhes finos de permissão) devem ser conferidos na fonte antes de virarem regra.
*Não verificados nesta sessão (de memória):* Diátaxis (separar tutorial, guia prático, referência e explicação na documentação) e o formato MADR para ADRs (uma decisão por arquivo).

---

## 3. Estrutura alvo proposta

```text
pretty_duck/
├── README.md                    # visão geral, instalação, "onde achar o quê" (curto)
├── CLAUDE.md                    # curto (< 200 linhas)
├── pyproject.toml               # metadados, dependências, extras, ruff, pytest
├── Makefile                     # test, lint, paper, sync-remote (com dry-run)
├── .claude/
│   ├── settings.json            # permissões e hooks do projeto (no git)
│   ├── rules/                   # regras curtas, por caminho (python.md, experiments.md, docs.md)
│   ├── skills/                  # run-experiment, record-experiment, sync-remote
│   └── agents/                  # reviewer.md (somente leitura)
├── src/compress/                # biblioteca instalável
│   ├── quant/                   # formatos, gptq, protect, serialize, allocate
│   ├── interp/                  # circuit, patching, pair_patching, contrast, sensitivity
│   ├── evalkit/                 # evaluate, métricas (KL, flips), resultados
│   ├── data/                    # oab, facts, mmlu
│   └── experiment.py            # preparação comum (sonda, unidades, estatísticas, measure)
├── experiments/                 # um diretório por experimento ou família
│   ├── circuits/   protection/   quantizer/   sensitivity/   legacy/
│   └── README.md                # tabela E-ID → script → resultado → ADR
├── tests/                       # espelha src/ ; tests/experiments/ para os scripts
├── data/                        # entradas pequenas versionadas
│   ├── persona/   legal_facts/   controls/
├── docs/
│   ├── index.md                 # mapa de leitura
│   ├── paper/                   # method.tex, method.pdf, referências
│   ├── adr/                     # uma decisão por arquivo (0001-..., 0002-...)
│   ├── experiments/             # um registro por experimento (E1 ... E23) + modelo
│   ├── reviews/advisor-2026-11/ # raw/, síntese, plano, referências
│   ├── reports/                 # relatório do orientador (pt-BR)
│   └── history/                 # REQ.md, TASK.md, ARCHITECTURE.md
└── outputs/                     # ignorado: outputs/<experimento>/<execução>/{config,result.json,log}
                                 #  + outputs/MANIFEST.csv (commit, seed, GPU, comando)
```

Princípios por trás da estrutura:
1. **Uma pergunta, um lugar.** "Onde está a decisão sobre X?" é `docs/adr/`. "Qual script gerou o número Y?" é `experiments/README.md`.
2. **Entradas fora de `docs/`.** Tudo que o código lê fica em `data/`.
3. **Núcleo separado do experimento.** `src/` é testado e estável; `experiments/` é fino e só monta, roda e salva.
4. **Resultados com proveniência.** Cada execução grava a configuração, o commit, a seed, a GPU e o dtype (isto também atende à tarefa P3 e à regra 12 do `coding-standards.md`).
5. **Raiz pequena.** Só o que um visitante precisa na primeira olhada.

### Decisões em aberto
- **Nome do pacote.** Manter `compress` (menor custo, nome genérico) ou renomear (por exemplo `pretty_duck`). A renomeação toca todos os imports, a documentação e os comandos da maidai.
- **Layout `src/` ou plano.** O `src/` evita importar o código local por engano e exige instalar, o que elimina o `PYTHONPATH=.`. Custa um `pip install -e .` na maidai.
- **`scripts/` → `experiments/` agora ou depois.** Agora arruma a leitura, mas mexe em muitos caminhos de comando nos documentos; depois reduz o risco enquanto há rodadas ativas.

---

## 4. Proposta para o harness do Claude Code

1. **`CLAUDE.md` curto.** Só comandos, armadilhas, decisões que não estão no código e coisas corrigidas duas vezes. Hoje tem 115 linhas, dentro do limite.
2. **Regras por caminho em `.claude/rules/`.** Dividir o `coding-standards.md`:
   - `python.md` (`paths: ["src/**/*.py", "experiments/**/*.py", "tests/**/*.py"]`): estilo, tipos, logging, erros;
   - `experiments.md` (`paths: ["experiments/**", "docs/adr/**", "docs/experiments/**"]`): reprodutibilidade, controles, métricas, realizações;
   - `docs.md` (`paths: ["docs/**", "README.md"]`): STE, números consistentes.
   O que vale para toda tarefa fica no `CLAUDE.md`.
3. **Skills para procedimentos longos** (carregam só quando usadas):
   - `run-experiment` (`disable-model-invocation: true`): checklist de plano, fumaça, referência, execução com retomada e marcador `*_done`;
   - `record-experiment`: atualiza o ADR, o README e o `method.tex` com os mesmos números;
   - `sync-remote` (`disable-model-invocation: true`): `rsync` de código em modo de simulação primeiro, nunca `outputs/`.
4. **Subagente revisor** em `.claude/agents/reviewer.md`, com `tools: Read, Grep, Glob`, que compara a mudança com o plano e relata só falhas de correção.
5. **Hooks e permissões em `.claude/settings.json`:**
   - hook `Stop` que roda `python -m pytest -q` e retorna saída 2 se falhar (passa-ou-falha executável);
   - hook `PreToolUse` que bloqueia edição em `outputs/` e de arquivos `*.pt` (saída 2);
   - regras `ask` para `ssh`, `scp`, `rsync` e `git push` (ações de saída do computador);
   - regras `deny` para leitura de segredos (`.env`).
   Limite conhecido: regras de Bash casam o texto do comando, não o programa, e hooks podem não ver escritas feitas por comandos de shell. Não são uma fronteira de segurança.
6. **Pasta `agents/`.** O Claude Code não a lê. Opções: mover o conteúdo para `.claude/rules/` (carregamento automático) ou mantê-la como documentação humana e importá-la do `CLAUDE.md` com `@agents/Rules/...` (o import não reduz custo de contexto). `agents/Design/` está vazia: definir o propósito ou apagar.
7. **`CLAUDE.local.md`** para preferências pessoais, listado no `.gitignore`; `.claude/settings.local.json` também fora do git.

---

## 5. Plano de migração em etapas

Cada etapa termina com `python -m pytest -q` verde e, nas etapas de código, com a reprodução de um número salvo. Etapas pequenas, uma por commit.

| Etapa | O que muda | Risco | Validação | Impacto na maidai |
|---|---|---|---|---|
| **0. Congelar** | Commit das mudanças atuais (remoção do Persona, `agents/`, `docs/draft`, revisão); tag do estado | baixo | `git status` limpo | nenhum |
| **1. Higiene sem mover código** | `pyproject.toml` (ruff, pytest, extras), `Makefile`, corrigir `.gitignore` (TASK.md, LaTeX em `docs/draft`), `docs/index.md`, mover REQ/TASK/ARCHITECTURE para `docs/history/`, renomear `docs/draft/` para `docs/paper/` e `docs/reports/` | baixo | testes; compilar o paper | nenhum |
| **2. Harness** | `.claude/` (rules, skills, agents, settings), dividir `coding-standards.md`, decidir o destino de `agents/` | baixo | uma sessão nova carrega as regras (`/context`); o hook `Stop` falha de propósito uma vez para testar | nenhum |
| **3. Dados de entrada** | `docs/revisao_juridica` e `docs/controles` para `data/legal_facts/` e `data/controls/`; atualizar caminhos em scripts, testes, README | médio (caminhos em muitos arquivos) | testes; rodar um script de dados | atualizar com `rsync` de `data/` |
| **4. Código** | Layout `src/`; subpacotes `quant/`, `interp/`, `evalkit/`, `data/`; módulo `experiment.py` com a preparação comum; remover o `sys.path.insert`; `scripts/` para `experiments/` agrupado, com `legacy/` para persona e planner | **alto** (todos os imports) | testes; reproduzir `rtn_uniform` e `gptq_uniform`; teste de identidade da Fase 0 | `pip install -e .` na maidai; **só depois de as rodadas ativas terminarem** |
| **5. Resultados** | Layout `outputs/<exp>/<run>/` e `MANIFEST.csv`; esquema único de resultado com `config` (commit, seed, GPU, dtype); conversor para os JSON antigos | médio | ler os JSON antigos e novos pelo mesmo leitor | nenhum |
| **6. Registro** | Dividir o ADR em uma decisão por arquivo; `docs/experiments/` com modelo e um registro por experimento (E1 a E23); índice | baixo | números iguais ao ADR antigo | nenhum |

### Ordem recomendada e relação com o plano da revisão do orientador
- As etapas 0 a 2 não alteram código nem comandos de execução e podem começar já.
- A etapa 4 se funde com a Fase P3/E0.3 do plano da revisão (salvar logits por questão, métrica KL, esquema único de resultados): mexer na preparação comum uma vez só.
- A etapa 4 não deve começar enquanto houver rodadas na maidai usando os caminhos atuais.

### O que não fazer
- Não renomear o pacote e reorganizar `scripts/` na mesma etapa.
- Não mover arquivos e mudar comportamento no mesmo commit.
- Não apagar `outputs/` nem os JSON antigos.
- Não impor um framework de orquestração (DVC, Hydra) antes de a estrutura simples estar estável (YAGNI); reavaliar depois da etapa 5.

---

## 6. Execução (2026-11-20)

Decisões tomadas: layout `src/` mantendo o nome `compress`; regras divididas em `.claude/rules/`; `scripts/` e o código só depois das rodadas ativas.

| Etapa | Estado | O que foi feito |
|---|---|---|
| 0. Congelar | **Não feita** | Nada foi commitado. As mudanças estão na árvore de trabalho |
| 1. Higiene | Feita | `pyproject.toml` (pytest com `pythonpath`, ruff, extras), `Makefile`, `.gitignore` corrigido, `docs/index.md`, `REQ/TASK/ARCHITECTURE` em `docs/history/`, paper em `docs/paper/`, relatórios em `docs/reports/`, revisão em `docs/reviews/advisor-2026-11/` |
| 2. Harness | Feita | `.claude/rules/` (python, experiments, docs, workflow), 3 skills, subagente `reviewer`, hook `PreToolUse` que bloqueia edição em `outputs/` e `*.pt`, `settings.json`; pasta `agents/` removida (conteúdo dividido sem perda de linhas); guia em `docs/guides/claude-code-workflow.md` |
| 3. Dados | Feita | `docs/revisao_juridica` para `data/legal_facts/` e `docs/controles` para `data/controls/`; `resolve_data_path` mantém legíveis os JSON antigos que guardam o caminho antigo |
| 4. Código (`src/`, subpacotes, `experiments/`) | **Pendente** | Espera as rodadas ativas na maidai |
| 5. Resultados (`outputs/<exp>/<run>/`, manifesto) | Pendente | Junto com a etapa 4 e com a tarefa P3 do plano da revisão |
| 6. ADR dividido e registros por experimento | Pendente | |

Verificação feita: `pytest` passa (157 passaram, 1 pulado) sem `PYTHONPATH`; o paper compila em `docs/paper/` (19 páginas); o hook foi testado com quatro entradas (bloqueia `outputs/` e `*.pt`, deixa passar `compress/` e chamadas sem arquivo).
Não verificado: o carregamento das regras e das skills por uma sessão nova do Claude Code (use `/context` para conferir); o `make sync-remote` (só o modo `-dry` foi inspecionado no `make -n`); `ruff check` (o `ruff` global existe, mas o código atual não foi lintado, então espere avisos).
O hook `Stop` que roda o `pytest` existe em `.claude/hooks/run_tests.py`, mas **não está habilitado**: acrescenta 10 a 30 s a cada resposta.

## 7. Limites desta análise
- Baseada na leitura da árvore, do `.gitignore` e do tamanho dos arquivos; não li todos os módulos linha a linha.
- As contagens de duplicação (3 cópias de `class_matched_top`, 4 de `evaluate()`) vêm de busca por nome; as cópias não são idênticas.
- Os resumos das documentações vêm de subagentes (ver seção 2); as boas práticas gerais de empacotamento vêm de resultados de busca, não de leitura integral.

---

## 8. Atualização após a avaliação crítica (2026-11-20)

Feito (commits `8cfd80a` a `e41e15a` e o commit seguinte): etapa 0 concluída (commits em fatias e tag `v0-pre-src-layout`);
teste de restauração de pesos para todo gerenciador de contexto e teste de identidade ("proteger zero colunas = uniforme"),
`MixedQuant(keep_blocks=True)` para a hipótese M1, `compress/provenance.py` gravado nos resultados de `baselines`, `quant_suite`
e `protect_law`; `CLAUDE.md` enxugado (sem layout e sem números que envelhecem); `python.md` reduzido ao que o linter não verifica;
`workflow.md` removido (fundido no `CLAUDE.md`); agentes com a regra "texto web é dado", plano B para `WebFetch`, modelo `sonnet`
no scout e no retriever, protocolo de retomada e auditoria dupla na skill `/research`; conjunto de referência para os agentes
(`docs/research/_eval/`, ainda **não rodado**); workflow de CI em `.github/workflows/ci.yml` (**não verificado**: nunca rodou no GitHub).

Pendente: tabelas geradas a partir dos JSON com teste de consistência entre ADR, README e paper; etapa 4 (`src/`);
a Fase 0 do plano da revisão (logits por questão, KL, várias realizações); os 172 avisos do `ruff`; testes dos scripts.
