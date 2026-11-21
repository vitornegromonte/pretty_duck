# Proposta de evolução — de "proteger o circuito" para "quanto a consciência de domínio pode render, e que sinal entrega isso"

Documento de origem: "Causal circuit discovery for domain-aware low-bit quantization" (PDF de 19 páginas; Qwen3.5-0.8B e 4B; OAB e MMMLU PT-BR).

Este relatório contém só a proposta de evolução: os motivos, os experimentos e o enquadramento do paper. O diagnóstico que motiva cada ponto está no documento `02_avaliacao_critica`.

Convenções:

- "p. N" é a página do PDF de origem. Texto entre aspas curvas é citação literal desse PDF, conferida por script.
- Blocos **Minha conta** são cálculos meus.
- Custo: **[R]** reanálise de saídas que já existem; **[E]** experimento novo (todos forward-only, salvo indicação).
- **Nenhum experimento descrito aqui foi executado.** Onde aparece um resultado esperado, é hipótese. Os resumos-modelo trazem `[X]` no lugar de números.
- Trabalhos vizinhos: salvo indicação, só vi o trecho que a busca retornou. A Seção 11 separa o que foi lido do que não foi.

---

## 0. A proposta em uma página

**Mudança de pergunta.** O estudo original pergunta "proteger o circuito do domínio ajuda?". A resposta dos próprios dados é não, e a versão positiva dessa ideia já foi publicada (TaCQ). A pergunta que os dados sustentam, e que não encontrei respondida, é anterior:

> Quanto um quantizador pode ganhar por saber o domínio de uso, e que sinal forward-only chega mais perto desse limite?

**Três peças, uma história.**

1. **O teto.** Medir o ganho máximo de uma alocação feita para o domínio-alvo sobre uma alocação genérica, em bytes iguais, com piso de ruído explícito, em 4, 3 e 2 bits. Começa como reanálise das tabelas que já existem.
2. **Por que a localização causal não chega lá.** O patching de pares mínimos acha componentes genericamente críticos, não componentes do domínio. Isso já está nos dados; falta fechar com dois testes baratos.
3. **O que chega mais perto.** Sensibilidade medida por unidade (cabeça, grupo de 64 neurônios): quantizar cada unidade sozinha e medir a distância ao modelo original. Usa a infraestrutura de intervenção em lote que o estudo já tem.

**Pré-condição.** Nada disso se sustenta sem um arnês de avaliação confiável. Hoje, execuções que o texto chama de equivalentes diferem 8,8 pontos no OAB. A Fase 0 resolve isso antes de qualquer outra coisa.

**Por que este enquadramento é o melhor.** Ele gera um paper publicável nos três desfechos possíveis do teto (Seção 5); transforma os resultados negativos em um limite quantitativo; e é a única parte do espaço que as buscas não mostraram ocupada, num campo que encheu em 2025–2026.

**Mínimo publicável.** Fases 0, 1 e 2 em dois modelos: workshop, TMLR ou venue nacional. Com as Fases 3 e 4: tentativa em trilha principal.

---

## 1. Ponto de partida

### 1.1 O que o estudo já tem e deve ser preservado

| Ativo | Onde | Por que importa para a proposta |
|---|---|---|
| Intervenção em lote por unidade (cada sequência do lote com uma unidade diferente alterada) | p. 2, Seção 2.2; p. 19, B.7 (68.096 neurônios por domínio) | É o motor da Fase 3 |
| Tabela de sensibilidade medida por camada, por domínio e por formato (186 camadas × 4 domínios × 3 formatos) | p. 15–16, B.1 | É a matéria-prima da Fase 1 |
| Evidência de aditividade dos efeitos por camada | p. 16 | Justifica o plano-oráculo |
| Controles de mesmo tamanho, mesma classe e mesma dose | Tabelas 9, 17, 18 | São o diferencial metodológico |
| Bytes exatos por fórmula, conferidos contra tensores | p. 4 | Permite comparar em bytes iguais |
| GPTQ próprio conferido contra o código dos autores no erro por matriz e na NLL | p. 10 | Quantizador de referência |
| Circuitos por pares mínimos com validação em fatos não usados | Tabelas 7, 15 | Objeto da Fase 2 |
| Planilha de fatos jurídicos e controle de siglas gerais no mesmo formato | p. 6 | Teste decisivo da Fase 2 |

### 1.2 O que não se sustenta e por isso sai do centro

- O ganho de +3,2 pontos da proteção do circuito legal (Tabela 9) vem de uma realização, com intervalo sobre questões. Quatro dos oito conjuntos-controle com intervalo também excluem zero, dois em cada direção.
- A retratação sob GPTQ (Seção 9.2 do PDF) compara com um uniforme de 49,5%, que está acima do fp16 (43,8%) e 8,8 pontos acima de outra execução do mesmo código.
- A acurácia por letra não acompanha o dano: perdas de NLL de 0,063 e 0,066 nat correspondem a 35,0% e 43,3% no OAB.

---

## 2. Por que mudar a pergunta: seis argumentos

### Argumento 1. A ideia positiva já foi publicada, com gradiente

O TaCQ (Xiao, Sung, Stengel-Eskin e Bansal, COLM 2025, arXiv:2504.07389) mantém em precisão alta um conjunto pequeno de pesos identificado como circuito da tarefa e reporta ganhos em 2–3 bits. Um paper cuja tese seja "circuito → proteger" será lido como variante sem gradiente de um método existente, com resultado mais fraco.

Há um detalhe no resumo do TaCQ que favorece a pergunta nova: os autores relatam um ganho também sem condicionar em tarefa específica. Isso sugere que parte do que o método captura é importância genérica, não de domínio. É a mesma leitura que os dados deste estudo permitem.

### Argumento 2. Os dados do estudo dizem que a sensibilidade é quase toda compartilhada entre domínios

- Erros por bloco de matemática e direito: correlação de postos 0,995 (p. 6).
- “Plans barely depend on the target domain” (p. 6).
- “Sensitivity is largely domain-shared” (p. 16): ΔL de matemática e de direito correlacionam 0,58 entre camadas em int4.
- A proteção do circuito legal ajuda direito e não-direito por igual no MMMLU (Tabela 10).

**Minha conta.** O 0,58 é uma correlação entre duas medidas ruidosas: só 44% das células camada-domínio em int4 passam de dois erros-padrão (p. 15). A correlação verdadeira é ρ = r_obs / √(r_xx·r_yy), com r_xx e r_yy as confiabilidades das duas tabelas. Se a confiabilidade for 0,6, ρ ≈ 0,97; se for 0,8, ρ ≈ 0,73; se for 0,9, ρ ≈ 0,64. Não conheço a confiabilidade. A resposta à pergunta "a sensibilidade depende do domínio?" está, portanto, a uma reanálise de distância, e os dois desfechos são informativos.

### Argumento 3. O que o patching acha são componentes genericamente críticos

- Quantizar um circuito eleva a NLL de texto neutro muito mais que conjuntos de mesma dose (+4,58 contra +0,20 no exemplo da p. 17). O próprio texto conclui: “so they are generally critical components, not recall-specific ones”.
- Um núcleo de 15 unidades compartilhadas tem necessidade 0,66 a 0,78 nos quatro domínios do 4B (p. 9).
- Há uma razão de desenho: o par mínimo troca um sujeito por outro do mesmo domínio. Tudo o que é comum ao domínio fica igual nas duas execuções e não pode aparecer.

Consequência: o circuito de recordação não é o seletor certo para um objetivo de domínio. Isso não é um defeito de execução; é o que o método mede.

### Argumento 4. O sinal que funciona nos dados é o medido, e ele tem uma lacuna de granularidade

Em bytes iguais (Tabela 19, +10% de bytes), a fração do dano recuperada é: camadas medidas 61%; dose por unidade 36%; circuito 24%; D por camada 5%; aleatório 8%. E o texto observa: “Unit-level dose does reasonably, which suggests the diagonal error is informative per column and fails when compared across layers.”

Ou seja: a medição direta vence, mas só existe por camada; a dose tem granularidade fina, mas não é comparável entre camadas. Uma medição direta por unidade junta as duas coisas. O estudo já fez o equivalente computacional (varredura de dezenas de milhares de neurônios).

Há ainda um fato que o texto trata como curiosidade: as colunas do circuito e as de maior dose quase não se sobrepõem (1 de 4.224 e 19 de 15.040; p. 11). São os dois fatores de um produto: o dano de quantizar um peso é, em primeira ordem, quanto o peso se move vezes quanto a saída depende dele. O estudo mediu os dois separadamente e os comparou como rivais.

### Argumento 5. O campo encheu, e a parte vazia é a do limite

As buscas retornaram, só de 2025–2026, ao menos seis trabalhos sobre quantização consciente de tarefa ou de domínio (Seção 6.3). Nenhum dos trechos que vi reporta um limite superior do ganho de domínio com piso de ruído. Dois deles trazem indícios coerentes com um limite baixo em 4 bits:

- No TAQ (arXiv:2511.06516), 87,5% das atribuições por camada não mudam quando a tarefa de calibração muda.
- Em arXiv:2609.26241 (finanças), a escolha do corpus de calibração importa pouco quando a quantização preserva a capacidade, e os autores recomendam medir o dano antes de construir calibração especializada.

Um limite medido organiza esses indícios e os resultados positivos em bits baixos numa mesma figura.

### Argumento 6. O arnês de avaliação atual fabrica e desfaz efeitos

**Minha conta.** Com 600 questões independentes, o desvio-padrão da diferença de acurácia entre duas realizações equivalentes é no máximo 2,89 pontos. A diferença observada entre duas execuções do mesmo código é 8,8 pontos. Com desvio de ~6 pontos por modelo, detectar 3 pontos com 80% de poder exigiria cerca de 63 realizações por braço. Em acurácia por letra, o programa é inviável; em distância ao fp16 (KL, NLL), que varia centésimos de nat entre execuções, é viável.

Isso não é só um problema a consertar. Mostrar como um ganho de +3,2 [+1,5; +4,8] aparece e some sob controles é uma contribuição de método que acompanha bem as outras três peças.

---

## 3. Pergunta, hipóteses e o que cada uma decide

**Pergunta central.** Quanto um quantizador pode ganhar por saber o domínio de uso, e que sinal forward-only chega mais perto desse limite?

| # | Hipótese | Teste | Se confirmada | Se refutada |
|---|---|---|---|---|
| H1 | A diferença entre execuções "equivalentes" no OAB tem um fator sistemático identificável (viés de letra ou avaliação), não é ruído por questão | Fase 0 | Arnês corrigido; possível seção sobre o mecanismo | O ruído é real: só métricas contínuas e várias realizações |
| H2 | Em 4 bits, o ganho de uma alocação por domínio sobre a genérica não passa do piso de ruído; em 3 e 2 bits, passa | Fase 1 | Ramo A do paper (Seção 5) | Ramo B (teto alto já em 4 bits) ou C (nulo em todos) |
| H3 | Circuitos de recordação são componentes genéricos: o circuito legal e o de siglas gerais são, em boa parte, o mesmo, e protegê-los não supera dose de mesmo tamanho | Fase 2 | Resultado negativo fechado, com explicação | A proteção por circuito volta a ser candidata; rever |
| H4 | Sensibilidade medida por unidade supera, em bytes iguais, a medida por camada e a dose por unidade | Fase 3 | Peça construtiva do paper | A granularidade fina não paga; fica o teto e o negativo |

---

## 4. Programa experimental

Ordem por dependência. Cada fase tem um critério de decisão; a seguinte só começa depois dele.

### Fase 0 — Arnês de avaliação confiável

**Por quê.** Todas as comparações posteriores dependem de uma medida de dano que não oscile mais que o efeito procurado.

#### E0.1 Mesmos pesos, duas máquinas [E, minutos]

- **Desenho.** Salvar o state dict quantizado (GPTQ int4) em uma GPU; carregá-lo na outra; avaliar as mesmas 600 questões com lote = 1 e o mesmo dtype. Fazer o mesmo com o fp16.
- **Decide.** Se o mesmo arquivo de pesos der acurácias diferentes, o problema é de avaliação (padding, lote, precisão). Se der a mesma, a diferença nasce na quantização.
- **Critério.** Mesma acurácia ±1 questão para o mesmo arquivo.

#### E0.2 Letras previstas e acurácia sem viés de letra [R se os logits foram salvos; senão E]

- **Desenho.** Para cada condição das Tabelas 9, 11 e 12 do PDF: histograma das letras previstas e do gabarito; acurácia após subtrair o log-prior médio por letra, estimado em questões fora das 600; acurácia média sobre as 4 rotações das opções.
- **Decide.** Se 27,2% (melhor NLL da Tabela 12) corresponder a prever quase sempre a mesma letra, H1 está confirmada no mecanismo.
- **Critério.** Tabela por condição; a dispersão entre execuções equivalentes cai para dentro do limite de 2,9 pontos após a correção, ou não cai e isso é reportado.

#### E0.3 Métricas e unidade de repetição [E]

- **Métrica primária.** KL(p_fp16 ‖ p_q) média por token em texto de domínio fora da calibração. Para múltipla escolha, KL sobre a distribuição das quatro letras.
- **Secundárias.** NLL; diferença de logit (LD) em pares mínimos de fatos não usados; fração de respostas trocadas em relação ao fp16, separada em melhorou/piorou. Acurácia só com rotação de opções.
- **Realizações.** No GPTQ, R sorteios de calibração com semente registrada. No RTN, que é determinístico, R deslocamentos aleatórios dos blocos de 64 colunas (ou permutações de colunas antes do agrupamento): são quantizadores equivalentes que só diferem no alinhamento dos grupos.
- **Análise.** Decomposição da variância em questões/linhas e realizações; toda comparação entre condições é pareada por realização.

**Minha conta.** Com desvio da diferença por realização de 1,2 a 1,7 ponto (dispersão dos controles sob RTN na Tabela 9), detectar 3 pontos pede R ≈ 1,3 a 2,5; R = 5 dá folga. Usar as ~2.000 questões do OAB fora da calibração baixa o erro-padrão por modelo de 2,04 para ~1,12 ponto.

#### E0.4 Teste unitário de identidade [R]

Proteger um conjunto vazio pelo caminho de código "protegido" deve reproduzir o uniforme bit a bit, em RTN e em GPTQ, para unidades contíguas e para colunas espalhadas. Resolve a dúvida sobre reagrupamento dos blocos e sobre a queda de 9 a 13 pontos com proteção aleatória sob GPTQ.

**Critério de decisão da Fase 0.** Existe uma métrica primária cujo desvio entre realizações é pelo menos três vezes menor que o efeito do int4 uniforme. Se não existir em 4 bits, o programa passa a operar em 3 bits, onde o dano é maior (Tabela 12).

**Esboço de código (E0.1–E0.3).**

```python
torch.save({k: v.cpu() for k, v in model.state_dict().items()}, "q.pt")      # GPU A
model.load_state_dict(torch.load("q.pt"))                                    # GPU B
logits = eval_letters(model, questions, batch_size=1)                        # [N, 4]

pred = logits.argmax(1)
hist_pred, hist_gold = np.bincount(pred, minlength=4), np.bincount(gold, minlength=4)
prior = logp_out_of_sample.mean(0)
acc_debiased = ((logits.log_softmax(1) - prior).argmax(1) == gold).float().mean()

kl = (p_fp16 * (p_fp16.log() - p_q.log())).sum(-1).mean()                    # métrica primária
improved = ((pred == gold) & (pred_fp16 != gold)).sum()
worsened = ((pred != gold) & (pred_fp16 == gold)).sum()
```

---

### Fase 1 — O teto do ganho de domínio

**Por quê.** É a peça central e a mais barata: começa com dados que já existem e define o ramo do paper.

#### E1.1 Confiabilidade e correlação corrigida [R]

- **Dados.** As tabelas ΔL por camada, domínio e formato do 0.8B (64 linhas por domínio, pareadas por linha).
- **Desenho.** Dividir as 64 linhas em duas metades, muitas vezes; correlacionar os perfis por camada das duas metades; corrigir por Spearman–Brown (r = 2r_½/(1+r_½)); obter r_xx por domínio e formato. Calcular ρ = r_xy/√(r_xx·r_yy) entre pares de domínios, com bootstrap sobre linhas.
- **Decide.** ρ perto de 1: a ordenação das camadas é a mesma entre domínios. ρ claramente abaixo de 1: há estrutura de domínio na sensibilidade.

#### E1.2 Plano-oráculo contra plano genérico, com piso de ruído [R → E]

- **Desenho.** Para o domínio-alvo T e orçamento B de bytes extras (+1, +2, +5, +10%):
  - plano_T: alocação pela tabela medida em T (metade A das linhas);
  - plano_G: alocação pela média dos outros domínios (sem T);
  - plano_T′: alocação pela tabela medida em T (metade B).
  - Avaliar os três em linhas de T não usadas em nenhuma tabela.
- **Medidas.** Ganho de domínio G(T,B) = [dano(plano_G) − dano(plano_T)] / dano(uniforme). Piso de ruído G₀(T,B) = mesma expressão com plano_T′ no lugar de plano_G. O teto é G − G₀.
- **Por que o piso.** Dois planos feitos com amostras diferentes do mesmo domínio também diferem. Sem G₀, qualquer diferença entre plano_T e plano_G seria lida como efeito de domínio. O próprio PDF reconhece o risco: “The advantage on NLL is partly in-distribution”.
- **Primeiro passo [R].** Ganho previsto pela aditividade, sem rodar nada. **Segundo passo [E].** Ganho medido, aplicando os planos.

#### E1.3 Onde o teto pode aparecer: bits e distância de domínio [E]

- **Bits.** Repetir a tabela medida com int3 e int2 como formatos. É onde o dano é grande e onde o TaCQ reporta ganhos.
- **Distância.** Os quatro domínios atuais são disciplinas do MMLU em inglês, próximas entre si. Acrescentar: texto jurídico em português (enunciados do OAB fora da avaliação, legislação), texto geral em português, e código. A mudança de língua é o deslocamento mais forte disponível e é o caso de uso do estudo.
- **Modelo.** Repetir no 4B.

**Critério de decisão da Fase 1.** Uma tabela teto(bits, distância de domínio) com intervalos. Ela escolhe o ramo (Seção 5).

**O que vai para o paper.** Figura principal: teto em função dos bits, uma curva por distância de domínio, com a faixa do piso de ruído. Tabela: ρ corrigida por par de domínios.

**Esboço de código (E1.1–E1.2).**

```python
def reliability(dL, n_splits=200):                 # dL: [camadas, linhas], um domínio
    r = []
    for _ in range(n_splits):
        idx = np.random.permutation(dL.shape[1]); a, b = idx[::2], idx[1::2]
        rh = spearmanr(dL[:, a].mean(1), dL[:, b].mean(1)).statistic
        r.append(2 * rh / (1 + rh))
    return np.mean(r)

rho_true = spearmanr(dL_T.mean(1), dL_U.mean(1)).statistic / np.sqrt(reliability(dL_T) * reliability(dL_U))

plan_T  = allocate(table=dL_T[:, half_A].mean(1), budget=B)
plan_G  = allocate(table=np.mean([dL_d.mean(1) for d in others], 0), budget=B)
plan_T2 = allocate(table=dL_T[:, half_B].mean(1), budget=B)
G  = (damage(plan_G,  T_heldout) - damage(plan_T, T_heldout)) / damage(uniform, T_heldout)
G0 = (damage(plan_T2, T_heldout) - damage(plan_T, T_heldout)) / damage(uniform, T_heldout)
```

---

### Fase 2 — O que a localização causal acha, e por que não é o domínio

**Por quê.** Fecha o resultado negativo com explicação, em vez de deixá-lo como ganho que sumiu.

#### E2.1 Circuito legal × circuito de siglas gerais [R]

- **Desenho.** Necessidade e suficiência do circuito legal nos pares de siglas gerais e do circuito de siglas gerais nos pares legais; Jaccard dos dois conjuntos em vários k; bootstrap sobre pares.
- **Decide.** Necessidade cruzada alta: o circuito é de expansão de sigla, não de direito.
- **Complemento.** Contar os primeiros tokens distintos entre os fatos legais e a fração de pares em que a resposta começa com a inicial da sigla. Mede o quanto a tarefa é ortográfica.

#### E2.2 Intervalos para todas as suficiências e necessidades [R]

Bootstrap sobre pares nas Tabelas 7, 8, 15 e 16 do PDF. Com 13 a 15 pares, é obrigatório.

#### E2.3 Criticidade genérica dos circuitos [E]

- **Desenho.** Para cada circuito (quatro domínios, legal, siglas gerais) e k ∈ {16, 32, 64}: quantizar só as colunas do circuito e medir KL em texto neutro e em texto do domínio; comparar com conjuntos aleatórios de mesma classe, mesma profundidade e dose mais próxima.
- **Medida.** Razão de especificidade = KL(domínio)/KL(neutro), para circuito e para controle.
- **Decide.** Razão do circuito indistinguível da do controle: o circuito é genérico (H3).

#### E2.4 Proteção com distribuição nula [E]

- **Desenho.** Em RTN e GPTQ, 4 e 3 bits, nos dois tamanhos: circuito legal; dose de mesmo tamanho (estilo OWQ); produto dose × suficiência; 30 conjuntos aleatórios de mesma classe e profundidade; R = 5 realizações. Métrica primária da Fase 0.
- **Análise.** Posição do circuito na distribuição dos aleatórios (p de permutação), pareada por realização; diferença circuito − dose com intervalo sobre realizações.
- **Decide.** H3 confirmada se o circuito não superar a dose e ficar dentro da distribuição nula na métrica contínua.

#### E2.5 Pares entre domínios [E, opcional]

Mesmo template, sujeito jurídico contra sujeito não jurídico. É o desenho que poderia isolar componentes de domínio. Se também achar só o núcleo compartilhado, o argumento 3 fica completo.

**O que vai para o paper.** Uma tabela legal × siglas gerais; uma figura com o histograma dos 30 aleatórios e a posição de circuito, dose e produto; um parágrafo sobre o que o par mínimo pode e não pode ver.

---

### Fase 3 — Sensibilidade medida por unidade

**Por quê.** É a peça construtiva. Responde "que sinal chega mais perto do teto" com algo que o estudo sabe fazer.

#### Definição

Para a unidade u (cabeça ou grupo de 64 neurônios de uma projeção de escrita W) e o domínio d:

s_u(d) = E_{x∼d} KL( p_fp16(·|x) ‖ p_{Q_u}(·|x) ),

onde Q_u é o modelo com apenas as colunas de u quantizadas. Como W·h com as colunas de u quantizadas é igual a W·h − E_u·h_u, com E_u = W_u − Ŵ_u, a intervenção é um hook que subtrai E_u·h_u da saída. Cada sequência do lote pode receber uma unidade diferente. Como as unidades estão alinhadas aos blocos de 64 colunas, as escalas dos demais blocos não mudam e a equivalência é exata.

```python
# E[u] = W[:, cols_u] - Wq[:, cols_u], pré-computado; uma unidade diferente por sequência do lote
def hook(module, inputs, output):
    h = inputs[0]                                          # [B, T, i]
    for b, u in enumerate(units_in_batch):
        output[b] -= h[b][:, cols[u]] @ E[u].T             # = quantizar só as colunas de u
    return output
```

#### E3.1 Varredura [E]

- **Desenho.** 0.8B: 1.680 unidades, pontuadas em int3 (intervenção forte), 64 linhas por domínio, dois domínios (alvo e geral). São 107.520 pares unidade-linha por domínio; o estudo já varreu 68.096 neurônios por domínio.
- **Obstáculo previsto.** O efeito de uma camada inteira em int4 já tem mediana de |ΔL| ≤ 0,001 nat (p. 15). Por unidade será menor.
- **Remédios.** KL em vez de diferença de NLL (pareada e não negativa); pontuar em int3 ou int2; varredura hierárquica (camada primeiro, unidades só nas camadas que concentram o dano; as 10 piores camadas carregam 52% do dano positivo em int4, p. 16).

#### E3.2 Aditividade no nível de unidade [E]

Como em B.1 do PDF: para planos aleatórios de 8, 32 e 128 unidades, comparar a soma dos efeitos individuais com o efeito conjunto, no mesmo texto e em texto separado. Sem aditividade aproximada, a alocação por soma não se justifica.

#### E3.3 Comparação em bytes iguais [E]

Seletores, todos avaliados no arranjo da Tabela 19 do PDF (texto separado, R = 5, métrica primária):

| Seletor | Granularidade | Precisa de | Papel |
|---|---|---|---|
| Aleatório de mesma classe | unidade | nada | piso |
| D por camada | camada | estatísticas | negativo conhecido |
| Dose por unidade (estilo OWQ) | unidade | estatísticas | baseline forte e barato |
| Camadas medidas | camada | 1 forward por camada | melhor atual |
| Circuito de recordação | unidade | patching | hipótese original |
| Dose × suficiência | unidade | já calculados | teste barato da leitura "dois fatores" |
| Dose calibrada (escala medida da camada × dose da unidade) | unidade | camadas medidas + estatísticas | híbrido de custo quase zero |
| **Unidades medidas** | unidade | 1 forward por unidade | proposta |
| Fisher diagonal agregado por unidade (com gradiente) | unidade | backward | referência superior, opcional |

- **Medida.** Fração do dano do uniforme recuperada por orçamento; e fração do teto da Fase 1 alcançada.
- **Decide.** H4 confirmada se "unidades medidas" superar "camadas medidas" e "dose por unidade" em ao menos dois orçamentos, com intervalo sobre realizações que exclua zero.
- **Critério de abandono.** Se não superar "camadas medidas" em nenhum orçamento, a peça construtiva passa a ser a "dose calibrada" (se funcionar) ou sai do paper.

#### E3.4 Versão por domínio [E]

Repetir a análise da Fase 1 com a tabela por unidade: plano por unidades de T contra plano por unidades genérico, com piso de ruído. Diz se a granularidade fina revela estrutura de domínio que a camada esconde.

#### E3.5 Custo [E]

Tabela com tempo e pico de memória de cada seletor no 0.8B e no 4B, incluindo o de gradiente. O argumento forward-only só vale se vier com números.

**O que vai para o paper.** Uma figura de dano recuperado por orçamento, uma curva por seletor; uma tabela de custo; a figura de aditividade.

---

### Fase 4 — Generalização

**Por quê.** "Uma família, um modelo" é a objeção mais provável.

- **Modelos.** O 4B e o 9B da mesma família; e ao menos um transformer convencional de 1 a 4 bilhões de parâmetros de outra família. A arquitetura híbrida do Qwen3.5 tem trabalho próprio de quantização em 2026, então precisa ser tratada como um caso, não como o caso.
- **Quantizadores.** RTN e GPTQ com clipping, em 4, 3 e 2 bits.
- **O que se repete.** Fase 1 completa; E2.4; E3.3 com os quatro seletores principais.
- **Critério.** O sinal do teto e a ordenação dos seletores se mantêm em pelo menos dois de três modelos; onde não se mantêm, isso é reportado como limite.

### Fase 5 (opcional) — TaCQ no mesmo arnês

Rodar o código público do TaCQ em um modelo, com a métrica primária, R realizações e os controles de mesmo tamanho. Requer gradiente. Dá ao paper a comparação que um revisor de quantização vai pedir e testa se o ganho relatado em bits baixos corresponde ao teto medido na Fase 1.

---

## 5. Árvore de decisão: desfecho → paper

| Ramo | Desfecho da Fase 1 | Tese | Papel das Fases 2 e 3 | Força |
|---|---|---|---|---|
| **A** | Teto dentro do piso em 4 bits; cresce em 3 e 2 bits | A consciência de domínio só rende onde o dano é grande; em 4 bits o ganho disponível é genérico | Fase 2 explica por que circuitos não ajudam; Fase 3 dá o melhor seletor genérico e mostra quanto do teto de bits baixos ele alcança | A mais forte: reconcilia resultados negativos e positivos da literatura numa figura |
| **B** | Teto claro já em 4 bits | Há ganho de domínio disponível e os seletores atuais não o capturam | Fase 3 vira o método; Fase 2 vira o contraste ("localizar não é medir") | Paper de método; exige Fase 5 |
| **C** | Teto dentro do piso em todos os bits e distâncias | Sob alocação de precisão mista, não há ganho de domínio mensurável nestes modelos | Fase 2 e 3 como evidência convergente | Negativo forte; Findings, TMLR, workshop |

A Fase 0 acrescenta, em qualquer ramo, a seção "como um ganho aparece e some". Se H1 se confirmar pelo viés de letra, há material para um artigo curto separado sobre recalibração do viés após quantização, que não deve ser misturado ao principal.

Minha expectativa, sem dado que a garanta, é o ramo A. Os indícios são a correlação 0,995 por bloco, os planos que não mudam com o domínio, e os ganhos do TaCQ estarem em 2–3 bits.

---

## 6. Enquadramento recomendado

### 6.1 Tese em uma frase

> Em quantização de baixo número de bits, saber o domínio de uso rende no máximo `[X]`% do dano em 4 bits e `[Y]`% em 2–3 bits; componentes localizados por patching causal não capturam esse ganho porque são genericamente críticos; a sensibilidade medida por unidade, sem gradiente, alcança `[Z]`% do limite.

### 6.2 Título

- Recomendado: **"How Much Can Domain Awareness Buy in Low-Bit LLM Quantization? A Ceiling Analysis with Causal Controls"**
- Alternativa se o ramo for B: "Measure, Don't Localize: Forward-Only Unit Sensitivity for Mixed-Precision LLM Quantization"
- Alternativa se o ramo for C: "Localization Is Not Sensitivity: Causal Circuits Do Not Tell Us Which Weights to Protect"

O título atual ("Causal circuit discovery for domain-aware low-bit quantization") promete o método que os dados não sustentam e deve sair.

### 6.3 Posição diante dos vizinhos

| Trabalho | O que faz (pelo que vi) | Como o paper se diferencia |
|---|---|---|
| TaCQ, arXiv:2504.07389, COLM 2025 | Circuito de pesos por gradiente, mantido em 16 bits; ganhos em 2–3 bits; ganho também sem condicionar em tarefa | Mede o limite do ganho de domínio em vez de propor outro seletor; sem gradiente; controles de mesmo tamanho e mesma dose |
| TAQ, arXiv:2511.06516 | Alocação por camada sem treino, condicionada à tarefa; 87,5% das atribuições estáveis entre tarefas | Quantifica o que essa estabilidade implica como ganho disponível; desce à unidade |
| OWQ, arXiv:2306.02272, AAAI 2024 | Colunas sensíveis por Hessiana diagonal × erro, mantidas em FP16 | É o baseline "dose"; deve ser nomeado assim |
| MXSens, arXiv:2607.17733 | Sensibilidade por camada quantizada isoladamente | Precedente da tabela por camada; a contribuição é a unidade e a análise por domínio |
| arXiv:2609.26241 (finanças) | Calibração por domínio só ajuda quando o dano é grande | Vizinho mais próximo do ramo A; a diferença é alocação/proteção com teto e piso de ruído, não dados de calibração. **Ler antes de fixar a tese** |
| arXiv:2607.00908; arXiv:2611.11079; GTAQ (IJCAI 2026) | Calibração e alocação conscientes de tarefa | Mesmo comentário: são métodos; o paper é a medida do espaço em que eles operam |
| Dutta et al., arXiv:2407.09141, NeurIPS 2024 | Flips e KL como métricas de modelos comprimidos | Adota as métricas; acrescenta a distribuição nula sobre conjuntos protegidos e a variância entre realizações |
| arXiv:2606.19558, "Displacement Is Not Direction: Evaluating Fidelity Metrics for Quantized LLM Deployment" | Pelo trecho retornado na busca sobre Dutta et al.: avalia se métricas de fidelidade servem para escolher entre modelos quantizados; separa flips em ganhos e perdas | Ler para não reivindicar o que já foi dito na parte de avaliação |
| arXiv:2610.00694, "How Divergence Becomes Decision Flips in Compressed Language Models" | Pelo trecho retornado na mesma busca: relaciona divergência e taxa de flips em modelos comprimidos | Idem |
| Hase et al., "Does Localization Inform Editing?" *(de memória)* | Localização por causal tracing não prediz onde editar | Análogo conceitual para a Fase 2; citar como precedente do tipo de resultado |

### 6.4 Contribuições, na ordem em que o paper deve listá-las

1. Um limite superior, com piso de ruído, do ganho de alocação por domínio em 2, 3 e 4 bits, em `[N]` modelos e em deslocamentos de disciplina, de língua e de tipo de texto.
2. Evidência de que circuitos localizados por patching de pares mínimos são genericamente críticos e não superam uma escolha por dose de mesmo tamanho; e a razão de desenho para isso.
3. Um seletor forward-only por unidade, medido diretamente, comparado em bytes iguais com sete alternativas, com custo reportado.
4. Um protocolo de avaliação (distância ao modelo original, distribuição nula sobre conjuntos, várias realizações) e um caso documentado em que o protocolo usual produz um ganho de +3,2 pontos que não sobrevive.

### 6.5 Estrutura em 9 páginas

1. **Introdução** (1 p.). A pergunta; a figura do teto como Figura 1.
2. **Trabalhos relacionados** (0,75 p.). Tabela da 6.3 em prosa.
3. **Protocolo** (1,25 p.). Quantizador, bytes exatos, métrica primária, realizações, controles.
4. **O teto do ganho de domínio** (1,75 p.). Fase 1.
5. **O que a localização causal acha** (1,5 p.). Fase 2.
6. **Sensibilidade medida por unidade** (1,5 p.). Fase 3.
7. **Como um ganho aparece e some** (0,75 p.). Fase 0 e o caso do +3,2.
8. **Limitações** (0,5 p.).

Apêndice: TARC, ablação pela média, D por camada e persona como negativos menores em uma tabela; detalhes de GPTQ; fatos jurídicos; reprodução. As Seções 12 e A do PDF vão para o repositório.

### 6.6 Figuras e tabelas

| # | Conteúdo | Fonte |
|---|---|---|
| Fig. 1 | Teto do ganho de domínio por bits, uma curva por distância de domínio, faixa do piso de ruído | E1.2, E1.3 |
| Fig. 2 | Histograma dos ganhos de 30 conjuntos aleatórios, com circuito, dose e produto marcados | E2.4 |
| Fig. 3 | Dano recuperado por orçamento de bytes, uma curva por seletor | E3.3 |
| Fig. 4 | ΔNLL contra acurácia por letra, por condição, antes e depois da correção de viés | E0.2 |
| Fig. 5 | Soma dos efeitos por unidade contra efeito conjunto | E3.2 |
| Tab. 1 | Correlação entre domínios, bruta e corrigida pela confiabilidade | E1.1 |
| Tab. 2 | Circuito legal × siglas gerais: necessidade, suficiência, Jaccard | E2.1 |
| Tab. 3 | Custo (tempo, memória) por seletor | E3.5 |
| Tab. 4 | Replicação por modelo | Fase 4 |

O PDF atual tem 19 tabelas e uma figura esquemática. Um paper com essa tese precisa que o leitor veja o teto em uma imagem.

### 6.7 Resumos-modelo (preencher só com resultados obtidos)

**Ramo A.**

> Task-aware quantization methods keep the weights that matter for a target domain at higher precision. We ask how much there is to gain. Using directly measured sensitivities and exact byte accounting, we compare allocations built for a target domain with allocations built without it, against a noise floor given by two allocations built from disjoint samples of the same domain. On `[N]` models, the domain-specific gain is within the noise floor at 4 bits and reaches `[Y]`% of the uniform-quantization damage at 2–3 bits, and it grows with domain distance (`[...]`). Components localized by minimal-pair causal patching do not capture this gain: they are critical for the model in general, a circuit for legal acronyms largely coincides with one for non-legal acronyms, and protecting them does not beat a same-size choice by quantization error. A forward-only sensitivity measured per unit recovers `[Z]`% of the available gain at equal bytes. We also document how a single-realization accuracy comparison yields a 3.2-point gain with a confidence interval excluding zero that does not survive a null distribution over protected sets.

**Ramo B.** Trocar a segunda metade por: o teto em 4 bits é `[X]`%; seletores por dose e por camada alcançam `[...]`%; o seletor por unidade alcança `[Z]`%, a `[...]` do custo de memória de um seletor por gradiente.

**Ramo C.** Trocar por: o ganho fica dentro do piso em todos os bits e distâncias testados; listar o que isso exclui e o que não exclui (granularidade de peso, quantização de ativações, modelos maiores).

---

## 7. Objeções previsíveis e resposta preparada

| Objeção | Resposta que o paper precisa ter |
|---|---|
| "Uma família de modelos, modelos pequenos" | Fase 4; declarar o intervalo de tamanhos como limite |
| "O teto não é um limite superior de verdade; é o ótimo de uma busca gulosa sobre uma tabela aditiva" | Chamar de "oráculo dentro da família de alocações por camada/unidade"; mostrar a aditividade (E3.2); conferir com busca exaustiva em subconjuntos pequenos |
| "Os domínios são parecidos" | E1.3: língua e código |
| "KL esconde o efeito na tarefa" | Reportar também LD em fatos e acurácia com rotação de opções e R realizações; mostrar a Fig. 4 |
| "Por que sem gradiente?" | Tab. 3 com memória e tempo; referência com gradiente em ao menos um modelo |
| "Faltou comparar com TaCQ" | Fase 5, ou ao menos um modelo; se não for feito, dizer que o paper mede o espaço, não compete no método, e mostrar onde os ganhos relatados pelo TaCQ caem na Fig. 1 |
| "Quantização simulada, sem formato nem aceleração" | Declarar; argumentar que unidades são blocos contíguos, mais simples de empacotar que pesos ou colunas espalhadas. Um trecho de busca (QEFT, arXiv:2410.08661) aponta dificuldade de implantação do formato misto do OWQ. É argumento, não demonstração |
| "Resultado negativo" | O teto converte o negativo em número; a Fase 3 é construtiva |
| "O efeito por unidade é ruído" | Confiabilidade split-half da tabela por unidade; varredura hierárquica; pontuação em int3 |
| "A planilha jurídica não foi revisada" | Revisão por jurista antes da submissão; licença do conjunto de exames esclarecida |

---

## 8. Venues e mínimo publicável

| Pacote | Conteúdo | Destino realista |
|---|---|---|
| Mínimo | Fases 0, 1 (com E1.3 em dois modelos) e 2 | Workshop de eficiência ou de interpretabilidade em conferência grande; TMLR; PROPOR, STIL ou BRACIS se o recorte for português jurídico |
| Completo | + Fases 3 e 4 | Trilha principal de NeurIPS/ICML/ICLR ou ACL/EMNLP (Findings como piso provável no ramo C) |
| Completo com comparação | + Fase 5 | Melhora a chance em trilha principal no ramo B |

Não confirmei prazos nem chamadas de 2027.

---

## 9. Riscos

| Risco | Sinal de alerta | Mitigação |
|---|---|---|
| Um dos trabalhos de 2026 já reporta um teto equivalente | Leitura de arXiv:2609.26241, 2611.11079 e 2607.00908 | Ler antes da Fase 1 medida; se houver, o diferencial passa a ser o piso de ruído, a unidade e os controles causais |
| A Fase 0 não acha métrica estável em 4 bits | Desvio entre realizações da mesma ordem do efeito | Operar em 3 bits; reportar 4 bits como "não mensurável neste arnês" |
| Efeito por unidade abaixo do ruído | Confiabilidade split-half baixa em E3.1 | Hierarquia; int2; agregar em blocos de unidades; cair para "dose calibrada" |
| Não aditividade entre unidades | E3.2 com razão longe de 1 | Alocar por blocos; reportar como limite |
| O teto é alto e nenhum seletor o alcança | Fase 3 muito abaixo do oráculo | Vira um resultado: há ganho e ele não é capturado por sinais forward-only; motiva a referência com gradiente |
| A queda com proteção aleatória sob GPTQ é um erro de implementação | E0.4 falha | Corrigir antes de qualquer número de GPTQ com proteção |

---

## 10. Sequência de trabalho

1. E0.1, E0.2, E0.4 (dias; decide o arnês).
2. Leitura dos seis trabalhos vizinhos de 2025–2026 (em paralelo).
3. E1.1 e o passo previsto de E1.2 (só reanálise).
4. E2.1 e E2.2 (só reanálise).
5. E0.3 com R realizações; passo medido de E1.2.
6. E1.3 (int3, int2, português, código, 4B). **Ponto de decisão do ramo.**
7. E2.3 e E2.4.
8. E3.1 a E3.5.
9. Fase 4; Fase 5 se o ramo for B.
10. Escrita na estrutura da 6.5.

Os itens 1 a 4 não exigem quantização nova além da já feita e bastam para saber se a tese se sustenta.

---

## 11. Limites deste documento

- É uma proposta. Nenhum experimento foi executado e nenhum resultado é antecipado.
- Parte de uma avaliação feita só a partir do PDF, sem código, logs ou planilhas.
- A expectativa de ramo A e a hipótese de viés de letra são conjecturas minhas.
- A correção de atenuação do Argumento 2 depende de uma confiabilidade que não conheço; dei três cenários.
- Literatura, em três níveis:
  - **Busquei diretamente e vi o resumo ou a ficha:** TaCQ, OWQ, Dutta et al., arquitetura do Qwen3.5.
  - **Só vi o trecho que uma busca retornou:** TAQ (arXiv:2511.06516), MXSens (2607.17733), arXiv:2609.26241, 2607.00908, 2611.11079, GTAQ (IJCAI 2026), 2606.19558, 2610.00694, QEFT (2410.08661), e os dois trabalhos sobre a arquitetura híbrida (2609.04098, 2607.02893). O que afirmo sobre eles vem só desses trechos.
  - **De memória:** Hase et al.
  - Os trabalhos do segundo nível ainda não foram lidos na íntegra. A tabela 6.3 deve ser refeita depois dessa leitura.
- Os números do PDF citados aqui foram conferidos contra o texto extraído; os de "Minha conta" estão em `apoio/contas_proposta.json` e `apoio/contas_revisor.json`.
