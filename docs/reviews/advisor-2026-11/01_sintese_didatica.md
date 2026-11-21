# Síntese didática da revisão do orientador

Fonte: `agentic-eval/` (`00_LEIA-ME`, `01_proposta_evolucao_paper`, `02_avaliacao_critica`), feita a partir do PDF de 19 páginas
(`docs/paper/method.pdf`, versão do commit `e8a76de`). Os códigos entre parênteses (G1, M2, H5, L4, E0.1...) remetem aos documentos originais.
Esta síntese explica o raciocínio; o plano de trabalho está em `02_plano_de_acao.md` e as referências em `referencias.csv`.

---

## 1. Em um minuto

1. **O documento é um registro honesto, mas ainda não é um paper.** Tem boas ideias de controle e resultados negativos limpos, mas conta a história na ordem em que os experimentos foram feitos, e o próprio resumo afirma um ganho e depois o retira.
2. **O ganho principal (+3,2 pontos no OAB ao proteger o circuito jurídico) não está estabelecido.** Ele vem de uma única execução, medida por acurácia de letra, e essa métrica oscila mais do que o efeito procurado.
3. **A ideia "achar o circuito da tarefa e mantê-lo em 16 bits" já existe** (TaCQ, COLM 2025), e escolher colunas por "dose" é o OWQ (AAAI 2024). Nenhum dos dois está citado.
4. **Nossos próprios dados sugerem que os circuitos achados são componentes importantes para o modelo em geral**, não componentes "de direito". Isso explica por que a proteção ajudou direito e não-direito por igual.
5. **A recomendação é mudar a pergunta do paper:** de "proteger o circuito ajuda?" para "quanto a consciência de domínio pode render, e que sinal chega mais perto desse limite?". Essa pergunta rende um paper qualquer que seja o resultado.
6. **Antes de qualquer experimento novo, há seis checagens baratas** (Seção 7 abaixo). Algumas só exigem reanálise do que já temos.

---

## 2. Três ideias para entender a revisão

### 2.1 "Ruído" não é uma explicação; é o que falta explicar
Quando duas execuções do mesmo código dão 40,7% e 49,5%, é tentador dizer "foi ruído de arredondamento". A revisão mostra que existe um teto para quanto a sorte *por questão* consegue mover a acurácia: com 600 questões, a diferença entre duas execuções equivalentes não deveria passar de cerca de 2,9 pontos de desvio-padrão. Observamos 8,8. Logo há algo que move muitas questões juntas (por exemplo, o modelo passar a preferir a letra "B"), ou as duas execuções não eram equivalentes de fato.

> Analogia: se duas balanças pesam o mesmo objeto e diferem em 3 kg, não adianta dizer "balanças têm erro". Ou uma está descalibrada, ou estão pesando coisas diferentes.

### 2.2 O intervalo de confiança responde a uma pergunta diferente da que fazemos
O intervalo de bootstrap sobre as 600 questões responde: *"estes dois modelos específicos diferem nestas questões?"*. Nossa pergunta é: *"escolher as colunas pelo circuito é melhor do que escolher de outro jeito?"*. Para essa, a unidade de repetição é o **conjunto protegido**, não a questão. A prova de que o intervalo não serve: dos oito conjuntos de controle, quatro também tinham intervalo que exclui zero (dois para cada lado). Se o teste "acusa efeito" em metade dos controles, ele não mede o efeito da seleção.

### 2.3 Localizar não é o mesmo que medir sensibilidade
O patching de pares mínimos troca *um sujeito por outro do mesmo domínio* (uma sigla jurídica por outra). Tudo o que é comum ao domínio fica igual nas duas execuções e, por construção, desaparece. O método encontra o que distingue **um fato de outro**, não o que o modelo usa para **direito** em geral. A sensibilidade à quantização, por sua vez, é medida diretamente: quantiza-se uma parte e vê-se quanto o modelo piora. São perguntas diferentes, e a segunda é a relevante para decidir onde gastar bits.

---

## 3. Os problemas graves, em linguagem simples

| Código | O problema | Por que importa | O que fazer |
|---|---|---|---|
| **G1** | Execuções "idênticas" dão 40,7% e 49,5%; o texto chama isso de ruído de arredondamento | A diferença passa do teto do que sorte por questão consegue produzir; a retratação do ganho se apoia nessa explicação | Avaliar os mesmos pesos quantizados nas duas GPUs; contar quantas vezes cada letra é prevista; calcular a acurácia sem o viés de letra |
| **G2** | "Recupera um terço da perda" supõe que a perda (8,8 pontos) existe | Entre quatro modelos int4, a acurácia varia de 34,7% a 49,5%, e o fp16 (43,8%) cai dentro dessa faixa: a perda média é 3,8, não 8,8 | Medir dano por KL ao fp16 e NLL; reescrever a frase do resumo |
| **G3** | O intervalo sobre questões não mede a incerteza da seleção | Metade dos controles "significativos" mostra que o teste não discrimina | Sortear 30 conjuntos aleatórios e dar a posição do circuito nessa distribuição (p de permutação) |
| **G4** | A retratação sob GPTQ compara com um único uniforme (49,5%), que é a execução anômala | Contra o outro uniforme (40,7%) o circuito ganha +6 a +7; contra os aleatórios do mesmo tamanho ganha +6 a +11. O sinal depende de qual referência se escolhe | Resolver G1 primeiro; repetir com 5 ou mais realizações; teste "zero colunas protegidas = uniforme" |
| **G5** | O desenho não mede "conhecimento de um domínio", e siglas são em parte ortografia | A letra inicial da sigla prevê a primeira palavra (S→Supremo, T→Tribunal); o circuito de siglas gerais deve ser parecido com o jurídico | Cruzar o circuito jurídico com o de siglas gerais (os dados já existem); trocar "domain" por "relation" no texto |
| **G6** | Faltam os antecedentes diretos: TaCQ, OWQ, SpQR, GPTQ | Um revisor de quantização vai procurá-los na primeira página e lerá o trabalho como reinvenção | Escrever a seção de trabalhos relacionados; renomear a "dose" como "OWQ-style" |
| **G7** | O documento é um caderno de laboratório | Ordem cronológica, "current method" que a Seção 9 nega, suítes ainda rodando, fora do estilo da conferência, 14 páginas de corpo | Reestruturar conforme a proposta (Seção 6 da 01) |

## 4. Os problemas moderados

| Código | Resumo | Ação |
|---|---|---|
| M1 | Em colunas espalhadas, refazer os blocos de 64 muda o arredondamento da camada inteira; o baseline "ativação" recebe uma perturbação extra | Manter os blocos originais; teste de identidade com conjunto vazio |
| M2 | Regime (int4/RTN) escolhido porque ali o OAB cai sem colapsar; no MMMLU a margem é de 10 e 3 questões | Dizer "especificidade não detectável com esta margem"; medir por KL |
| M3 | O patching de persona não tem controle de dose | Controle aleatório pareado em energia, ou mover persona para negativos |
| M4 | "Neurônios quase disjuntos" depende do limiar; em enriquecimento sobre o acaso, grupos e neurônios se sobrepõem de forma parecida | Reportar razão sobre o acaso em vários k |
| M5 | A tabela medida tem ruído e é "in-distribution"; o erro D pode estar só mal normalizado entre camadas | Testar D relativo (a energia de saída da camada) antes de descartá-lo |
| M6 | "Validated" é mais forte do que a evidência | Dizer "confere no erro por matriz e na NLL" |
| M7 | Planilha sem revisão de jurista; conjunto de exames sem licença; uma família de modelos | Revisão jurídica; usar as ~2.000 questões fora da calibração; um segundo modelo |
| M8 | Suficiência sem intervalos; controles não pareados em profundidade | Bootstrap sobre pares |

---

## 5. Os achados valiosos escondidos nos nossos resultados

A revisão aponta que o material mais forte do documento é o que ele trata como incômodo:

- **H1: mesma NLL, 8 pontos de diferença; melhor NLL, acurácia de acaso.** Um modelo com perda de NLL de 0,063 faz 35,0% e outro com 0,066 faz 43,3%; a combinação de melhor NLL da Tabela 12 faz 27,2%. Com o diagnóstico de G1 isso vira a figura principal de um paper de avaliação.
- **H3: metade dos controles é "significativa" nas duas direções.** É um experimento de taxa de falso positivo já meio feito. Com 30 conjuntos, vira um resultado citável.
- **H4: sob GPTQ, o circuito fica 6 a 11 pontos acima de todos os aleatórios do mesmo tamanho, e os aleatórios ficam 9 a 13 pontos abaixo do uniforme.** Ou há um problema na implementação das colunas protegidas, ou proteger colunas arbitrárias atrapalha a compensação de erro e o circuito escapa disso. As duas saídas interessam.
- **H5: o resultado do MMMLU era previsível.** O experimento E11 já mostrava que quantizar o circuito eleva a NLL de texto neutro muito mais do que um conjunto de mesma dose (+4,58 contra +0,20): os circuitos são genericamente críticos.
- **H6: pelo critério do próprio estudo, a "dose" por unidade ganha do circuito em 3 de 4 comparações** (Tabelas 18 e 19), e a alocação por camadas medidas recupera cerca do dobro do circuito em todo orçamento.
- **H7: dose funciona por coluna e falha entre camadas**, o que aponta para um fator de escala por camada que falta em D.
- **H8: a inversão por tipo de camada em arquiteturas híbridas já tem trabalho de 2026** (arXiv:2609.04098, 2607.02893): a Tabela 14 é confirmação, não resultado novo.
- **H9: o efeito de persona é majoritariamente lexical** (só 30% dos 2,82 nats é específico da instrução no 0,8B).

---

## 6. A nova pergunta e o caminho proposto

> **Quanto um quantizador pode ganhar por saber o domínio de uso, e que sinal forward-only chega mais perto desse limite?**

Quatro peças, na ordem em que dependem uma da outra:

| Fase | Pergunta | Em palavras simples |
|---|---|---|
| **0. Arnês confiável** | Nossa medida de dano oscila menos do que o efeito que queremos ver? | Consertar a régua antes de medir qualquer coisa |
| **1. O teto** | Quanto uma alocação feita para o domínio ganha sobre uma genérica, com piso de ruído? | Descobrir o ganho máximo possível, e ver se ele sai do ruído |
| **2. Por que localizar não basta** | Os circuitos são do domínio ou genéricos? | Fechar o resultado negativo com uma explicação |
| **3. Sensibilidade por unidade** | Medir cada cabeça ou grupo de neurônios diretamente supera a medida por camada e a dose? | A peça construtiva: um sinal que chega mais perto do teto |
| **4. Generalização** | Vale em outros modelos, tamanhos e arquiteturas? | Responder "uma família, um modelo" |

**O ramo do paper depende do teto (Fase 1):**

- **A (expectativa da revisão):** o teto fica dentro do ruído em 4 bits e cresce em 2 a 3 bits. Tese: a consciência de domínio só rende onde o dano é grande. É o ramo mais forte, porque reconcilia resultados positivos e negativos da literatura numa figura.
- **B:** há ganho claro já em 4 bits e os seletores atuais não o capturam. Vira um paper de método e pede a comparação com TaCQ.
- **C:** nenhum ganho em nenhum regime. Resultado negativo forte (Findings, TMLR, workshop).

Título sugerido (ramo A): *"How Much Can Domain Awareness Buy in Low-Bit LLM Quantization? A Ceiling Analysis with Causal Controls"*.

Pacote mínimo (Fases 0 a 2 em dois modelos): workshop, TMLR, ou PROPOR/STIL/BRACIS. Pacote completo (com Fases 3 e 4): trilha principal.

---

## 7. As seis checagens baratas (que dizem se a tese se sustenta)

1. **Mesmos pesos, duas GPUs** (E0.1): salvar o modelo quantizado, avaliar nas duas máquinas e comparar com o fp16 de cada uma.
2. **Letras previstas por condição e acurácia sem viés de letra** (E0.2): um histograma por condição já diz se 27,2% é "responder sempre a mesma letra".
3. **Proteger zero colunas = uniforme, bit a bit** (E0.4), em RTN e GPTQ.
4. **Confiabilidade das tabelas de sensibilidade e correlação corrigida entre domínios** (E1.1): a correlação de 0,58 entre matemática e direito pode ser, na verdade, perto de 1 (se a tabela for pouco confiável) ou perto de 0,6 (se for confiável); a conta depende de uma medida que ainda não fizemos.
5. **Cruzar o circuito jurídico com o de siglas gerais** e pôr intervalos em todas as suficiências e necessidades (E2.1 e E2.2): só reanálise.
6. **Ler os trabalhos vizinhos de 2025 a 2026**, principalmente arXiv:2609.26241 e 2611.11079, que podem já reportar algo parecido com o teto.

## 8. O que o repositório já responde (conferido no código e nos resultados)

Vários pedidos da revisão (seção 5 do `00_LEIA-ME`) podem ser respondidos agora:

| Pedido | Resposta |
|---|---|
| Blocos de 64 colunas com colunas protegidas espalhadas | **São refeitos sobre as colunas restantes** (`compress/protect.py`, `MixedQuant`; `compress/gptq.py`, `gptq_quantize`). Para unidades contíguas e alinhadas (cabeças de 128 ou 256 canais, grupos de 64 neurônios), as fronteiras não mudam; para colunas espalhadas (AWQ-like, `outlier_all`) mudam. A hipótese M1 se confirma na implementação. |
| Modo das colunas protegidas no GPTQ nos números reportados | **Congeladas.** A opção `sink` (colunas ajustáveis como "error sinks") foi acrescentada depois e não entrou nas tabelas. |
| Três sorteios aleatórios da Tabela 9 | Sob GPTQ: 47,0; 46,8; 49,8 (32 unidades) e 40,8; 37,8; 36,0 (128 unidades). Em `outputs/baselines_4b.json`. |
| O fp16 dá 43,8% nas duas GPUs? | **Sim.** O fp16 deu 43,8% (NLL 5,068) na 3090 e na 3080 Ti. Isso enfraquece a hipótese de que a avaliação em si difere entre as máquinas, ao menos para o fp16; para os pesos quantizados continua em aberto. |
| Lote, padding e dtype | A avaliação percorre **uma questão por vez**, com as quatro letras avaliadas juntas, sem preenchimento entre questões; dtype fp16 em GPU. |
| NLL em texto neutro da 3080 Ti | **Não foi registrada** (a rodada começou antes de acrescentarmos essa métrica). |
| Camada de saída amarrada ao embedding | **Não é quantizada** (só as Linear dentro das camadas do decodificador). |
| Logits por questão | **Não foram salvos**: guardamos só acerto/erro e a NLL da letra correta por questão. Por isso E0.2 é experimento, não reanálise. |
| Venue e prazo pretendidos | **Pendente: decisão sua e do orientador.** |

## 9. Onde ter cuidado ao ler a revisão

- Ela foi feita só a partir do PDF: sem código, logits nem planilhas. Várias afirmações são **hipóteses** (viés de letra, diferença de avaliação entre GPUs, tarefa de siglas parcialmente ortográfica), e o documento diz isso.
- A literatura recente foi vista em trechos de busca; os títulos e autores do `referencias.csv` foram conferidos no arXiv, mas o **conteúdo** dos trabalhos de 2026 ainda precisa ser lido antes de qualquer afirmação de posicionamento.
- A expectativa de que o resultado cairá no ramo A é uma conjectura, não um dado.
- Os experimentos que fizemos depois do PDF (suítes de calibração, bits mistos, baixo posto) usam a métrica antiga de acurácia de letra; devem ser relidos pela NLL neutra e pelo `law_logp` ou refeitos depois da Fase 0.
