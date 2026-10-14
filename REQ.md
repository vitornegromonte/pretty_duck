# Tarefa para o Codex: desenvolver compressão heterogênea por domínio a partir do código original do Persona

## 1. Contexto do repositório

O repositório conterá apenas o código original do método Persona, que servirá como referência e ponto de partida para a implementação.

**Não presuma que existam módulos de quantização, coleta de estatísticas, alocação de bits, avaliação de compressão ou scripts experimentais.** Primeiro, inspecione o código disponível e identifique o que pode ser reutilizado.

O objetivo é evoluir essa implementação para investigar uma nova hipótese: utilizar informações de ativação específicas por domínio para decidir onde preservar precisão e onde aplicar compressão mais agressiva, mantendo o desempenho no domínio-alvo.

O código original deve ser preservado. Faça alterações incrementais, reutilize componentes quando apropriado e mantenha a possibilidade de executar o método original como baseline.

## 2. Objetivo científico

Implementar e avaliar um método de compressão heterogênea de modelos de linguagem que:

- preserve o desempenho em um domínio-alvo;
- reduza o tamanho do modelo por meio da quantização seletiva de pesos;
- utilize estatísticas de ativação coletadas separadamente por domínio;
- compare alocação heterogênea de precisão com quantização uniforme;
- avalie se as informações de importância por domínio utilizadas pelo Persona contribuem para a decisão de compressão.

Não presuma que a hipótese funciona. O código deve permitir testá-la experimentalmente.

## 3. Restrições obrigatórias

1. Não treinar redes auxiliares.
2. Não realizar fine-tuning do modelo-base.
3. Não introduzir treinamento de adaptadores como requisito do método principal.
4. Não substituir o método original por uma implementação inventada sem antes inspecionar o código.
5. Não afirmar que uma configuração utiliza determinada precisão ou formato se o backend não a implementar de fato.
6. Não estimar o tamanho final apenas pelo número nominal de bits: medir os bytes reais dos pesos, escalas, metadados e demais componentes relevantes.
7. Não alterar silenciosamente o comportamento do Persona original.
8. Não produzir resultados experimentais fictícios.
9. Evitar dependências desnecessárias e arquiteturas excessivamente complexas.

## 4. Etapa inicial: auditoria do código Persona

Antes de implementar novas funcionalidades:

1. Leia os arquivos disponíveis e identifique os pontos de entrada.
2. Mapeie como o código coleta ativações, calcula importância, gera máscaras e avalia o modelo.
3. Identifique a arquitetura e os tipos de camadas suportados.
4. Verifique como são carregados o modelo, o tokenizer e os dados.
5. Identifique quais partes podem ser reutilizadas para coletar estatísticas por domínio.
6. Determine se a implementação original aplica máscaras esparsas, remove pesos, modifica módulos ou utiliza outro mecanismo.
7. Registre limitações e incompatibilidades relevantes para a quantização.

Crie uma documentação curta da arquitetura encontrada antes de fazer alterações estruturais.

## 5. Estatísticas de ativação por domínio

Implemente coleta de estatísticas para o domínio-alvo \(T\) e os domínios secundários \(S_1,\ldots,S_n\).

Para cada camada ou bloco compatível, estime:

\[
A_{d,j}=\mathbb{E}_{x\sim d}[|h_j(x)|]
\]

\[
V_{d,j}=\mathbb{E}_{x\sim d}[h_j(x)^2]
\]

Requisitos:

- processar os dados em batches;
- acumular estatísticas sem armazenar todas as ativações;
- evitar contar padding como conteúdo válido;
- controlar seeds e registrar a configuração de calibração;
- reutilizar hooks ou mecanismos de captura já existentes quando isso for adequado;
- testar se as estatísticas são calculadas corretamente.

Documente a convenção utilizada para os eixos de entrada e saída de cada camada.

## 6. Importância por domínio

Investigue a reutilização do mecanismo de importância do Persona.

Como ponto de partida, considere:

\[
I_{d,ij}=|W_{ij}|A_{d,j}
\]

Adapte os índices à convenção real dos módulos encontrados.

Implemente, quando compatível, uma comparação entre:

- importância baseada em ativações absolutas;
- importância baseada em segundos momentos;
- importância contrastiva entre o domínio-alvo e os domínios secundários;
- sensibilidade estimada diretamente ao erro de quantização.

A importância contrastiva pode ser definida como:

\[
C_{ij}=I_{T,ij}-\lambda I_{S,ij}
\]

Ela deve ser tratada como uma hipótese de priorização, não como garantia de menor perda.

Não presuma que as máscaras originais do Persona sejam equivalentes a uma política de alocação de bits. Preserve a distinção entre seleção de pesos, esparsidade e quantização.

## 7. Estimativa do erro de quantização

Para cada camada ou bloco compatível, compare os pesos originais \(W\) com os pesos reconstruídos \(W_q\).

Defina:

\[
E=W-W_q
\]

Uma estimativa do erro de reconstrução da saída, para entradas do domínio \(d\), é:

\[
D_d=\mathbb{E}_{x\sim d}
\left[\|xE^\top\|_2^2\right]
\]

Quando viável, calcule essa estimativa diretamente a partir das ativações de calibração. Como aproximação mais barata, utilize segundos momentos por canal:

\[
D_d\approx\sum_{i,j} E_{ij}^2V_{d,j}
\]

Registre quando o método diagonal é utilizado e deixe claro que ele ignora termos de correlação entre canais.

Não interprete a soma desses erros locais como uma medida exata da perda final do modelo. Valide as decisões por meio de avaliações end-to-end.

## 8. Alocação heterogênea de precisão

Implemente um alocador que selecione formatos de quantização por camada ou bloco, respeitando os formatos efetivamente disponíveis no backend.

O objetivo é minimizar o armazenamento:

\[
\min_{\{b_g\}}\sum_g B_g(b_g)
\]

sujeito a uma restrição de degradação no domínio-alvo:

\[
\Delta L_T\leq\varepsilon
\]

Em que \(B_g(b_g)\) representa o custo real de armazenamento do bloco \(g\).

Como primeira heurística, implemente uma estratégia gulosa que parte de uma configuração de maior precisão e avalia reduções de precisão pelo custo incremental estimado:

\[
\rho=
\frac{D_T(b')-D_T(b)}
{B(b)-B(b')}
\]

Priorize as alterações com menor aumento estimado de erro por byte economizado, mantendo a configuração dentro do orçamento de degradação definido.

Requisitos:

- suportar um orçamento de armazenamento ou degradação configurável;
- contabilizar escalas, metadados e alinhamento quando aplicável;
- detectar formatos indisponíveis;
- não simular uma quantização inexistente;
- registrar a decisão tomada para cada bloco;
- oferecer uma configuração uniforme para comparação;
- permitir avaliar a sensibilidade da decisão aos hiperparâmetros.

Se o backend não suportar múltiplas precisões em um mesmo modelo, documente a limitação e implemente primeiro a alternativa compatível mais simples. Não construa um sistema de kernels personalizado sem necessidade demonstrada.

## 9. Correções de baixa ordem: somente como extensão

Não implemente correções low-rank como requisito inicial.

Primeiro, estabeleça uma baseline confiável de quantização uniforme e uma implementação funcional da alocação heterogênea.

Somente depois, se a arquitetura e os resultados justificarem, investigue uma correção:

\[
W\approx W_q+UV^\top
\]

Avalie o custo de armazenamento de \(U\) e \(V\), o custo de inferência e o ganho efetivo no domínio-alvo.

Não introduza treinamento de uma rede auxiliar. Se a obtenção da correção exigir otimização adicional, mantenha-a separada, documentada e opcional.

## 10. Organização e interface

Adapte a organização às convenções do repositório original. Não crie uma estrutura extensa antes de entender o código existente.

Separe logicamente as funcionalidades de:

- coleta de estatísticas;
- estimativa de erro;
- alocação de precisão;
- aplicação da quantização;
- avaliação;
- execução dos experimentos.

Se apropriado, disponibilize comandos equivalentes a:

- `collect_stats`
- `estimate_quant_error`
- `allocate_bits`
- `quantize_model`
- `evaluate_compression`

Esses nomes são sugestões, não uma exigência de criar cinco executáveis independentes.

Forneça configurações explícitas para o modelo, os conjuntos de calibração, o domínio-alvo, os domínios secundários, os formatos suportados, o orçamento de compressão e a seed.

## 11. Experimentos mínimos

Implemente comparações entre:

1. Modelo original, sem compressão.
2. Quantização uniforme com um formato suportado.
3. Quantização heterogênea com alocação baseada no erro estimado do domínio-alvo.
4. Quantização heterogênea com importância derivada das estatísticas do Persona.
5. Quantização heterogênea com importância contrastiva, se a implementação for compatível.

Mantenha o método Persona original como baseline adicional quando sua tarefa e suas métricas forem comparáveis.

Avalie separadamente o domínio-alvo e cada domínio secundário.

Registre:

- métrica de desempenho da tarefa;
- degradação em relação ao modelo original;
- bytes reais do artefato;
- taxa de compressão;
- latência de inferência;
- memória de pico, quando mensurável;
- tempo de calibração e custo de execução;
- configuração e formatos efetivamente utilizados.

Use os mesmos dados de avaliação, seeds e condições de comparação sempre que possível. Não utilize os dados de teste para escolher a alocação de bits.

## 12. Testes e critérios de aceitação

Adicione testes para:

- cálculo das estatísticas de ativação;
- tratamento de padding;
- estimativa do erro de quantização;
- contabilização do armazenamento;
- validade das configurações geradas pelo alocador;
- serialização e carregamento do modelo comprimido;
- preservação do comportamento do Persona original;
- execução de um experimento pequeno de ponta a ponta.

O trabalho só deve ser considerado funcional quando for possível executar um experimento reproduzível e comparar pelo menos a baseline original com uma configuração de compressão.

## 13. Ordem de execução

Trabalhe em etapas pequenas:

1. Auditar e documentar o código original.
2. Executar e registrar a baseline original.
3. Implementar e testar a coleta de estatísticas.
4. Implementar a estimativa de erro.
5. Integrar um backend real de quantização.
6. Implementar a alocação heterogênea.
7. Executar os experimentos mínimos.
8. Investigar ablações e extensões somente após validar o pipeline.

Ao final de cada etapa, informe quais arquivos foram alterados, quais testes foram executados, quais limitações foram encontradas e como reproduzir o resultado.

**Prioridade máxima:** construir uma implementação pequena, verificável e cientificamente honesta a partir do código Persona disponível. Não reescreva o projeto inteiro, não implemente funcionalidades especulativas antes da baseline e não afirme ganhos de compressão ou desempenho sem medi-los.