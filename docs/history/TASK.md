# Prompt-base para Codex — Compressão Heterogênea de LLMs Orientada por Domínio

## 1. Objetivo

Implemente, neste repositório, um pipeline experimental de compressão de modelos de linguagem orientado por domínio, baseado em estatísticas de ativação, sensibilidade à quantização e alocação heterogênea de precisão por bloco de pesos.

A proposta deve preservar o desempenho de um domínio-alvo, permitindo compressão mais agressiva em componentes que toleram maior erro de quantização.

**Não comece reescrevendo o projeto.** Primeiro, inspecione a estrutura do repositório, identifique os módulos existentes, entenda o pipeline atual e reaproveite o máximo possível da implementação.

O código existente pode incluir coleta de estatísticas por persona, máscaras contrastivas, avaliação de múltipla escolha e geração. Preserve essas funcionalidades, salvo quando houver uma razão técnica clara para modificá-las.

## 2. Restrições de projeto

- Não treinar uma rede auxiliar.
- Não realizar fine-tuning do modelo principal.
- Não depender de gradientes para calcular a sensibilidade inicial.
- Usar calibração sem rótulos, quando possível.
- Manter o modelo original intacto durante a coleta de estatísticas e a avaliação das quantizações candidatas.
- Evitar armazenar matrizes de covariância completas quando estatísticas diagonais forem suficientes para o experimento inicial.
- Não carregar simultaneamente cópias desnecessárias do modelo na GPU.
- Não presumir que pesos zerados geram economia real de memória.
- Não presumir que a quantização de baixa precisão acelera a inferência sem suporte do formato, dos kernels e do hardware.
- Registrar sementes, configurações, tamanhos de calibração e métricas para permitir reprodução dos experimentos.

## 3. Antes de implementar

Execute uma inspeção inicial do repositório.

1. Identifique os arquivos de entrada e os scripts principais.
2. Localize as funções de coleta de ativações, construção de máscaras, quantização, avaliação e geração.
3. Identifique os formatos de dados de calibração e teste.
4. Verifique as versões e APIs das dependências instaladas.
5. Identifique quais formatos de quantização já são suportados.
6. Execute os testes existentes, se houver.
7. Apresente um plano curto de implementação e, em seguida, implemente as alterações.

Não invente módulos, argumentos CLI ou formatos de checkpoint sem verificar a estrutura existente.

## 4. Formulação matemática

Considere o modelo dividido em blocos de pesos:

\[
\theta = \{W_g\}_{g=1}^{G}
\]

Seja \(T\) o domínio-alvo e \(D_1,\ldots,D_m\) os domínios secundários.

Cada bloco recebe uma precisão \(b_g\), selecionada de um conjunto de precisões suportadas pelo backend, por exemplo:

\[
b_g \in \{2,4,8,16\}
\]

O objetivo global é:

\[
\min_{\{b_g\}} \sum_g B_g(b_g)
\]

sujeito a:

\[
\Delta L_T(\{b_g\}) \leq \varepsilon
\]

Onde:

- \(B_g(b_g)\) é o tamanho real armazenado do bloco na precisão escolhida.
- \(\Delta L_T\) é o aumento de perda no domínio-alvo em relação ao modelo original.
- \(\varepsilon\) é o limite de degradação permitido.

A otimização global exata não é necessária para a primeira versão. Implemente uma aproximação gulosa baseada em estimativas locais e valide o resultado no modelo completo.

## 5. Etapa A — Estatísticas de ativação por domínio

Para cada domínio \(d\), use um conjunto de calibração \(C_d\).

Para cada camada linear \(g\), colete as entradas \(h_g(x)\) e calcule, por canal:

\[
A_{g,d,j} =
\mathbb{E}_{x\sim C_d}
\left[|h_g(x)_j|\right]
\]

e:

\[
V_{g,d,j} =
\mathbb{E}_{x\sim C_d}
\left[h_g(x)_j^2\right]
\]

Requisitos:

- Acumular estatísticas incrementalmente, sem manter todas as ativações em memória.
- Agregar sobre tokens válidos e excluir padding.
- Garantir que as dimensões correspondam às colunas dos pesos da camada linear.
- Calcular as estatísticas em precisão numérica apropriada.
- Tratar sequências vazias, dados inválidos e domínios sem amostras.
- Remover hooks mesmo quando ocorrer uma exceção.
- Evitar acumular tensores na GPU desnecessariamente.
- Salvar as estatísticas e os metadados necessários para reutilizá-las.

A coleta deve funcionar em modo de inferência e não modificar os pesos do modelo.

## 6. Etapa B — Importância por domínio

Calcule a pontuação de importância por conexão:

\[
I_{g,d,ij} =
|W_{g,ij}|A_{g,d,j}
\]

Para um bloco \(g\), agregue:

\[
I_{g,d} =
\sum_{(i,j)\in g}
|W_{g,ij}|A_{g,d,j}
\]

Calcule também, quando houver domínio secundário:

\[
C_g = I_{g,T} - \lambda I_{g,S}
\]

O parâmetro \(\lambda\) deve ser configurável.

Essa pontuação contrastiva é um sinal auxiliar para análise e priorização. Não a use como único critério de escolha dos bits, porque importância por ativação não equivale necessariamente à sensibilidade à quantização.

Suporte a múltiplos domínios secundários deve ser possível sem exigir uma reescrita do algoritmo.

## 7. Etapa C — Sensibilidade à quantização

Para cada bloco \(g\) e precisão candidata \(b\):

\[
E_g^{(b)} = W_g - Q_b(W_g)
\]

Estime o erro local no domínio \(d\):

\[
D_{g,d}(b) =
\mathbb{E}_{x\sim C_d}
\left[
\left\|x(E_g^{(b)})^T\right\|_2^2
\right]
\]

Implemente inicialmente a aproximação diagonal:

\[
D_{g,d}(b) \approx
\sum_{i,j}
\left(W_{g,ij}-Q_b(W_{g,ij})\right)^2
V_{g,d,j}
\]

Requisitos:

- Reutilizar o backend de quantização existente, se houver.
- Se for necessário adicionar um quantizador, implementar uma interface modular e documentar o formato.
- Distinguir claramente quantização simulada de quantização efetivamente armazenada em baixa precisão.
- Calcular o erro após quantização e desquantização, conforme o formato pretendido.
- Não chamar uma simulação FP32/FP16 de compressão real.
- Armazenar os resultados por camada, domínio e precisão candidata.
- Evitar criar cópias completas do modelo para cada combinação.
- Documentar que a sensibilidade local é uma aproximação, não uma garantia de preservação da perda final.

Caso uma precisão candidata não seja suportada pelo backend, não a simule silenciosamente como se estivesse implementada. Registre a limitação e continue com as precisões realmente suportadas.

## 8. Etapa D — Alocação gulosa de bits

Para cada bloco e precisão, determine o custo real:

\[
B_g(b) = \text{bytes efetivamente armazenados}
\]

Inicialize os blocos na maior precisão suportada.

Para cada redução candidata \(b\rightarrow b'\), calcule:

\[
\rho_{g,b\rightarrow b'} =
\frac{D_{g,T}(b')-D_{g,T}(b)}
{B_g(b)-B_g(b')}
\]

Escolha a redução válida com menor aumento de erro estimado por byte economizado.

Repita até que nenhuma redução adicional respeite o orçamento de erro.

Requisitos:

- Nunca aceitar uma redução que economize zero bytes ou aumente o custo de armazenamento.
- Tratar divisões por zero e valores não finitos.
- Impedir que o erro estimado ultrapasse o orçamento.
- Registrar cada decisão de alocação.
- Permitir configurar o conjunto de precisões, o orçamento e a granularidade dos blocos.
- Salvar um plano de quantização reproduzível.
- Calcular o tamanho final a partir dos artefatos reais, incluindo escalas, metadados e outros overheads.
- Não assumir que a soma das sensibilidades locais corresponde exatamente à perda global.

O código deve separar a geração do plano de quantização da aplicação efetiva do plano ao modelo.

## 9. Etapa E — Formato de armazenamento

Implemente uma interface clara para:

1. Quantizar um bloco.
2. Salvar os pesos quantizados.
3. Salvar escalas e metadados.
4. Carregar os pesos quantizados.
5. Executar inferência usando um backend compatível.
6. Calcular o tamanho real dos arquivos.
7. Comparar a saída do modelo quantizado com a do modelo original.

Se o repositório já usa NF4, investigue como integrar o plano de alocação heterogênea ao backend existente.

Não implemente um formato proprietário complexo antes de verificar as bibliotecas e os formatos já disponíveis.

Se não for possível executar inferência diretamente com o formato escolhido, documente essa limitação e mantenha a simulação de quantização como uma etapa experimental separada.

## 10. Etapa F — Correções de baixa ordem

Não faça desta etapa um requisito para a primeira versão funcional.

Depois de validar a alocação de bits, investigue uma correção de baixa ordem:

\[
W_g \approx Q_b(W_g) + R_{g,T}
\]

com:

\[
R_{g,T}=U_gV_g^T
\]

A correção deve minimizar aproximadamente:

\[
\left\|
\left(W_g-Q_b(W_g)-R_{g,T}\right)
C_{g,T}^{1/2}
\right\|_F^2
\]

Use uma aproximação de baixa ordem, sem treinar uma rede auxiliar.

Compare a economia de armazenamento da base quantizada com o custo adicional da correção. Não mantenha a correção se ela eliminar o benefício líquido da compressão.

## 11. Interfaces de linha de comando

Inspecione a CLI atual e preserve os comandos existentes.

Adicione, adaptando a nomenclatura do projeto, comandos equivalentes a:

- `collect_stats`: coletar estatísticas por domínio.
- `estimate_quant_error`: estimar erros para cada bloco e precisão.
- `allocate_bits`: construir o plano de alocação.
- `quantize_model`: materializar o modelo conforme o plano.
- `evaluate_compression`: avaliar qualidade, armazenamento e latência.

Cada comando deve validar os argumentos, emitir logs úteis e retornar erros compreensíveis.

Sempre que possível, reutilize as convenções de argumentos e diretórios existentes, evitando duplicar funcionalidade.

## 12. Avaliação experimental

Compare pelo menos:

1. Modelo original BF16.
2. Quantização uniforme NF4, se suportada pelo backend.
3. Alocação heterogênea de bits orientada por sensibilidade.
4. Alocação heterogênea com importância contrastiva como sinal auxiliar.

Avalie:

- Desempenho no domínio-alvo.
- Desempenho em cada domínio secundário.
- Perda de validação.
- Tamanho real dos artefatos.
- Memória de pico.
- Latência de inferência.
- Tempo de calibração e preparação.

Use os mesmos conjuntos de avaliação para todos os métodos.

Inclua uma comparação com orçamento de armazenamento semelhante sempre que possível. Não atribua melhorias à importância contrastiva sem uma ablação específica.

A avaliação deve distinguir explicitamente:

- Resultados medidos.
- Estimativas de erro local.
- Resultados simulados.
- Resultados de inferência com pesos realmente quantizados.

## 13. Testes obrigatórios

Adicione testes para:

- Dimensões e valores das estatísticas de ativação.
- Tratamento de padding e sequências vazias.
- Compatibilidade entre dimensões dos pesos e estatísticas.
- Monotonicidade aproximada do erro ao variar a precisão, quando esperada para o quantizador escolhido.
- Cálculo de bytes e overhead.
- Alocação gulosa dentro do orçamento.
- Reprodutibilidade com a mesma semente.
- Salvamento e carregamento do plano de quantização.
- Preservação dos pesos originais durante a coleta de estatísticas.
- Inferência básica após quantização, quando o backend estiver disponível.

Use modelos e tensores pequenos nos testes unitários. Não baixe modelos grandes nem execute experimentos caros automaticamente durante os testes.

## 14. Critérios de aceitação

A implementação será considerada pronta quando:

- O pipeline executar do início ao fim em uma configuração pequena.
- A coleta de estatísticas funcionar para os domínios configurados.
- O plano de bits for salvo e reproduzível.
- O tamanho final for medido, e não apenas estimado a partir de bits por parâmetro.
- Os resultados puderem ser comparados com a quantização uniforme.
- Os testes cobrirem os componentes centrais.
- As limitações de hardware e backend estiverem documentadas.
- O relatório final distinguir resultados medidos de hipóteses ainda não validadas.

## 15. Forma de trabalho

Implemente em etapas pequenas.

Após a inspeção inicial:

1. Informe quais arquivos serão modificados e por quê.
2. Implemente a coleta de estatísticas.
3. Adicione os testes correspondentes.
4. Implemente a estimativa de sensibilidade.
5. Implemente a alocação gulosa.
6. Integre o backend de quantização existente.
7. Execute testes pequenos e corrija os erros encontrados.
8. Execute uma avaliação mínima, somente se o ambiente e os dados necessários estiverem disponíveis.
9. Atualize a documentação com comandos de execução, formatos, limitações e exemplos reais.

Não afirme que um teste foi executado se ele não foi executado. Não invente resultados experimentais.

**Prioridade final:** entregar primeiro uma implementação mínima, correta, mensurável e reproduzível da alocação heterogênea de bits. Só depois adicionar correções de baixa ordem ou mecanismos de especialização mais complexos.
