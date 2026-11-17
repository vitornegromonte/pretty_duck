# Revisão jurídica dos fatos em lacuna

Este arquivo pede a ajuda de quem tem formação em direito brasileiro. Os fatos em
`fatos_direito_br_rascunho.csv` são um **rascunho** feito sem revisão: cada linha é uma
frase com uma lacuna, e o modelo de linguagem deve completar a primeira palavra da resposta.
Eles servem para testar se o modelo guarda esses conhecimentos em circuitos localizáveis.

## O que revisar

Para cada linha, confira a coluna `resposta_completa` e preencha `status`:

- `ok`: o fato está correto e inequívoco;
- `corrigir`: está errado ou impreciso (escreva a correção na coluna `verificacao`);
- `remover`: ambíguo, desatualizado ou controverso.

Colunas: `relacao` (grupo de fatos com o mesmo modelo de frase), `sujeito` (o que entra na
lacuna `{}`), `resposta_completa` (a resposta certa), `resposta_primeira_palavra` (a palavra
que o teste realmente verifica), `verificacao` (como conferir).

## O que acrescentar (o mais valioso)

Hoje são só 60 fatos em 3 relações. Precisamos de pelo menos **200**, em relações novas.
Critérios:

1. A resposta começa por **uma palavra** (sem número; o modelo separa dígitos e o teste perde sentido).
2. O fato é estável e a resposta é única (sem divergência entre doutrinas).
3. Todas as linhas de uma relação usam o **mesmo modelo de frase**, só trocando o sujeito.
4. Prefira conteúdo que cai na OAB: competências, princípios, institutos, órgãos, remédios constitucionais.
5. Evite duas linhas da mesma relação com a mesma primeira palavra de resposta em excesso.

Se possível, acrescente também uma versão do mesmo fato em outra formulação (para testarmos se o
circuito depende da frase).

Quem revisar: devolva o CSV com a coluna `status` preenchida. Não precisa saber programação.
