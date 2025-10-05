# EVOHealthAgent: Agente Autônomo para Simulações Evolutivas em Saúde com Interpretabilidade

# Descrição:

Um agente autônomo inovador que integra simulações de biologia evolutiva/bioquímica com técnicas de Machine Learning interpretável (SHAP) para otimizar pesquisas em saúde, com foco inicial na evolução da resistência antimicrobiana. O agente utiliza um loop de feedback com um Large Language Model (LLM) para adaptar suas estratégias de simulação e análise, buscando padrões significativos em sequências genéticas.

Singularidade: Combinação de Algoritmo Genético(GA), ML interpretável (SHAP), LLM feedback loop e bases de dados como CARD.

# Funcionalidades Chave:

Simulações Evolutivas Adaptativas: Implementa algoritmos genéticos (mutação, crossover, seleção) para simular a evolução de patógenos.
Interpretabilidade de ML com SHAP: Utiliza modelos de Machine Learning (Árvores de Decisão) e SHAP para identificar k-mers (características genéticas) mais relevantes na predição de resistência, tornando o modelo transparente.
Agente Autônomo com LLM (Simulado): Um loop de feedback impulsionado por um LLM simula a tomada de decisões, ajustando hiperparâmetros de simulação e ML, e refinando queries de pesquisa com base nos resultados das iterações.
Integração com Bases de Conhecimento (CARD simulada): Simula a consulta à base de dados CARD (Comprehensive Antibiotic Resistance Database) para associar k-mers relevantes a mecanismos de resistência conhecidos.
Geração de Relatórios Interativos: Gera relatórios JSON e HTML detalhados por iteração, incluindo visualizações SHAP e um resumo da análise do LLM.

# Como Funciona (Visão Geral)

O projeto segue o seguinte ciclo iterativo: Geração de sequências -> Treinamento ML -> Análise SHAP -> Consulta CARD -> Feedback LLM -> Nova iteração.

# Instalação e Configuração

Pré-requisitos: Python 3.8+
Clonar o repositório:git clone https://github.com/Lucas-medeiros29/EvoHealthAgent.git
cd EvoHealthAgent

Criar ambiente virtual (recomendado):python -m venv venv
source venv/bin/activate  # No Windows: venv\Scripts\activate

Instalar dependências:pip install -r requirements.txt

Configuração da Chave da API do NCBI (Opcional):
Para simulação PubMed real, uma chave no NCBI é necessária, mas para a demonstração atual, a busca é simulada.


# Como Usar:

Executar o agente:python src/evohealth_agent.py

Será solicitada a chave NCBI e então iniciará as iterações. Caso não tenha, apenas deixe em branco.

# Resultados e Visualizações:

Descrição dos arquivos gerados:

evohealth_report_iter_X.html >> Gera um relatório individual em html para cada iteração da simulação.

evohealth_overall_report.html >> Gera um relatório geral da simulação em html.

accuracy_over_iterations.png >> Gera uma imagem com acurácias observadas para cada iteração.


Os relatórios HTML contêm os gráficos SHAP e os sumários das análises do LLM para cada iteração.

Segue uma imagem do gráfico SHAP mais recente e da acurácia ao longo das iterações para dar um "sabor" visual imediato do que o projeto faz:


# Próximos Passos e Melhorias Futuras:

Integração Real com PubMed/NCBI: Conectar a busca por artigos científicos a APIs reais.
Expansão da Base CARD: Integrar dados mais complexos e reais da CARD ou outras bases.
Modelos de ML Mais Avançados: Experimentar com Random Forests, Gradient Boosting, LSTMs para sequências.
Otimização do LLM: Refinar os prompts e a lógica de feedback para respostas mais sofisticadas e adaptativas.
Interface Gráfica (GUI): Desenvolver uma interface para visualização interativa das simulações.
Geração de Sequências Mais Realistas: Usar modelos generativos (ex: GANs, VAEs) para criar sequências mais próximas da realidade biológica.

# Contribuições:

Contribuições são muito bem vindas! Fiquem a vontade para enviar pull requests, issues e sugestões para o projeto.

# Licença:

Licença de código aberto (ex: MIT, Apache 2.0). Crie um arquivo LICENSE no diretório raiz.
