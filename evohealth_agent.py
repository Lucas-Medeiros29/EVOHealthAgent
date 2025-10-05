pip install numpy pandas requests biopython scikit-learn shap matplotlib

import random
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import json
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import xml.etree.ElementTree as ET
import requests
import time
from IPython.display import display, HTML
import base64
from io import BytesIO
import os # Importar para manipulação de caminhos de arquivo

# --- Configurações Globais ---
K_MER_SIZE = 3
SEQUENCE_LENGTH = 100
GENERATION_LENGTH = 50 # Quantidade de bases para o AG simular por geração
POPULATION_SIZE = 20
NUM_GENERATIONS = 10
AMINO_ACIDS = ['A', 'T', 'C', 'G']

# --- Agente EvoHealth Autônomo ---
class EvoHealthAgent:
    def __init__(self):
        self.iteration_count = 0
        self.all_simulation_reports = []
        self.accuracy_history = []
        self.top_kmers_history = []
        self.pubmed_api_key = None
        # Base de conhecimento CARD simulada (para evitar a API externa)
        self.card_knowledge_base = {
            "GGG": [
                {"ARO": "ARO:3000000", "name": "aminoglycoside resistance", "mechanism": "ribosomal protection"},
                {"ARO": "ARO:3000001", "name": "beta-lactamase", "mechanism": "beta-lactam hydrolysis"}
            ],
            "ATT": [
                {"ARO": "ARO:3000002", "name": "macrolide efflux pump", "mechanism": "efflux of macrolides"}
            ],
            "CGA": [
                {"ARO": "ARO:3000003", "name": "fluoroquinolone resistance", "mechanism": "DNA gyrase mutation"}
            ],
            "TTC": [
                {"ARO": "ARO:3000004", "name": "tetracycline resistance", "mechanism": "ribosomal protection protein"}
            ],
            "GGC": [
                {"ARO": "ARO:3000005", "name": "vancomycin resistance", "mechanism": "cell wall precursor modification"}
            ],
            "ACA": [
                {"ARO": "ARO:3000006", "name": "rifampicin resistance", "mechanism": "RNA polymerase mutation"}
            ]
            # Adicione mais associações k-mer -> ARO/mecanismo conforme desejar
        }


    # --- Funções Auxiliares Genéticas ---
    def _generate_random_sequence(self, length):
        return ''.join(random.choice(AMINO_ACIDS) for _ in range(length))

    def _generate_initial_population(self, sequence, size):
        return [list(sequence) for _ in range(size)]

    def _calculate_fitness(self, individual):
        # Para esta simulação, vamos simplificar a fitness
        # Poderíamos ter uma função mais complexa baseada em mutações específicas
        # ou padrões de resistência. Por enquanto, é um valor dummy.
        return random.random()

    def _apply_mutation(self, individual, mutation_rate):
        for i in range(len(individual)):
            if random.random() < mutation_rate:
                individual[i] = random.choice(AMINO_ACIDS)
        return individual

    def _apply_crossover(self, parent1, parent2, crossover_rate):
        if random.random() < crossover_rate:
            # Ponto de crossover aleatório
            crossover_point = random.randint(1, len(parent1) - 1)
            child1 = parent1[:crossover_point] + parent2[crossover_point:]
            child2 = parent2[:crossover_point] + parent1[crossover_point:]
            return child1, child2
        return parent1, parent2

    def _run_genetic_algorithm(self, initial_sequence, mutation_rate, crossover_rate):
        population = self._generate_initial_population(initial_sequence, POPULATION_SIZE)
        best_individual = None
        best_fitness = -1

        for _ in range(NUM_GENERATIONS):
            new_population = []
            population.sort(key=lambda x: self._calculate_fitness(x), reverse=True) # Ordena pela fitness

            # Elitismo: Mantém os dois melhores indivíduos
            new_population.extend(population[:2])

            # Preenche o restante da nova população com crossover e mutação
            while len(new_population) < POPULATION_SIZE:
                parent1, parent2 = random.choices(population, k=2)
                child1, child2 = self._apply_crossover(parent1, parent2, crossover_rate)
                new_population.append(self._apply_mutation(child1, mutation_rate))
                if len(new_population) < POPULATION_SIZE:
                    new_population.append(self._apply_mutation(child2, mutation_rate))
            population = new_population

            current_best_individual = population[0] # Melhor da geração
            current_best_fitness = self._calculate_fitness(current_best_individual)
            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                best_individual = current_best_individual

        return "".join(best_individual if best_individual else population[0]) # Retorna o melhor indivíduo encontrado


    # --- Funções de Geração de Dados para ML (Mais Realistas) ---
    def _extract_kmers(self, sequence, k):
        return [sequence[i:i+k] for i in range(len(sequence) - k + 1)]

    def _generate_ml_data_default(self, num_samples, k_mer_size):
        X, y = [], []
        # Gera um conjunto de features de k-mers a partir de uma sequência longa para garantir representatividade
        kmer_features = sorted(list(set(self._extract_kmers(self._generate_random_sequence(SEQUENCE_LENGTH * 5), k_mer_size))))

        for _ in range(num_samples // 2): # Classe 0 (Não resistente)
            sequence = self._generate_random_sequence(SEQUENCE_LENGTH)
            # Inserir alguns k-mers "ruído"
            num_noise_kmers = random.randint(0, 5)
            for _ in range(num_noise_kmers):
                idx = random.randint(0, SEQUENCE_LENGTH - k_mer_size)
                # Exemplos de k-mers não resistentes ou comuns, com probabilidade de sobreposição com k-mers de foco
                # para aumentar a dificuldade
                noise_kmer_options = ['CAT', 'TAG', 'ATC', 'AAA', 'TTT', 'CCC', 'GGG'] # Incluindo alguns dos de foco como ruído para simular dificuldade
                sequence = sequence[:idx] + random.choice(noise_kmer_options) + sequence[idx+k_mer_size:]
            X.append(sequence)
            y.append(0)

        for _ in range(num_samples // 2): # Classe 1 (Resistente)
            sequence = self._generate_random_sequence(SEQUENCE_LENGTH)
            # Inserir k-mers "resistência" com alguma variabilidade e sobreposição
            resistance_kmers_pool = ['GGG', 'ATT', 'CGA', 'TTC', 'GGC', 'ACA'] # k-mers associados a resistência

            # Garante que pelo menos um k-mer de resistência seja inserido para a classe 1
            num_resistance_kmers = random.randint(1, 6)
            resistance_kmers_to_insert = random.sample(resistance_kmers_pool, num_resistance_kmers)

            for kmer in resistance_kmers_to_insert:
                idx = random.randint(0, SEQUENCE_LENGTH - k_mer_size)
                sequence = sequence[:idx] + kmer + sequence[idx+k_mer_size:]

            # Inserir algum ruído também para dificultar
            num_noise_kmers = random.randint(0, 3)
            for _ in range(num_noise_kmers):
                idx = random.randint(0, SEQUENCE_LENGTH - k_mer_size)
                noise_kmer_options = ['CAT', 'TAG', 'ATC', 'AAA', 'TTT', 'CCC']
                sequence = sequence[:idx] + random.choice(noise_kmer_options) + sequence[idx+k_mer_size:]
            X.append(sequence)
            y.append(1)

        # Vetorização
        X_vectorized = []
        for seq in X:
            kmer_counts = {kmer: seq.count(kmer) for kmer in kmer_features}
            X_vectorized.append([kmer_counts.get(k, 0) for k in kmer_features])

        return np.array(X_vectorized), np.array(y), kmer_features

    def _generate_ml_data_focused(self, num_samples, k_mer_size, focus_kmers):
        X, y = [], []
        # Garantir que os k-mers de foco estejam entre as features e adicionar outros k-mers comuns
        all_possible_kmers = sorted(list(set(self._extract_kmers(self._generate_random_sequence(SEQUENCE_LENGTH * 5), k_mer_size))))
        # Certifica-se de que `focus_kmers` são strings e não tuplas (k-mer, shap_value)
        cleaned_focus_kmers = [k[0] if isinstance(k, tuple) else k for k in focus_kmers]
        kmer_features = sorted(list(set(cleaned_focus_kmers + all_possible_kmers)))

        for _ in range(num_samples // 2): # Classe 0 (Não resistente)
            sequence = self._generate_random_sequence(SEQUENCE_LENGTH)
            # Menor chance de ter k-mers de foco
            for f_kmer in cleaned_focus_kmers:
                if random.random() < 0.15: # Pequena chance de contaminação com k-mer de foco
                    idx = random.randint(0, SEQUENCE_LENGTH - k_mer_size)
                    sequence = sequence[:idx] + f_kmer + sequence[idx+k_mer_size:]
            # Inserir k-mers ruído para aumentar a dificuldade
            num_noise_kmers = random.randint(1, 4)
            for _ in range(num_noise_kmers):
                idx = random.randint(0, SEQUENCE_LENGTH - k_mer_size)
                sequence = sequence[:idx] + random.choice(['CAT', 'TAG', 'ATC', 'TCA', 'AGA', 'CTC']) + sequence[idx+k_mer_size:]
            X.append(sequence)
            y.append(0)

        for _ in range(num_samples // 2): # Classe 1 (Resistente)
            sequence = self._generate_random_sequence(SEQUENCE_LENGTH)
            # Maior chance de ter k-mers de foco
            num_focus_insertions = random.randint(2, 7) # Mais inserções dos k-mers de foco
            for _ in range(num_focus_insertions):
                f_kmer = random.choice(cleaned_focus_kmers)
                idx = random.randint(0, SEQUENCE_LENGTH - k_mer_size)
                sequence = sequence[:idx] + f_kmer + sequence[idx+k_mer_size:]
            # Inserir algum ruído também para dificultar
            num_noise_kmers = random.randint(0, 3)
            for _ in range(num_noise_kmers):
                idx = random.randint(0, SEQUENCE_LENGTH - k_mer_size)
                sequence = sequence[:idx] + random.choice(['CAT', 'TAG', 'ATC', 'TCA', 'AGA', 'CTC']) + sequence[idx+k_mer_size:]
            X.append(sequence)
            y.append(1)

        # Vetorização
        X_vectorized = []
        for seq in X:
            kmer_counts = {kmer: seq.count(kmer) for kmer in kmer_features}
            X_vectorized.append([kmer_counts.get(k, 0) for k in kmer_features])

        return np.array(X_vectorized), np.array(y), kmer_features

    def _train_ml_model(self, X, y, kmer_features, max_depth, min_samples_leaf):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        print(f"DEBUG ML: X_train shape: {X_train.shape}")
        print(f"DEBUG ML: X_test shape: {X_test.shape}")
        print(f"DEBUG ML: Length of kmer_features: {len(kmer_features)}")

        # Adiciona uma verificação para garantir que X_test tenha amostras e features
        if X_test.shape[0] == 0 or X_test.shape[1] == 0:
            print(f"WARNING: X_test tem shape {X_test.shape}. Não é possível calcular valores SHAP significativos. Retornando valores dummy.")
            return None, 0.0, [], pd.DataFrame({'kmer': [], 'shap_value': []}), X_test # Retorna None para o modelo, 0 para acurácia

        model = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # SHAP para interpretar o modelo
        explainer = shap.TreeExplainer(model)
        shap_values_raw = explainer.shap_values(X_test) # Renomeado para evitar conflito

        selected_shap_values = None

        if isinstance(shap_values_raw, list):
            # Para classificadores multi-classe ou binários que retornam uma lista de arrays SHAP
            print(f"DEBUG SHAP: shap_values_raw é uma lista. Comprimento: {len(shap_values_raw)}")
            # Geralmente, shap_values[0] é para a classe 0 (negativa) e shap_values[1] para a classe 1 (positiva)
            if len(shap_values_raw) > 1:
                selected_shap_values = shap_values_raw[1] # Pega os valores SHAP para a classe positiva (índice 1)
                print(f"DEBUG SHAP: Selecionado shap_values para a classe 1 (da lista). Shape: {selected_shap_values.shape}")
            else:
                print(f"WARNING: shap_values_raw é uma lista, mas não tem valores para a classe 1. Shape: {shap_values_raw[0].shape if shap_values_raw else 'N/A'}")
                return model, accuracy, [], pd.DataFrame({'kmer': [], 'shap_value': []}), X_test
        elif isinstance(shap_values_raw, np.ndarray):
            # Para classificadores que retornam um array 2D ou 3D diretamente
            print(f"DEBUG SHAP: shap_values_raw é diretamente um array. Shape: {shap_values_raw.shape}")
            if shap_values_raw.ndim == 3 and shap_values_raw.shape[2] > 1:
                # É um array 3D (samples, features, classes), pega a slice para a classe 1
                selected_shap_values = shap_values_raw[:, :, 1]
                print(f"DEBUG SHAP: Selecionado shap_values para a classe 1 do array 3D. Shape: {selected_shap_values.shape}")
            elif shap_values_raw.ndim == 2:
                # É um array 2D (samples, features), comum para modelos de regressão ou binários simples que dão um output direto.
                # Para classificação, pode ser o output da classe 1 se o modelo for configurado para isso.
                # Assumimos que, se é 2D, já são os valores para a classe de interesse, ou para um modelo que não tem distinção explícita de classes em SHAP.
                selected_shap_values = shap_values_raw
                print(f"DEBUG SHAP: Usando shap_values 2D diretamente. Shape: {selected_shap_values.shape}")
            else:
                print(f"WARNING: Formato inesperado para shap_values_raw (ndarray com ndim={shap_values_raw.ndim}). Shape: {shap_values_raw.shape}")
                return model, accuracy, [], pd.DataFrame({'kmer': [], 'shap_value': []}), X_test
        else:
            print(f"WARNING: Tipo inesperado para shap_values_raw: {type(shap_values_raw)}. Não é possível calcular valores SHAP significativos.")
            return model, accuracy, [], pd.DataFrame({'kmer': [], 'shap_value': []}), X_test

        if selected_shap_values is None or selected_shap_values.shape[0] == 0 or selected_shap_values.shape[1] == 0:
            print(f"WARNING: selected_shap_values está vazio ou malformado após a seleção. Shape: {selected_shap_values.shape if selected_shap_values is not None else 'None'}. Não é possível calcular valores SHAP significativos.")
            return model, accuracy, [], pd.DataFrame({'kmer': [], 'shap_value': []}), X_test

        # Garante que selected_shap_values tenha o número esperado de features
        if selected_shap_values.shape[1] != len(kmer_features):
            raise ValueError(f"Contagem de features incompatível entre SHAP ({selected_shap_values.shape[1]}) e k-mers ({len(kmer_features)}). Verifique a geração de dados e features.")

        # Média absoluta dos valores SHAP por feature
        avg_shap_values = np.abs(selected_shap_values).mean(axis=0)

        print(f"DEBUG SHAP: Tipo de avg_shap_values: {type(avg_shap_values)}, shape: {avg_shap_values.shape}")
        print(f"DEBUG SHAP: Tipo de kmer_features: {type(kmer_features)}, comprimento: {len(kmer_features)}")

        # Verifica explicitamente se as dimensões de avg_shap_values e kmer_features são compatíveis
        if avg_shap_values.shape[0] != len(kmer_features):
            raise ValueError(f"Dimensão incompatível entre valores SHAP e k-mers. Features SHAP: {avg_shap_values.shape[0]}, Features K-mer: {len(kmer_features)}")

        # Cria um dataframe para mapear k-mers com seus valores SHAP
        shap_df = pd.DataFrame({'kmer': kmer_features, 'shap_value': avg_shap_values})
        shap_df = shap_df.sort_values(by='shap_value', ascending=False)

        top_k_mers_shap = [(row['kmer'], row['shap_value']) for index, row in shap_df.head(5).iterrows()]

        return model, accuracy, top_k_mers_shap, shap_df, X_test # Retorna X_test para o plot SHAP

    # --- Funções de Busca e Integração de Conhecimento ---
    def search_pubmed(self, query):
        if not self.pubmed_api_key:
            print("Chave da API do NCBI não fornecida. Pulando busca no PubMed.")
            return []

        # Usar o endpoint esearch para encontrar IDs de artigos
        esearch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        esearch_params = {
            "db": "pubmed",
            "term": query,
            "retmax": 5, # Limita a 5 resultados
            "usehistory": "y",
            "api_key": self.pubmed_api_key
        }

        try:
            response = requests.get(esearch_url, params=esearch_params)
            response.raise_for_status()
            esearch_tree = ET.fromstring(response.content)

            id_list = [element.text for element in esearch_tree.findall(".//IdList/Id")]

            if not id_list:
                print(f"Nenhum artigo encontrado no PubMed para a query: '{query}'.")
                return []

            # Usar o endpoint efetch para obter detalhes dos artigos
            efetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
            efetch_params = {
                "db": "pubmed",
                "id": ",".join(id_list),
                "retmode": "xml",
                "api_key": self.pubmed_api_key
            }

            time.sleep(0.5) # Pausa para evitar sobrecarga da API
            efetch_response = requests.get(efetch_url, params=efetch_params)
            efetch_response.raise_for_status()
            efetch_tree = ET.fromstring(efetch_response.content)

            articles = []
            for article in efetch_tree.findall(".//PubmedArticle"):
                title_element = article.find(".//Article/ArticleTitle") # Caminho corrigido
                abstract_element = article.find(".//Article/Abstract/AbstractText") # Caminho corrigido
                pmid_element = article.find(".//MedlineCitation/PMID") # Caminho corrigido

                title = title_element.text if title_element is not None else "N/A"
                abstract = abstract_element.text if abstract_element is not None else "N/A"
                pmid = pmid_element.text if pmid_element is not None else "N/A"

                articles.append({
                    "title": title,
                    "abstract": abstract,
                    "pmid": pmid
                })
            return articles
        except requests.exceptions.RequestException as e:
            print(f"Erro na busca no PubMed: {e}")
            return []
        except ET.ParseError as e:
            print(f"Erro ao parsear XML do PubMed: {e}")
            print(f"Conteúdo XML que causou o erro: {response.content.decode('utf-8') if 'response' in locals() else 'N/A'}")
            return []

    def fetch_from_card(self, top_kmers_shap):
        print("Buscando informações no CARD (base de conhecimento simulada)...")
        findings = []
        for kmer_tuple in top_kmers_shap:
            kmer = kmer_tuple[0] # Pega apenas a string do k-mer
            if kmer in self.card_knowledge_base:
                for entry in self.card_knowledge_base[kmer]:
                    findings.append({"kmer": kmer, **entry})
        return findings

    # --- Funções de Geração de Relatórios e Plots ---
    def _generate_shap_plot_base64(self, model, X_test_for_plot, kmer_features, iteration_num):
        if model is None or X_test_for_plot.shape[0] == 0 or X_test_for_plot.shape[1] == 0:
            print(f"Não foi possível gerar o plot SHAP para a iteração {iteration_num} devido à falta de dados ou modelo inválido.")
            return None

        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test_for_plot)

            # Lógica para obter a classe positiva (similar à função _train_ml_model)
            selected_shap_values = None
            if isinstance(shap_values, list) and len(shap_values) > 1:
                selected_shap_values = shap_values[1] # Classe positiva
            elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3 and shap_values.shape[2] > 1:
                selected_shap_values = shap_values[:, :, 1] # Classe positiva
            elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 2:
                selected_shap_values = shap_values # Assumimos que já é para a classe positiva

            if selected_shap_values is None or selected_shap_values.shape[0] == 0 or selected_shap_values.shape[1] == 0:
                 print(f"Formato inesperado dos valores SHAP após seleção para o plot. Não foi possível gerar o plot.")
                 return None

            plt.figure(figsize=(10, 7))

            # Use shap.summary_plot diretamente. `max_display` para limitar features.
            # `color_bar_label` para customizar a legenda da cor.
            shap.summary_plot(selected_shap_values, X_test_for_plot, feature_names=kmer_features,
                              show=False, plot_type="dot", max_display=15,
                              color_bar_label="Contagem do K-mer")

            plt.title(f'Impacto dos K-mers na Predição de Resistência (SHAP) - Iteração {iteration_num}')
            plt.tight_layout()

            buf = BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            plt.close()
            return base64.b64encode(buf.getvalue()).decode('utf-8')
        except Exception as e:
            print(f"Erro ao gerar o plot SHAP para a iteração {iteration_num}: {e}")
            return None

    def _generate_html_report_content(self, report_data):
        shap_plot_html = ""
        if report_data['shap_plot_base64']:
            shap_plot_html = f"""
            <h3>Plot SHAP</h3>
            <img src="data:image/png;base64,{report_data['shap_plot_base64']}" alt="SHAP Plot" style="max-width:100%;">
            """
        else:
            shap_plot_html = "<p>Plot SHAP não gerado devido a dados insuficientes ou erro.</p>"

        pubmed_articles_html = "<h4>Artigos do PubMed</h4>"
        if report_data['pubmed_articles']:
            pubmed_articles_html += "<ul>"
            for article in report_data['pubmed_articles']:
                pubmed_articles_html += f"""
                <li>
                    <strong>PMID:</strong> {article['pmid']}<br>
                    <strong>Título:</strong> {article['title']}<br>
                    <strong>Abstract:</strong> {article['abstract'][:200]}...<br>
                    <a href="https://pubmed.ncbi.nlm.nih.gov/{article['pmid']}/" target="_blank">Ver no PubMed</a>
                </li>
                """
            pubmed_articles_html += "</ul>"
        else:
            pubmed_articles_html += "<p>Nenhum artigo relevante encontrado no PubMed.</p>"

        card_findings_html = "<h4>Achados no CARD</h4>"
        if report_data['card_findings']:
            card_findings_html += "<ul>"
            for finding in report_data['card_findings']:
                card_findings_html += f"""
                <li>
                    <strong>K-mer:</strong> {finding['kmer']}<br>
                    <strong>ARO:</strong> {finding['ARO']}<br>
                    <strong>Nome:</strong> {finding['name']}<br>
                    <strong>Mecanismo:</strong> {finding['mechanism']}
                </li>
                """
            card_findings_html += "</ul>"
        else:
            card_findings_html += "<p>Nenhum achado relevante na base de conhecimento CARD.</p>"

        top_kmers_shap_html = "<ul>"
        if report_data['top_kmers_shap']:
            for kmer, shap_value in report_data['top_kmers_shap']:
                top_kmers_shap_html += f"<li>{kmer} (Impacto SHAP: {shap_value:.4f})</li>"
        else:
            top_kmers_shap_html += "<li>Nenhum k-mer principal identificado.</li>"
        top_kmers_shap_html += "</ul>"

        llm_guidance = report_data['llm_response'] # Já é um dicionário aqui

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Relatório EvoHealth - Iteração {report_data['iteration']}</title>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 20px; }}
                .container {{ max-width: 1000px; margin: auto; background: #f4f4f4; padding: 20px; border-radius: 8px; }}
                h1, h2, h3, h4 {{ color: #333; }}
                .section {{ background: #fff; padding: 15px; margin-bottom: 10px; border-radius: 5px; border: 1px solid #ddd; }}
                .plot-container {{ text-align: center; margin-top: 20px; }}
                table {{ width: 100%; border-collapse: collapse; margin-top: 15px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                pre {{ background-color: #eee; padding: 10px; border-radius: 5px; overflow-x: auto; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Relatório de Simulação EvoHealth - Iteração {report_data['iteration']}</h1>
                <div class="section">
                    <h2>Sumário da Iteração</h2>
                    <p><strong>Timestamp:</strong> {report_data['timestamp']}</p>
                    <p><strong>Sequência Inicial:</strong> {report_data['initial_sequence'][:50]}...</p>
                    <p><strong>Sequência Evoluída:</strong> {report_data['evolved_sequence'][:50]}...</p>
                    <p><strong>Taxa de Mutação:</strong> {report_data['mutation_rate']:.4f}</p>
                    <p><strong>Taxa de Crossover:</strong> {report_data['crossover_rate']:.2f}</p>
                    <p><strong>Hiperparâmetros ML:</strong> Max Depth={report_data['ml_hyperparameters']['max_depth']}, Min Samples Leaf={report_data['ml_hyperparameters']['min_samples_leaf']}</p>
                    <p><strong>Acurácia do Modelo ML:</strong> {report_data['ml_accuracy']:.3f}</p>
                </div>

                <div class="section">
                    <h2>Interpretabilidade do Modelo</h2>
                    <h4>Top 5 K-mers por Impacto (SHAP)</h4>
                    {top_kmers_shap_html}
                    {shap_plot_html}
                </div>

                <div class="section">
                    <h2>Pesquisa de Conhecimento</h2>
                    <h3>PubMed</h3>
                    <p><strong>Query Utilizada:</strong> {report_data['pubmed_query']}</p>
                    {pubmed_articles_html}
                    <h3>CARD (Base de Conhecimento Simulada)</h3>
                    {card_findings_html}
                </div>

                <div class="section">
                    <h2>Análise e Orientação do Agente Autônomo (LLM)</h2>
                    <h4>Sumário da Análise</h4>
                    <p>{llm_guidance.get('llm_analysis_summary', 'N/A')}</p>
                    <h4>Próximas Decisões do Agente</h4>
                    <ul>
                        <li><strong>Próxima Sequência Inicial:</strong> {llm_guidance.get('next_initial_sequence', 'N/A')[:50]}...</li>
                        <li><strong>Próxima Query PubMed:</strong> {llm_guidance.get('next_pubmed_query', 'N/A')}</li>
                        <li><strong>Próxima Estratégia de ML:</strong> {llm_guidance.get('next_strategy_ml', 'N/A')}</li>
                        <li><strong>Próxima Taxa de Mutação:</strong> {llm_guidance.get('next_mutation_rate', 'N/A'):.4f}</li>
                        <li><strong>Próxima Taxa de Crossover:</strong> {llm_guidance.get('next_crossover_rate', 'N/A'):.2f}</li>
                        <li><strong>Próximos Hiperparâmetros ML:</strong> Max Depth={llm_guidance.get('next_ml_hyperparameters', {}).get('max_depth', 'N/A')}, Min Samples Leaf={llm_guidance.get('next_ml_hyperparameters', {}).get('min_samples_leaf', 'N/A')}</li>
                    </ul>
                </div>
            </div>
        </body>
        </html>
        """
        return html_content

    def _generate_overall_html_report(self):
        # Gera o plot de acurácia em base64
        accuracy_plot_base64 = self._generate_accuracy_plot_base64()

        overall_html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Relatório Agregado da Simulação EvoHealth</title>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 20px; }}
                .container {{ max-width: 1200px; margin: auto; background: #f4f4f4; padding: 20px; border-radius: 8px; }}
                h1, h2 {{ color: #333; }}
                .plot-container {{ text-align: center; margin-top: 20px; background: #fff; padding: 20px; border-radius: 8px; border: 1px solid #ddd; }}
                table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; font-size: 0.9em; }}
                th {{ background-color: #f2f2f2; }}
                .llm-summary {{ margin-top: 30px; padding: 15px; background: #e9f7ef; border-left: 5px solid #4CAF50; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Relatório Agregado da Simulação EvoHealth</h1>

                <div class="plot-container">
                    <h2>Acurácia do Modelo ML ao Longo das Iterações</h2>
                    <img src="data:image/png;base64,{accuracy_plot_base64}" alt="Acurácia ao longo das iterações">
                </div>

                <h2>Detalhes por Iteração</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Iteração</th>
                            <th>Acurácia</th>
                            <th>Top K-mers</th>
                            <th>Query PubMed</th>
                            <th>Taxa Mutação</th>
                            <th>Taxa Crossover</th>
                            <th>ML Max Depth</th>
                            <th>ML Min Samples Leaf</th>
                            <th>Sumário LLM</th>
                        </tr>
                    </thead>
                    <tbody>
        """
        for report in self.all_simulation_reports:
            iteration = report.get('iteration', 'N/A')
            accuracy = report.get('ml_accuracy', 0.0)
            top_k_mers_list = [k[0] for k in report.get('top_kmers_shap', [])] # Pega apenas a string do k-mer
            top_k_mers = ', '.join(top_k_mers_list) if top_k_mers_list else 'Nenhum' # Formata como string de k-mers
            pubmed_query = report.get('pubmed_query', 'N/A')
            mutation_rate = report.get('mutation_rate', 'N/A')
            crossover_rate = report.get('crossover_rate', 'N/A')
            ml_hyperparameters = report.get('ml_hyperparameters', {})
            max_depth = ml_hyperparameters.get('max_depth', 'N/A')
            min_samples_leaf = ml_hyperparameters.get('min_samples_leaf', 'N/A')
            llm_response = report.get('llm_response', {}) # Já é um dicionário
            llm_summary = llm_response.get('llm_analysis_summary', 'N/A')

            overall_html_content += f"""
                        <tr>
                            <td>{iteration}</td>
                            <td>{accuracy:.3f}</td>
                            <td>{top_k_mers}</td>
                            <td>{pubmed_query}</td>
                            <td>{mutation_rate:.4f}</td>
                            <td>{crossover_rate:.2f}</td>
                            <td>{max_depth}</td>
                            <td>{min_samples_leaf}</td>
                            <td>{llm_summary[:150]}...</td>
                        </tr>
            """
        overall_html_content += """
                    </tbody>
                </table>
                <div class="llm-summary">
                    <h2>Sumário Final da Análise do LLM</h2>
                    <p>""" + self.all_simulation_reports[-1]['llm_response'].get("llm_analysis_summary", "N/A") + """</p>
                </div>
            </div>
        </body>
        </html>
        """
        return overall_html_content


    def _generate_accuracy_plot_base64(self):
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(self.accuracy_history) + 1), self.accuracy_history, marker='o', linestyle='-', color='skyblue')
        plt.title('Acurácia do Modelo ML ao Longo das Iterações')
        plt.xlabel('Iteração')
        plt.ylabel('Acurácia')
        plt.grid(True)
        plt.ylim(0, 1) # Acurácia entre 0 e 1
        plt.xticks(range(1, len(self.accuracy_history) + 1))

        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        return base64.b64encode(buf.getvalue()).decode('utf-8')


    def _call_llm_for_guidance(self, current_state):
        # Esta função simula a chamada a um LLM.
        # Em um sistema real, você faria uma requisição para uma API como a da Gemini.
        # Aqui, vamos simular a lógica de decisão do LLM.

        accuracy = current_state["ml_accuracy"]
        top_kmers_shap = current_state["top_kmers_shap"]
        card_findings = current_state["card_findings"]
        current_query_pubmed = current_state["pubmed_query"]
        current_mutation_rate = current_state["mutation_rate"]
        current_crossover_rate = current_state["crossover_rate"]
        current_max_depth = current_state["ml_hyperparameters"]["max_depth"]
        current_min_samples_leaf = current_state["ml_hyperparameters"]["min_samples_leaf"]

        llm_analysis_summary = ""
        next_query_pubmed = current_query_pubmed
        next_strategy_ml = "default"
        next_mutation_rate = current_mutation_rate
        next_crossover_rate = current_crossover_rate
        next_max_depth = current_max_depth
        next_min_samples_leaf = current_min_samples_leaf
        next_initial_sequence = self._generate_random_sequence(SEQUENCE_LENGTH)

        # Extrai apenas as strings dos k-mers para o LLM
        top_kmer_strings = [k[0] for k in top_kmers_shap]

        # Lógica de decisão do LLM
        if accuracy > 0.85: # Aumentei o limiar para "alta acurácia" para incentivar mais exploração se for razoável
            llm_analysis_summary += f"Modelo de ML com alta acurácia ({accuracy:.2f}). Top k-mers identificados: {top_kmer_strings}. Focando a próxima geração de dados de ML nesses padrões para refinar o entendimento e buscar literatura específica."
            next_strategy_ml = "focused" # Alterado para usar a estratégia focused
            next_query_pubmed = f"{top_kmer_strings[0]} mutation drug resistance influenza" if top_kmer_strings else "influenza drug resistance mutations"
            next_mutation_rate = max(0.0001, current_mutation_rate * 0.75) # Diminui a taxa de mutação
            next_crossover_rate = max(0.40, current_crossover_rate * 0.85) # Diminui um pouco o crossover
            next_max_depth = min(16, current_max_depth + 2) # Aumenta a complexidade do modelo
            next_min_samples_leaf = max(2, current_min_samples_leaf - 1)
            # Seed a próxima sequência com o top k-mer
            if top_kmer_strings:
                idx = random.randint(0, SEQUENCE_LENGTH - K_MER_SIZE)
                next_initial_sequence = next_initial_sequence[:idx] + top_kmer_strings[0] + next_initial_sequence[idx + K_MER_SIZE:]
        elif accuracy < 0.60: # Limiar para "baixa acurácia"
            llm_analysis_summary += f"Acurácia do modelo baixa ({accuracy:.2f}). Sugere que os padrões são difíceis de discernir ou há muito ruído. Aumentarei a taxa de mutação e crossover para explorar mais o espaço genético e simplificarei o modelo de ML para evitar overfitting."
            next_strategy_ml = "default" # Volta para estratégia padrão para explorar mais
            next_query_pubmed = "influenza viral evolution drug resistance broad patterns"
            next_mutation_rate = min(0.01, current_mutation_rate * 1.5) # Aumenta a taxa de mutação
            next_crossover_rate = min(0.90, current_crossover_rate * 1.1) # Aumenta o crossover
            next_max_depth = max(5, current_max_depth - 2) # Simplifica o modelo
            next_min_samples_leaf = min(10, current_min_samples_leaf + 2)
        else: # Acurácia moderada (0.60 a 0.85)
            llm_analysis_summary += f"O modelo tem acurácia razoável ({accuracy:.2f}). Há padrões, mas podem ser sutis ou sobrecarregados por ruído. Continuarei com a estratégia atual de dados de ML, mas com uma busca PubMed mais focada para validar os k-mers. Manterei taxas de evolução padrão. Ajustarei os hiperparâmetros do ML para um equilíbrio entre complexidade e generalização."
            # A estratégia "default" ou "focused" aqui dependerá da iteração anterior.
            # Se a iteração anterior já tentou focar e não melhorou, talvez "default" seja melhor.
            # Por simplicidade, vamos deixar como "default" se não atingir alta acurácia, para re-explorar.
            next_strategy_ml = "default"
            next_query_pubmed = f"{top_kmer_strings[0]} influenza resistance patterns" if top_kmer_strings else "influenza drug resistance patterns"
            next_mutation_rate = current_mutation_rate # Mantém
            next_crossover_rate = current_crossover_rate # Mantém
            next_max_depth = min(12, current_max_depth + 1) # Leve aumento na complexidade, mas limitado
            next_min_samples_leaf = max(3, current_min_samples_leaf - 1) # Leve diminuição para mais flexibilidade

        # Integração de achados do CARD
        if card_findings:
            card_kmer_info = []
            for finding in card_findings:
                card_kmer_info.append(f"K-mer '{finding['kmer']}' associado a {finding['name']} ({finding['ARO']}) via {finding['mechanism']}.")
            llm_analysis_summary += " Achados do CARD: " + " ".join(card_kmer_info)
            llm_analysis_summary += " Usarei essas informações para refinar a query PubMed e a próxima sequência inicial, focando nesses mecanismos de resistência."
            # Ajusta a query PubMed para incluir termos do CARD
            if card_findings and card_findings[0]['name'] not in next_query_pubmed:
                next_query_pubmed += f" {card_findings[0]['name']} {card_findings[0]['mechanism']}"
            # Seeding da sequência com k-mers do CARD
            if card_findings and card_findings[0]['kmer'] not in next_initial_sequence:
                idx = random.randint(0, SEQUENCE_LENGTH - K_MER_SIZE)
                next_initial_sequence = next_initial_sequence[:idx] + card_findings[0]['kmer'] + next_initial_sequence[idx + K_MER_SIZE:]

        llm_response = {
            "llm_analysis_summary": llm_analysis_summary,
            "next_initial_sequence": next_initial_sequence,
            "next_pubmed_query": next_query_pubmed,
            "next_strategy_ml": next_strategy_ml,
            "next_mutation_rate": next_mutation_rate,
            "next_crossover_rate": next_crossover_rate,
            "next_ml_hyperparameters": {
                "max_depth": next_max_depth,
                "min_samples_leaf": next_min_samples_leaf
            },
            "top_k_mers": top_kmer_strings # Adiciona top_k_mers à resposta do LLM para uso em `focused` strategy
        }
        return json.dumps(llm_response)


    def run_autonomous(self, initial_autonomous_prompt, max_iterations=3, ml_sample_size=200):
        print("==================================================")
        print("INICIANDO AGENTE EVOHEALTH AUTÔNOMO")
        print("==================================================")

        # Solicita a chave API do NCBI uma vez
        self.pubmed_api_key = input("Por favor, insira sua chave da API do NCBI (deixe em branco para pular PubMed): ")
        if not self.pubmed_api_key:
            print("Chave da API do NCBI não fornecida. As buscas no PubMed serão ignoradas.")

        # O prompt da chave CARD foi removido, pois usaremos a base simulada.

        # Variáveis iniciais controladas pelo LLM
        current_initial_sequence = self._generate_random_sequence(SEQUENCE_LENGTH) # Começa com uma sequência aleatória
        current_pubmed_query = initial_autonomous_prompt
        current_strategy_ml = "default"
        current_mutation_rate = 0.001
        current_crossover_rate = 0.5
        current_ml_hyperparameters = {"max_depth": 10, "min_samples_leaf": 5}

        for i in range(1, max_iterations + 1):
            self.iteration_count = i
            print(f"\n========== ITERAÇÃO AUTÔNOMA {i}/{max_iterations} ==========\n")
            print(f"--- Iniciando Ciclo de Simulação (Iteração {i}) ---")
            print(f"Sequência Inicial: {current_initial_sequence[:20]}...")
            print(f"Query PubMed: {current_pubmed_query}")
            print(f"Taxa de Mutação: {current_mutation_rate:.4f}, Taxa de Crossover: {current_crossover_rate:.2f}")
            print(f"Hiperparâmetros ML: max_depth={current_ml_hyperparameters['max_depth']}, min_samples_leaf={current_ml_hyperparameters['min_samples_leaf']}")

            # 1. Simulação da Evolução Genética
            print("Simulando evolução do patógeno...")
            evolved_sequence = self._run_genetic_algorithm(current_initial_sequence, current_mutation_rate, current_crossover_rate)

            # 2. Busca de Literatura (PubMed)
            pubmed_articles = self.search_pubmed(current_pubmed_query)

            # 3. Geração de Dados para ML
            print(f"Treinando modelo de ML com {ml_sample_size} amostras (Classe 0: {ml_sample_size//2}, Classe 1: {ml_sample_size//2})...")

            top_kmers_for_focus = []
            if current_strategy_ml == "focused":
                # Certifica-se de que estamos pegando os top_k_mers da *última* resposta do LLM, se houver
                if self.all_simulation_reports:
                     last_llm_guidance = self.all_simulation_reports[-1]['llm_response'] # LLM response já é um dict
                     top_kmers_for_focus = last_llm_guidance.get("top_k_mers", [])

                if not top_kmers_for_focus: # Fallback se não houver top_k_mers no LLM anterior ou na primeira iteração
                    top_kmers_for_focus = ['GGG', 'ATT', 'CGA'] # Default ou do prompt inicial
                print(f"DEBUG ML Data: Usando estratégia 'focused' com k-mers de foco: {top_kmers_for_focus}")
                X_ml, y_ml, kmer_features = self._generate_ml_data_focused(ml_sample_size, K_MER_SIZE, top_kmers_for_focus)
            else: # "default" strategy
                print("DEBUG ML Data: Usando estratégia 'default'.")
                X_ml, y_ml, kmer_features = self._generate_ml_data_default(ml_sample_size, K_MER_SIZE)

            # 4. Treinamento e Interpretação do Modelo ML
            ml_model, accuracy, top_kmers_shap, shap_df, X_test_for_plot = self._train_ml_model(
                X_ml, y_ml, kmer_features,
                current_ml_hyperparameters["max_depth"],
                current_ml_hyperparameters["min_samples_leaf"]
            )
            print(f"Acurácia do modelo: {accuracy:.2f}")
            self.accuracy_history.append(accuracy)

            # Se a acurácia for 0.0 (modelo inválido/plot impossível), top_kmers_shap será []
            if top_kmers_shap:
                self.top_kmers_history.append(top_kmers_shap)
            else:
                self.top_kmers_history.append([]) # Adiciona lista vazia para manter o histórico


            # 5. Busca no CARD (simulado)
            card_findings = self.fetch_from_card(top_kmers_shap)


            # 6. Geração do Plot SHAP (para o relatório)
            shap_plot_base64 = self._generate_shap_plot_base64(ml_model, X_test_for_plot, kmer_features, i)


            # 7. LLM para Análise e Decisão
            current_state_for_llm = {
                "iteration": i,
                "ml_accuracy": accuracy,
                "top_kmers_shap": top_kmers_shap,
                "card_findings": card_findings,
                "pubmed_articles": pubmed_articles, # Passa os artigos para o LLM analisar (opcionalmente)
                "pubmed_query": current_pubmed_query,
                "mutation_rate": current_mutation_rate,
                "crossover_rate": current_crossover_rate,
                "ml_hyperparameters": current_ml_hyperparameters
            }
            llm_response_json_str = self._call_llm_for_guidance(current_state_for_llm)
            # A resposta do LLM já é um JSON string, precisa ser parseada para dicionário no relatório.
            llm_guidance = json.loads(llm_response_json_str)
            print(f"Resposta do LLM para o ciclo atual: LLM Response: {json.dumps(llm_guidance, indent=2)[:200]}...") # Print parcial da resposta do LLM

            # 8. Geração e Armazenamento do Relatório
            print("Gerando relatório...")
            report_data = {
                "timestamp": pd.Timestamp.now().isoformat(),
                "iteration": i,
                "initial_sequence": current_initial_sequence,
                "evolved_sequence": evolved_sequence,
                "mutation_rate": current_mutation_rate,
                "crossover_rate": current_crossover_rate,
                "ml_hyperparameters": current_ml_hyperparameters,
                "ml_accuracy": accuracy,
                "top_kmers_shap": top_kmers_shap,
                "pubmed_query": current_pubmed_query,
                "pubmed_articles": pubmed_articles,
                "card_findings": card_findings,
                "llm_response": llm_guidance, # Armazena o dicionário diretamente
                "shap_plot_base64": shap_plot_base64
            }
            self.all_simulation_reports.append(report_data)

            # Salvar relatório JSON
            json_filename = f'evohealth_report_iter_{i}.json'
            with open(json_filename, 'w') as f:
                json.dump(report_data, f, indent=4)
            print(f"Relatório JSON gerado com sucesso: {os.path.abspath(json_filename)}")

            # Salvar relatório HTML
            html_filename = f'evohealth_report_iter_{i}.html'
            html_content = self._generate_html_report_content(report_data)
            with open(html_filename, 'w') as f:
                f.write(html_content)
            print(f"Relatório HTML gerado com sucesso: {os.path.abspath(html_filename)}")

            print(f"Sumário da Análise do LLM: {llm_guidance.get('llm_analysis_summary', 'N/A')}")

            # Atualiza variáveis para a próxima iteração com base na orientação do LLM
            current_initial_sequence = llm_guidance["next_initial_sequence"]
            current_pubmed_query = llm_guidance["next_pubmed_query"]
            current_strategy_ml = llm_guidance["next_strategy_ml"]
            current_mutation_rate = llm_guidance["next_mutation_rate"]
            current_crossover_rate = llm_guidance["next_crossover_rate"]
            current_ml_hyperparameters = llm_guidance["next_ml_hyperparameters"]

            print(f"Próxima Sequência Inicial: {current_initial_sequence[:20]}...")
            print(f"Próxima Query PubMed: {current_pubmed_query}")
            print(f"Próxima Estratégia de ML: {current_strategy_ml}")
            print(f"Próxima Taxa de Mutação: {current_mutation_rate:.4f}")
            print(f"Próxima Taxa de Crossover: {current_crossover_rate:.2f}")
            print(f"Próximos Hiperparâmetros ML: max_depth={current_ml_hyperparameters['max_depth']}, min_samples_leaf={current_ml_hyperparameters['min_samples_leaf']}")

        # Geração do gráfico de acurácia após todas as iterações
        plt.figure(figsize=(12, 6))
        plt.plot(range(1, len(self.accuracy_history) + 1), self.accuracy_history, marker='o', linestyle='-', color='skyblue')
        plt.title('Acurácia do Modelo ML ao Longo das Iterações')
        plt.xlabel('Iteração')
        plt.ylabel('Acurácia')
        plt.grid(True)
        plt.ylim(0, 1)
        plt.xticks(range(1, len(self.accuracy_history) + 1))

        accuracy_plot_path = 'accuracy_over_iterations.png'
        plt.savefig(accuracy_plot_path)
        plt.close()
        print(f"\nPlot 'accuracy_over_iterations.png' gerado e salvo em: {os.path.abspath(accuracy_plot_path)}")

        # Geração do relatório HTML agregado
        overall_html_report_path = 'evohealth_overall_report.html'
        overall_html_report_content = self._generate_overall_html_report()
        with open(overall_html_report_path, 'w') as f:
            f.write(overall_html_report_content)
        print(f"Relatório HTML agregado 'evohealth_overall_report.html' gerado com sucesso em: {os.path.abspath(overall_html_report_path)}")

        print("\nSimulação autônoma finalizada. O último relatório salvo corresponde à última iteração.")
        print("Detalhes de todos os relatórios podem ser encontrados na variável 'all_simulation_reports'.")

        if self.all_simulation_reports:
            last_report_llm_response = self.all_simulation_reports[-1].get('llm_response', {})
            print(f"Sumário da Análise do LLM na última iteração: {last_report_llm_response.get('llm_analysis_summary', 'N/A')}")
        else:
            print("Nenhum relatório gerado.")

        return self.all_simulation_reports

# --- Execução da Simulação ---
if __name__ == "__main__":
    agent = EvoHealthAgent()

    initial_autonomous_prompt = "Simular a evolução da resistência antiviral em vírus da gripe, com foco em mutações importantes."

    all_simulation_reports = agent.run_autonomous(
        initial_autonomous_prompt,
        max_iterations=3,
        ml_sample_size=200 # Mantendo o número de amostras para ML
    )

    print("\nSimulação autônoma finalizada. O último relatório salvo corresponde à última iteração.")
    print("Detalhes de todos os relatórios podem ser encontrados na variável 'all_simulation_reports'.")

    if all_simulation_reports:
        last_report_llm_response = all_simulation_reports[-1].get('llm_response', {})
        print(f"Sumário da Análise do LLM na última iteração: {last_report_llm_response.get('llm_analysis_summary', 'N/A')}")
    else:
        print("Nenhum relatório gerado.")

