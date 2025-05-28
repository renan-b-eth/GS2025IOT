import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, messagebox
import datetime
import os

# --- PARTE 1: Carregamento e Pré-processamento dos Dados ---

# Nome do arquivo do Kaggle (ajuste se o seu for diferente)
DATA_FILE = 'DailyDelhiClimateTrain.csv'

# Verificar se o arquivo existe
if not os.path.exists(DATA_FILE):
    print(f"Erro: O arquivo '{DATA_FILE}' não foi encontrado.")
    print("Por favor, baixe o 'DailyDelhiClimateTrain.csv' do Kaggle e coloque-o no mesmo diretório do script.")
    exit()

try:
    # Carregar o dataset real do Kaggle
    # Foco na temperatura média diária (meantemp)
    df = pd.read_csv(DATA_FILE, parse_dates=['date'])
    print(f"Dataset '{DATA_FILE}' carregado com sucesso.")
    print("Amostra dos dados:")
    print(df.head())
    print("\nInformações sobre as colunas:")
    df.info()

    # Engenharia de Features: Extrair informações temporais úteis
    df['day_of_year'] = df['date'].dt.dayofyear
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['day_of_week'] = df['date'].dt.dayofweek # 0=Monday, 6=Sunday

    # Selecionar as features e o target
    # Usaremos umidade (humidity), pressão (pressure), dia do ano, mês e dia da semana.
    # Outras features como 'meantemp', 'mintemp', 'maxtemp' são o que queremos predizer ou são correlacionadas.
    # Vamos predizer 'meantemp' (temperatura média diária)
    features = ['humidity', 'meanpressure', 'day_of_year', 'month', 'day_of_week']
    target = 'meantemp'

    # Verificar se as colunas essenciais existem
    for col in features + [target]:
        if col not in df.columns:
            raise ValueError(f"A coluna '{col}' não foi encontrada no dataset. Verifique o nome da coluna no CSV.")

    # Remover linhas com valores ausentes (se houver)
    initial_rows = len(df)
    df.dropna(subset=features + [target], inplace=True)
    if len(df) < initial_rows:
        print(f"\nForam removidas {initial_rows - len(df)} linhas com valores ausentes.")

    X = df[features]
    y = df[target]

    print(f"\nFeatures selecionadas: {features}")
    print(f"Variável alvo: {target}")

    # Dividir os dados em conjuntos de treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"\nTamanho do conjunto de treino: {len(X_train)}")
    print(f"Tamanho do conjunto de teste: {len(X_test)}")

    # --- PARTE 2: Treinamento do Modelo ---

    # Escolher e treinar o modelo de Regressão
    # RandomForestRegressor é robusto e geralmente tem bom desempenho para dados tabulares
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1) # n_jobs=-1 usa todos os núcleos disponíveis
    print("\nIniciando o treinamento do modelo RandomForestRegressor...")
    model.fit(X_train, y_train)
    print("Modelo treinado com sucesso!")

    # Avaliar o modelo
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\n--- Avaliação do Modelo ---")
    print(f"Erro Quadrático Médio (MSE): {mse:.2f}")
    print(f"Coeficiente de Determinação (R²): {r2:.2f}") # Quanto mais próximo de 1, melhor
    print(f"Raiz do Erro Quadrático Médio (RMSE): {np.sqrt(mse):.2f}")


    # Visualizar algumas predições vs. valores reais
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.3, label='Predições vs. Reais')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Linha de Identidade')
    plt.xlabel("Temperatura Média Real (°C)")
    plt.ylabel("Temperatura Média Predita (°C)")
    plt.title("Temperatura Média Real vs. Predita (Conjunto de Teste)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

except Exception as e:
    print(f"\nOcorreu um erro durante o carregamento ou pré-processamento dos dados: {e}")
    print("Certifique-se de que o arquivo CSV está correto e as colunas existem.")
    exit()


# --- PARTE 3: Simulador com Interface Gráfica (Tkinter) ---

class TemperaturePredictorApp:
    def __init__(self, master, model, features, df_original):
        self.master = master
        master.title("Simulador de Predição de Temperatura Média Diária")
        master.geometry("900x750") # Ajustar o tamanho da janela
        master.configure(bg='#e6f0f7') # Cor de fundo mais clara

        self.model = model # O modelo treinado
        self.features = features # As features usadas no treinamento
        self.df_original = df_original.copy() # Cópia do DataFrame original para gráficos

        # Estilos
        self.style = ttk.Style()
        self.style.theme_use('clam') # Tema moderno
        self.style.configure('TFrame', background='#e6f0f7')
        self.style.configure('TLabel', background='#e6f0f7', font=('Helvetica', 10))
        self.style.configure('TButton', font=('Helvetica', 10, 'bold'))
        self.style.configure('TEntry', font=('Helvetica', 10))
        self.style.configure('TCombobox', font=('Helvetica', 10))
        self.style.configure('TLabelFrame', background='#e6f0f7', foreground='#333', font=('Helvetica', 11, 'bold'))

        # Frame de Entrada
        self.input_frame = ttk.LabelFrame(master, text="Entrada de Dados para Predição", padding="15")
        self.input_frame.pack(pady=15, padx=25, fill="x")

        # Rótulos e Campos de Entrada
        self.create_input_widgets(self.input_frame)

        # Botão de Predição
        self.predict_button = ttk.Button(master, text="Prever Temperatura", command=self.predict_temperature)
        self.predict_button.pack(pady=10)

        # Frame de Resultado
        self.result_frame = ttk.LabelFrame(master, text="Resultado da Predição", padding="15")
        self.result_frame.pack(pady=10, padx=25, fill="x")

        self.result_label = ttk.Label(self.result_frame, text="Temperatura Média Predita: -- °C", font=('Helvetica', 15, 'bold'), foreground='#0055aa')
        self.result_label.pack(pady=10)

        # Frame para Gráficos
        self.chart_frame = ttk.LabelFrame(master, text="Visualização da Predição", padding="15")
        self.chart_frame.pack(pady=10, padx=25, fill="both", expand=True)

        self.fig, self.ax = plt.subplots(figsize=(8, 4.5)) # Ajustar tamanho do gráfico
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.chart_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.plot_initial_data()

    def create_input_widgets(self, parent_frame):
        # Mapeamento dos nomes de features para rótulos na interface
        feature_labels = {
            'humidity': 'Umidade (%)',
            'meanpressure': 'Pressão (hPa)',
            'day_of_year': 'Dia do Ano (1-366)',
            'month': 'Mês (1-12)',
            'day_of_week': 'Dia da Semana (0=Seg, 6=Dom)'
        }
        
        self.entry_widgets = {}
        for i, feature in enumerate(self.features):
            ttk.Label(parent_frame, text=f"{feature_labels.get(feature, feature.capitalize())}:").grid(row=i, column=0, padx=8, pady=5, sticky="w")
            entry = ttk.Entry(parent_frame, width=15)
            entry.grid(row=i, column=1, padx=8, pady=5, sticky="w")
            self.entry_widgets[feature] = entry

            # Preencher com valores médios/atuais como sugestão
            if feature in ['humidity', 'meanpressure']:
                mean_val = self.df_original[feature].mean()
                entry.insert(0, f"{mean_val:.1f}")
            elif feature == 'day_of_year':
                entry.insert(0, str(datetime.datetime.now().timetuple().tm_yday))
            elif feature == 'month':
                entry.insert(0, str(datetime.datetime.now().month))
            elif feature == 'day_of_week':
                entry.insert(0, str(datetime.datetime.now().weekday())) # Python's weekday: 0=Monday, 6=Sunday

        # Botão para usar data e hora atual
        self.use_current_time_button = ttk.Button(parent_frame, text="Usar Data Atual", command=self.set_current_time)
        self.use_current_time_button.grid(row=len(self.features), column=0, columnspan=2, pady=10)

    def set_current_time(self):
        now = datetime.datetime.now()
        if 'day_of_year' in self.entry_widgets:
            self.entry_widgets['day_of_year'].delete(0, tk.END)
            self.entry_widgets['day_of_year'].insert(0, str(now.timetuple().tm_yday))
        if 'month' in self.entry_widgets:
            self.entry_widgets['month'].delete(0, tk.END)
            self.entry_widgets['month'].insert(0, str(now.month))
        if 'day_of_week' in self.entry_widgets:
            self.entry_widgets['day_of_week'].delete(0, tk.END)
            self.entry_widgets['day_of_week'].insert(0, str(now.weekday()))

    def predict_temperature(self):
        input_values = {}
        try:
            for feature in self.features:
                value = float(self.entry_widgets[feature].get())
                input_values[feature] = value
            
            # Validações básicas (ajuste conforme a necessidade do seu dataset)
            if not (0 <= input_values.get('humidity', 50) <= 100):
                messagebox.showerror("Erro de Entrada", "Umidade deve estar entre 0 e 100.")
                return
            if not (900 <= input_values.get('meanpressure', 1000) <= 1100): # Valores típicos de pressão
                messagebox.showerror("Erro de Entrada", "Pressão parece fora de um range realista (900-1100 hPa).")
                return
            if not (1 <= input_values.get('day_of_year', 1) <= 366):
                messagebox.showerror("Erro de Entrada", "Dia do Ano deve estar entre 1 e 366.")
                return
            if not (1 <= input_values.get('month', 1) <= 12):
                messagebox.showerror("Erro de Entrada", "Mês deve estar entre 1 e 12.")
                return
            if not (0 <= input_values.get('day_of_week', 0) <= 6):
                messagebox.showerror("Erro de Entrada", "Dia da Semana deve estar entre 0 (Seg) e 6 (Dom).")
                return

            # Criar um DataFrame com os dados de entrada
            # A ordem das colunas no input_data deve ser a mesma das features usadas no treinamento
            input_data = pd.DataFrame([input_values], columns=self.features)

            # Fazer a predição
            predicted_temp = self.model.predict(input_data)[0]

            self.result_label.config(text=f"Temperatura Média Predita: {predicted_temp:.2f} °C")

            # Atualizar o gráfico
            # Para o gráfico, precisamos de uma data para o ponto predito.
            # Vamos usar uma data "representativa" baseada no dia do ano e mês.
            # Para simplificar, usaremos o ano mais recente do dataset para a visualização
            # ou um ano fixo se não tivermos histórico.
            
            # Gerar uma data para o ponto predito (arbitrariamente usando o ano do último registro no df)
            try:
                # Se o DataFrame original tem datas, use o último ano para manter a consistência visual
                if 'date' in self.df_original.columns:
                    current_year = self.df_original['date'].dt.year.max()
                else:
                    current_year = datetime.datetime.now().year # Fallback
                
                # Cuidado: a data pode ser inválida para 29 de fevereiro em anos não bissextos, mas para visualização é aceitável.
                simulated_date = datetime.datetime(current_year, int(input_values['month']), 1) + \
                                 datetime.timedelta(days=int(input_values['day_of_year']) - 1)
            except ValueError:
                 simulated_date = datetime.datetime.now() # Fallback se a data gerada for inválida

            self.update_plot(simulated_date, predicted_temp)

        except ValueError:
            messagebox.showerror("Erro de Entrada", "Por favor, insira valores numéricos válidos para todas as entradas.")
        except Exception as e:
            messagebox.showerror("Erro", f"Ocorreu um erro na predição: {e}")

    def plot_initial_data(self):
        self.ax.clear()
        # Plotar a temperatura média ao longo do tempo
        self.ax.plot(self.df_original['date'], self.df_original['meantemp'], label='Histórico de Temperatura Média', color='skyblue')
        self.ax.set_xlabel("Data")
        self.ax.set_ylabel("Temperatura Média (°C)")
        self.ax.set_title("Histórico de Temperatura Média Diária")
        self.ax.grid(True, linestyle='--', alpha=0.7)
        self.ax.legend()
        self.fig.autofmt_xdate() # Formatar as datas no eixo X
        self.canvas.draw()

    def update_plot(self, prediction_date, predicted_temp):
        self.ax.clear()
        # Plotar os dados históricos
        self.ax.plot(self.df_original['date'], self.df_original['meantemp'], label='Histórico de Temperatura Média', color='skyblue')
        
        # Plotar o ponto predito
        self.ax.plot(prediction_date, predicted_temp, 'ro', markersize=10, label=f'Predição: {predicted_temp:.2f}°C')
        
        self.ax.set_xlabel("Data")
        self.ax.set_ylabel("Temperatura Média (°C)")
        self.ax.set_title("Histórico de Temperatura Média Diária com Predição")
        self.ax.grid(True, linestyle='--', alpha=0.7)
        self.ax.legend()
        self.fig.autofmt_xdate()
        self.canvas.draw()


if __name__ == "__main__":
    # Garanta que 'df' e 'model' estão definidos globalmente ou passados corretamente
    # Eles são definidos nas primeiras partes do script.
    root = tk.Tk()
    app = TemperaturePredictorApp(root, model, features, df)
    root.mainloop()