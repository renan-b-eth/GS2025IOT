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
import random

# --- Configurações de Arquivos ---
TRAIN_DATA_FILE = 'DailyDelhiClimateTrain.csv'
TEST_DATA_FILE = 'DailyDelhiClimateTest.csv'


# Verificar se o arquivo de treino existe
if not os.path.exists(TRAIN_DATA_FILE):
    print(f"Erro: O arquivo '{TRAIN_DATA_FILE}' não foi encontrado.")
    print("Por favor, baixe o 'DailyDelhiClimateTrain.csv' do Kaggle e coloque-o no mesmo diretório do script.")
    exit()

try:
    df_train = pd.read_csv(TRAIN_DATA_FILE, parse_dates=['date'])
    print(f"Dataset de Treino '{TRAIN_DATA_FILE}' carregado com sucesso.")
    print("Amostra dos dados de treino:")
    print(df_train.head())

    # Engenharia de Features para o treino
    df_train['day_of_year'] = df_train['date'].dt.dayofyear
    df_train['month'] = df_train['date'].dt.month
    df_train['year'] = df_train['date'].dt.year
    df_train['day_of_week'] = df_train['date'].dt.dayofweek

    # Selecionar features e target para o treino
    # Incluindo 'wind_speed' como nova feature
    features = ['humidity', 'meanpressure', 'wind_speed', 'day_of_year', 'month', 'day_of_week']
    target = 'meantemp'

    # Verificar se as colunas essenciais existem no dataset de treino
    for col in features + [target]:
        if col not in df_train.columns:
            raise ValueError(f"A coluna '{col}' não foi encontrada no dataset de TREINO. Verifique o nome da coluna no CSV.")

    # Remover linhas com valores ausentes no treino
    initial_rows_train = len(df_train)
    df_train.dropna(subset=features + [target], inplace=True)
    if len(df_train) < initial_rows_train:
        print(f"\nForam removidas {initial_rows_train - len(df_train)} linhas com valores ausentes no treino.")

    X_train = df_train[features]
    y_train = df_train[target]

    print(f"\nFeatures selecionadas para TREINO: {features}")
    print(f"Variável alvo para TREINO: {target}")
    print(f"Tamanho do conjunto de treino: {len(X_train)}")

    #Treinamento do Modelo
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    print("\nIniciando o treinamento do modelo RandomForestRegressor...")
    model.fit(X_train, y_train)
    print("Modelo treinado com sucesso!")
    
    # --- Carregar e Pré-processar os Dados de TESTE (para simulação) ---
    if not os.path.exists(TEST_DATA_FILE):
        print(f"Erro: O arquivo '{TEST_DATA_FILE}' não foi encontrado.")
        print("Por favor, baixe o 'DailyDelhiClimateTest.csv' do Kaggle e coloque-o no mesmo diretório do script.")
        exit()

    df_test_sim = pd.read_csv(TEST_DATA_FILE, parse_dates=['date'])
    print(f"\nDataset de Teste (simulação) '{TEST_DATA_FILE}' carregado com sucesso.")
    print("Amostra dos dados de simulação:")
    print(df_test_sim.head())

    # Engenharia de Features para os dados de simulação
    df_test_sim['day_of_year'] = df_test_sim['date'].dt.dayofyear
    df_test_sim['month'] = df_test_sim['date'].dt.month
    df_test_sim['year'] = df_test_sim['date'].dt.year
    df_test_sim['day_of_week'] = df_test_sim['date'].dt.dayofweek

    # Verificar se as colunas essenciais existem no dataset de teste de simulação
    for col in features + [target]: 
        if col not in df_test_sim.columns:
            raise ValueError(f"A coluna '{col}' não foi encontrada no dataset de TESTE (simulação). Verifique o nome da coluna no CSV.")
    
    # Remover linhas com valores ausentes no dataset de simulação
    initial_rows_test_sim = len(df_test_sim)
    df_test_sim.dropna(subset=features + [target], inplace=True)
    if len(df_test_sim) < initial_rows_test_sim:
        print(f"\nForam removidas {initial_rows_test_sim - len(df_test_sim)} linhas com valores ausentes no dataset de simulação.")

    if df_test_sim.empty:
        raise ValueError("O dataset de teste de simulação está vazio após o pré-processamento. Não há dados para simular.")

    # Definir os ranges para simulação dinâmica com base nos dados de teste
    ranges = {}
    for feature in ['humidity', 'meanpressure', 'wind_speed']:
        if feature in df_test_sim.columns:
            ranges[feature] = (df_test_sim[feature].min(), df_test_sim[feature].max())
        else:
           
            print(f"Aviso: Coluna '{feature}' não encontrada em {TEST_DATA_FILE}. Usando range do treino ou genérico.")
            if feature in df_train.columns:
                 ranges[feature] = (df_train[feature].min(), df_train[feature].max())
            else:
                 ranges[feature] = (0, 100)
    
    df_for_initial_plot = df_train


except Exception as e:
    print(f"\nOcorreu um erro durante o carregamento ou pré-processamento dos dados: {e}")
    print("Certifique-se de que os arquivos CSV estão corretos e as colunas existem.")
    exit()


#Simulador com Interface Gráfica

class TemperaturePredictorApp:
    def __init__(self, master, model, features, df_train_original, df_test_sim, ranges):
        self.master = master
        master.title("Simulador de Predição de Temperatura Média Diária")
        master.geometry("900x750")
        master.configure(bg='#e6f0f7')

        self.model = model
        self.features = features
        self.df_train_original = df_train_original.copy() 
        self.df_test_sim = df_test_sim.copy()
        self.ranges = ranges
        
        self.current_sim_idx = 0 
        self.max_sim_idx = len(self.df_test_sim) - 1

        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure('TFrame', background='#e6f0f7')
        self.style.configure('TLabel', background='#e6f0f7', font=('Helvetica', 10))
        self.style.configure('TButton', font=('Helvetica', 10, 'bold'))
        self.style.configure('TEntry', font=('Helvetica', 10))
        self.style.configure('TCombobox', font=('Helvetica', 10))
        self.style.configure('TLabelFrame', background='#e6f0f7', foreground='#333', font=('Helvetica', 11, 'bold'))

        self.input_frame = ttk.LabelFrame(master, text="Entrada de Dados do Sensor (Dinâmico)", padding="15")
        self.input_frame.pack(pady=15, padx=25, fill="x")

        self.create_input_widgets(self.input_frame)

        self.predict_button = ttk.Button(master, text="Prever Temperatura (Manual)", command=self.predict_temperature_manual)
        self.predict_button.pack(pady=10)

        self.auto_update_active = tk.BooleanVar(value=False)
        self.auto_update_button = ttk.Checkbutton(master, text="Ativar Simulação Dinâmica (a cada 10s)",
                                                  variable=self.auto_update_active,
                                                  command=self.toggle_auto_update)
        self.auto_update_button.pack(pady=5)

        self.result_frame = ttk.LabelFrame(master, text="Resultado da Predição", padding="15")
        self.result_frame.pack(pady=10, padx=25, fill="x")

        self.result_label = ttk.Label(self.result_frame, text="Temperatura Média Predita: -- °C", font=('Helvetica', 15, 'bold'), foreground='#0055aa')
        self.result_label.pack(pady=10)
        
        self.real_temp_label = ttk.Label(self.result_frame, text="Temperatura Real (Simulada): -- °C", font=('Helvetica', 12), foreground='green')
        self.real_temp_label.pack(pady=5)


        self.chart_frame = ttk.LabelFrame(master, text="Visualização da Predição", padding="15")
        self.chart_frame.pack(pady=10, padx=25, fill="both", expand=True)

        self.fig, self.ax = plt.subplots(figsize=(8, 4.5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.chart_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.plot_initial_data()
        self.job_id = None
        
        self.set_initial_simulated_values()

    def create_input_widgets(self, parent_frame):
        feature_labels = {
            'humidity': 'Umidade (%)',
            'meanpressure': 'Pressão (hPa)',
            'wind_speed': 'Velocidade do Vento (km/h)',
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

        self.get_simulated_values_button = ttk.Button(parent_frame, text="Obter Próximo Dado de Sensor", command=self.get_next_simulated_values)
        self.get_simulated_values_button.grid(row=len(self.features), column=0, columnspan=2, pady=10)

    def set_initial_simulated_values(self):
       
        initial_data = self.df_test_sim.sample(1).iloc[0]
        self._update_input_fields_from_series(initial_data)

    def get_next_simulated_values(self):
        
        self.current_sim_idx = (self.current_sim_idx + 1) % len(self.df_test_sim)
        simulated_data = self.df_test_sim.iloc[self.current_sim_idx]
        self._update_input_fields_from_series(simulated_data)
        self.predict_temperature_manual() 

    def _update_input_fields_from_series(self, data_series):
        
        for feature in self.features:
            if feature in data_series:
                self.entry_widgets[feature].delete(0, tk.END)
                self.entry_widgets[feature].insert(0, f"{data_series[feature]:.1f}")
        
        # Atualiza a temperatura real simulada
        if 'meantemp' in data_series:
            self.real_temp_label.config(text=f"Temperatura Real (Simulada): {data_series['meantemp']:.2f} °C")

    def predict_temperature_manual(self):
        self._perform_prediction(manual_trigger=True)

    def predict_temperature_auto(self):
       
        self.get_next_simulated_values()
        self._perform_prediction(manual_trigger=False)

    def _perform_prediction(self, manual_trigger):
        input_values = {}
        try:
            for feature in self.features:
                value_str = self.entry_widgets[feature].get()
                if not value_str:
                    messagebox.showerror("Erro de Entrada", f"O campo '{feature}' não pode estar vazio.")
                    return
                value = float(value_str)
                input_values[feature] = value
            
            # Validações básicas
            if not (self.ranges['humidity'][0] <= input_values.get('humidity', 50) <= self.ranges['humidity'][1]):
                if manual_trigger: messagebox.showerror("Erro de Entrada", f"Umidade deve estar entre {self.ranges['humidity'][0]:.1f} e {self.ranges['humidity'][1]:.1f}.")
                return
            if not (self.ranges['meanpressure'][0] <= input_values.get('meanpressure', 1000) <= self.ranges['meanpressure'][1]):
                if manual_trigger: messagebox.showerror("Erro de Entrada", f"Pressão deve estar entre {self.ranges['meanpressure'][0]:.1f} e {self.ranges['meanpressure'][1]:.1f}.")
                return
            if not (self.ranges['wind_speed'][0] <= input_values.get('wind_speed', 5) <= self.ranges['wind_speed'][1]):
                if manual_trigger: messagebox.showerror("Erro de Entrada", f"Velocidade do Vento deve estar entre {self.ranges['wind_speed'][0]:.1f} e {self.ranges['wind_speed'][1]:.1f}.")
                return
            if not (1 <= input_values.get('day_of_year', 1) <= 366):
                if manual_trigger: messagebox.showerror("Erro de Entrada", "Dia do Ano deve estar entre 1 e 366.")
                return
            if not (1 <= input_values.get('month', 1) <= 12):
                if manual_trigger: messagebox.showerror("Erro de Entrada", "Mês deve estar entre 1 e 12.")
                return
            if not (0 <= input_values.get('day_of_week', 0) <= 6):
                if manual_trigger: messagebox.showerror("Erro de Entrada", "Dia da Semana deve estar entre 0 (Seg) e 6 (Dom).")
                return

            input_data = pd.DataFrame([input_values], columns=self.features)
            predicted_temp = self.model.predict(input_data)[0]

            self.result_label.config(text=f"Temperatura Média Predita: {predicted_temp:.2f} °C")

            # Obter a temperatura real para comparação
            real_temp = self.df_test_sim.iloc[self.current_sim_idx]['meantemp']
            self.real_temp_label.config(text=f"Temperatura Real (Simulada): {real_temp:.2f} °C")

            try:
               
                simulated_date = self.df_test_sim.iloc[self.current_sim_idx]['date']
            except Exception:
                 simulated_date = datetime.datetime.now()

            self.update_plot(simulated_date, predicted_temp, real_temp)

        except ValueError:
            if manual_trigger: messagebox.showerror("Erro de Entrada", "Por favor, insira valores numéricos válidos para todas as entradas.")
        except Exception as e:
            if manual_trigger: messagebox.showerror("Erro", f"Ocorreu um erro na predição: {e}")
            else: print(f"Erro na atualização automática: {e}")

    def plot_initial_data(self):
        self.ax.clear()
        self.ax.plot(self.df_train_original['date'], self.df_train_original['meantemp'], label='Histórico de Temperatura Média (Treino)', color='skyblue')
        self.ax.set_xlabel("Data")
        self.ax.set_ylabel("Temperatura Média (°C)")
        self.ax.set_title("Histórico de Temperatura Média Diária")
        self.ax.grid(True, linestyle='--', alpha=0.7)
        self.ax.legend()
        self.fig.autofmt_xdate()
        self.canvas.draw()

    def update_plot(self, prediction_date, predicted_temp, real_temp):
        self.ax.clear()
        self.ax.plot(self.df_train_original['date'], self.df_train_original['meantemp'], label='Histórico de Temperatura Média (Treino)', color='skyblue')
        
        
        self.ax.plot(prediction_date, predicted_temp, 'ro', markersize=10, label=f'Predição: {predicted_temp:.2f}°C')
       
        self.ax.plot(prediction_date, real_temp, 'go', markersize=10, label=f'Real: {real_temp:.2f}°C')
        
        self.ax.set_xlabel("Data")
        self.ax.set_ylabel("Temperatura Média (°C)")
        self.ax.set_title("Histórico de Temperatura Média Diária com Predição Dinâmica")
        self.ax.grid(True, linestyle='--', alpha=0.7)
        self.ax.legend()
        self.fig.autofmt_xdate()
        self.canvas.draw()
    
    def toggle_auto_update(self):
        if self.auto_update_active.get():
            print("Ativando atualização automática...")
            self.schedule_update()
        else:
            if self.job_id:
                self.master.after_cancel(self.job_id)
                self.job_id = None
                print("Atualização automática desativada.")

    def schedule_update(self):
        self.predict_temperature_auto()
        self.job_id = self.master.after(1000, self.schedule_update)


if __name__ == "__main__":
    root = tk.Tk()
    app = TemperaturePredictorApp(root, model, features, df_train, df_test_sim, ranges)
    root.mainloop()