import tkinter as tk
from tkinter import ttk, messagebox
import random
import math
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import datetime
import os

# --- Configurações Globais para Medidores de Sensores ---
UPDATE_INTERVAL = 2000  # Intervalo de atualização em milissegundos
TEMP_MIN, TEMP_MAX = 10.0, 40.0  # Temperatura em °C (ajustado para incluir valores do dataset)
OCC_MIN, OCC_MAX = 0, 100      # Ocupação em %
HUM_MIN_SENSOR, HUM_MAX_SENSOR = 20, 100 # Umidade em % (para o sensor, diferente do input da previsão)

TEMP_ALERT_THRESHOLD = 35.0
OCC_ALERT_THRESHOLD = 80
HUM_ALERT_THRESHOLD_SENSOR = 85 # Limite para umidade alta no sensor

COLOR_NORMAL_GAUGE = "#4CAF50"
COLOR_ALERT_GAUGE = "#F44336"
COLOR_GAUGE_BACKGROUND = "#E0E0E0"
COLOR_TEXT_ON_GAUGE = "#333333"
COLOR_WINDOW_BG = "#ECEFF1"
COLOR_FRAME_BG = "#FFFFFF" 
COLOR_TAB_BG = "#FAFAFA"   
COLOR_LABEL_TEXT = "#263238"
COLOR_TREEVIEW_HEADING = "#E0E0E0"
FONT_MAIN_TITLE = ("Helvetica", 18, "bold")
FONT_GAUGE_TITLE = ("Helvetica", 12)
FONT_GAUGE_VALUE = ("Helvetica", 20, "bold")
FONT_GAUGE_MINMAX = ("Helvetica", 9)
FONT_TAB_HEADER = ("Helvetica", 11, "bold")
FONT_LABEL_NORMAL = ('Helvetica', 10)
FONT_BUTTON_NORMAL = ('Helvetica', 10, 'bold')
FONT_INPUT_NORMAL = ('Helvetica', 10)

SHELTERS_DATA = [
    {"id": "A001", "name": "Abrigo Central Alfa", "capacity_total": 100, "capacity_current": random.randint(10, 50),
     "resources": ["água", "comida", "primeiros_socorros"], "safety_level": "alta"},
    {"id": "B002", "name": "Escola Segura Beta", "capacity_total": 150, "capacity_current": random.randint(100, 145),
     "resources": ["água", "comida"], "safety_level": "média"},
    {"id": "C003", "name": "Ginásio Comunitário Gama", "capacity_total": 200, "capacity_current": random.randint(30, 80),
     "resources": ["água", "comida", "primeiros_socorros", "energia"], "safety_level": "alta"},
    {"id": "D004", "name": "Posto Avançado Delta", "capacity_total": 50, "capacity_current": random.randint(5, 20),
     "resources": ["água", "primeiros_socorros"], "safety_level": "média"},
    {"id": "E005", "name": "Centro Comunitário Épsilon", "capacity_total": 80, "capacity_current": random.randint(60, 75),
     "resources": ["comida"], "safety_level": "baixa"},
]
POSSIBLE_RESOURCES = ["água", "comida", "primeiros_socorros", "energia", "abrigo_animais"]
SAFETY_LEVEL_MAP = {"alta": 3, "média": 2, "baixa": 1}
SAFETY_LEVEL_MAP_INV = {v: k for k, v in SAFETY_LEVEL_MAP.items()}


# --- PARTE 1: Carregamento e Pré-processamento dos Dados para Previsão ---
DATA_FILE = 'DailyDelhiClimateTrain.csv'
model_predictor = None
features_predictor = []
df_predictor_original = None

def load_and_train_prediction_model():
    """Carrega os dados, pré-processa e treina o modelo de previsão de temperatura."""
    global model_predictor, features_predictor, df_predictor_original

    if not os.path.exists(DATA_FILE):
        messagebox.showerror("Erro de Arquivo",
                             f"O ficheiro '{DATA_FILE}' não foi encontrado.\n"
                             "Por favor, descarregue-o do Kaggle (Daily Delhi Climate) e coloque-o "
                             "no mesmo diretório do script.\n\nA aplicação não pode iniciar sem este ficheiro.")
        return False
    try:
        df = pd.read_csv(DATA_FILE, parse_dates=['date'])
        df['day_of_year'] = df['date'].dt.dayofyear
        df['month'] = df['date'].dt.month
        df['year'] = df['date'].dt.year
        df['day_of_week'] = df['date'].dt.dayofweek

        required_cols = ['humidity', 'meanpressure', 'meantemp', 'date']
        for col in required_cols:
            if col not in df.columns:
                if col == 'meanpressure' and 'pressure' in df.columns:
                    df.rename(columns={'pressure': 'meanpressure'}, inplace=True)
                elif col == 'meantemp' and 'temperature' in df.columns:
                     df.rename(columns={'temperature': 'meantemp'}, inplace=True)
                else:
                    messagebox.showerror("Erro de Coluna",
                                         f"A coluna '{col}' (ou uma alternativa válida) não foi encontrada no dataset '{DATA_FILE}'.\n"
                                         "Verifique o ficheiro CSV.")
                    return False

        features_predictor = ['humidity', 'meanpressure', 'day_of_year', 'month', 'day_of_week']
        target_predictor = 'meantemp'

        for col in features_predictor + [target_predictor]:
            if col not in df.columns:
                 messagebox.showerror("Erro de Coluna", f"Coluna '{col}' ainda ausente após verificações.")
                 return False

        df_predictor_original = df.copy()
        df.dropna(subset=features_predictor + [target_predictor], inplace=True)

        if df.empty:
            messagebox.showerror("Erro de Dados", "Não há dados suficientes após remover valores ausentes para treinar o modelo.")
            return False

        X = df[features_predictor]
        y = df[target_predictor]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model_predictor = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model_predictor.fit(X_train, y_train)

        y_pred_test = model_predictor.predict(X_test)
        mse = mean_squared_error(y_test, y_pred_test)
        r2 = r2_score(y_test, y_pred_test)
        print(f"--- Avaliação do Modelo de Previsão (na consola) ---")
        print(f"MSE: {mse:.2f}, R²: {r2:.2f}, RMSE: {np.sqrt(mse):.2f}")
        return True
    except Exception as e:
        messagebox.showerror("Erro no Processamento de Dados",
                             f"Ocorreu um erro durante o carregamento ou treino do modelo de previsão:\n{e}")
        return False

# --- Geração de Dados Aleatórios para Sensores (Medidores) ---
def get_random_temperature_sensor():
    return round(random.uniform(TEMP_MIN, TEMP_MAX), 1)
def get_random_occupancy_sensor():
    return random.randint(OCC_MIN, OCC_MAX)
def get_random_humidity_sensor():
    return random.randint(HUM_MIN_SENSOR, HUM_MAX_SENSOR)

# --- Widget de Medidor (Gauge) ---
class Gauge(ttk.Frame):
    def __init__(self, parent, title, unit, min_val, max_val, alert_threshold, is_float=False):
        super().__init__(parent, padding="10 10 10 10")
        self.configure(style="Gauge.TFrame")
        self.title_text, self.unit, self.min_val, self.max_val = title, unit, min_val, max_val
        self.alert_threshold, self.current_value, self.is_float = alert_threshold, min_val, is_float
        self.canvas_width, self.gauge_thickness = 150, 16
        self.canvas_height = (self.canvas_width / 2) + self.gauge_thickness + 15
        self.title_label = ttk.Label(self, text=self.title_text, font=FONT_GAUGE_TITLE, style="GaugeTitle.TLabel")
        self.title_label.pack(pady=(0, 6))
        self.canvas = tk.Canvas(self, width=self.canvas_width, height=self.canvas_height, bg=COLOR_FRAME_BG, highlightthickness=0)
        self.canvas.pack()
        value_text_formatted = f"{self.current_value:.1f}{self.unit}" if self.is_float else f"{self.current_value}{self.unit}"
        self.value_label = ttk.Label(self, text=value_text_formatted, font=FONT_GAUGE_VALUE, style="GaugeValue.TLabel")
        self.value_label.pack(pady=(6,0))
        self._draw_gauge()

    def _draw_gauge(self):
        self.canvas.delete("all")
        arc_bbox_x1, arc_bbox_y1 = self.gauge_thickness, self.gauge_thickness
        arc_bbox_x2, arc_bbox_y2 = self.canvas_width - self.gauge_thickness, self.canvas_width - self.gauge_thickness
        self.canvas.create_arc(arc_bbox_x1, arc_bbox_y1, arc_bbox_x2, arc_bbox_y2, start=0, extent=180, outline=COLOR_GAUGE_BACKGROUND, width=self.gauge_thickness, style=tk.ARC)
        value_range = self.max_val - self.min_val
        percentage = (self.current_value - self.min_val) / value_range if value_range != 0 else 0.0
        percentage = max(0.0, min(1.0, percentage))
        value_sweep_angle = 180 * percentage
        current_gauge_color = COLOR_ALERT_GAUGE if self.current_value > self.alert_threshold else COLOR_NORMAL_GAUGE
        if value_sweep_angle > 0:
            self.canvas.create_arc(arc_bbox_x1, arc_bbox_y1, arc_bbox_x2, arc_bbox_y2, start=180, extent=-value_sweep_angle, outline=current_gauge_color, width=self.gauge_thickness, style=tk.ARC)
        arc_center_y = arc_bbox_y1 + (arc_bbox_y2 - arc_bbox_y1) / 2
        text_offset_y = 10
        self.canvas.create_text(arc_bbox_x1 - 5, arc_center_y + text_offset_y, text=str(self.min_val), anchor="ne", font=FONT_GAUGE_MINMAX, fill=COLOR_TEXT_ON_GAUGE)
        self.canvas.create_text(arc_bbox_x2 + 5, arc_center_y + text_offset_y, text=str(self.max_val), anchor="nw", font=FONT_GAUGE_MINMAX, fill=COLOR_TEXT_ON_GAUGE)

    def update_value(self, new_value):
        self.current_value = new_value
        value_text_formatted = f"{self.current_value:.1f}{self.unit}" if self.is_float else f"{self.current_value}{self.unit}"
        self.value_label.config(text=value_text_formatted)
        self._draw_gauge()

# --- Interface de Previsão de Temperatura ---
class TemperaturePredictorInterface:
    def __init__(self, master_frame, model, features_list, df_hist_data):
        self.master = master_frame
        self.master.configure(style="Tab.TFrame")
        self.model, self.features = model, features_list
        self.df_original = df_hist_data.copy() if df_hist_data is not None else pd.DataFrame()
        self.input_frame = ttk.LabelFrame(self.master, text="Entrada de Dados para Predição", padding="15", style="TLabelframe")
        self.input_frame.pack(pady=10, padx=20, fill="x")
        self.create_input_widgets(self.input_frame)
        self.predict_button = ttk.Button(self.master, text="Prever Temperatura Média", command=self.predict_temperature, style="Accent.TButton")
        self.predict_button.pack(pady=8)
        self.result_frame = ttk.LabelFrame(self.master, text="Resultado da Predição", padding="15", style="TLabelframe")
        self.result_frame.pack(pady=8, padx=20, fill="x")
        self.result_label = ttk.Label(self.result_frame, text="Temperatura Média Predita: -- °C", font=('Helvetica', 14, 'bold'), foreground='#00695C', style="Result.TLabel")
        self.result_label.pack(pady=8)
        self.chart_frame = ttk.LabelFrame(self.master, text="Visualização da Predição", padding="15", style="TLabelframe")
        self.chart_frame.pack(pady=10, padx=20, fill="both", expand=True)
        self.fig, self.ax = plt.subplots(figsize=(7, 4))
        try: plt.style.use('seaborn-v0_8-whitegrid')
        except: pass # Usa o estilo default se seaborn não estiver disponível
        self.canvas_plot = FigureCanvasTkAgg(self.fig, master=self.chart_frame)
        self.canvas_widget = self.canvas_plot.get_tk_widget()
        self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        if not self.df_original.empty: self.plot_initial_data()
        else:
            self.ax.text(0.5, 0.5, "Dados históricos não disponíveis.", ha='center', va='center', transform=self.ax.transAxes)
            self.canvas_plot.draw()

    def create_input_widgets(self, parent_frame):
        feature_labels = {'humidity': 'Umidade (%)', 'meanpressure': 'Pressão (hPa)', 'day_of_year': 'Dia do Ano (1-366)', 'month': 'Mês (1-12)', 'day_of_week': 'Dia da Semana (0=Seg, 6=Dom)'}
        self.entry_widgets = {}
        num_features, cols = len(self.features), 2
        rows_per_col = (num_features + cols -1) // cols
        for i, feature in enumerate(self.features):
            row_idx, col_idx = i % rows_per_col, (i // rows_per_col) * 2
            ttk.Label(parent_frame, text=f"{feature_labels.get(feature, feature.capitalize())}:").grid(row=row_idx, column=col_idx, padx=5, pady=5, sticky="w")
            entry = ttk.Entry(parent_frame, width=12, font=FONT_INPUT_NORMAL)
            entry.grid(row=row_idx, column=col_idx + 1, padx=5, pady=5, sticky="ew")
            self.entry_widgets[feature] = entry
            if not self.df_original.empty and feature in ['humidity', 'meanpressure'] and feature in self.df_original.columns:
                entry.insert(0, f"{self.df_original[feature].mean():.1f}")
            if feature == 'day_of_year': entry.insert(0, str(datetime.datetime.now().timetuple().tm_yday))
            elif feature == 'month': entry.insert(0, str(datetime.datetime.now().month))
            elif feature == 'day_of_week': entry.insert(0, str(datetime.datetime.now().weekday()))
        parent_frame.columnconfigure(1, weight=1)
        if num_features > rows_per_col: parent_frame.columnconfigure(3, weight=1)
        self.use_current_time_button = ttk.Button(parent_frame, text="Usar Data Atual", command=self.set_current_time)
        self.use_current_time_button.grid(row=rows_per_col, column=0, columnspan=max(2, cols*2), pady=10)

    def set_current_time(self):
        now = datetime.datetime.now()
        for feature, entry_widget in self.entry_widgets.items():
            entry_widget.delete(0, tk.END)
            if feature == 'day_of_year': entry_widget.insert(0, str(now.timetuple().tm_yday))
            elif feature == 'month': entry_widget.insert(0, str(now.month))
            elif feature == 'day_of_week': entry_widget.insert(0, str(now.weekday()))
            # Para humidity e meanpressure, poderia buscar dados em tempo real se tivesse API, ou manter média
            elif not self.df_original.empty and feature in self.df_original.columns:
                 entry_widget.insert(0, f"{self.df_original[feature].mean():.1f}")


    def predict_temperature(self):
        if self.model is None:
            messagebox.showerror("Erro de Modelo", "O modelo de predição não foi carregado ou treinado.")
            return
        input_values = {}
        try:
            for feature in self.features:
                value_str = self.entry_widgets[feature].get()
                if not value_str: messagebox.showerror("Erro de Entrada", f"O campo '{feature}' não pode estar vazio."); return
                input_values[feature] = float(value_str)
            if not (0 <= input_values.get('humidity', 50) <= 100): messagebox.showerror("Erro de Entrada", "Umidade deve estar entre 0 e 100."); return
            input_data = pd.DataFrame([input_values], columns=self.features)
            predicted_temp = self.model.predict(input_data)[0]
            self.result_label.config(text=f"Temperatura Média Predita: {predicted_temp:.2f} °C")
            if not self.df_original.empty:
                try:
                    current_year = self.df_original['date'].dt.year.max() if 'date' in self.df_original.columns else datetime.datetime.now().year
                    day_of_year_input = int(input_values['day_of_year'])
                    day_of_year_viz = 365 if not (datetime.date(current_year, 1, 1) + datetime.timedelta(days=365)).year == current_year and day_of_year_input == 366 else day_of_year_input
                    simulated_date = datetime.datetime(current_year, 1, 1) + datetime.timedelta(days=day_of_year_viz - 1)
                    try: simulated_date = simulated_date.replace(month=int(input_values['month']))
                    except ValueError: pass
                except ValueError as ve:
                     messagebox.showwarning("Aviso de Data", f"Não foi possível gerar data exata para visualização ({ve}). Usando data aproximada.")
                     simulated_date = datetime.datetime(current_year, int(input_values['month']), 1)
                self.update_plot(simulated_date, predicted_temp)
        except ValueError: messagebox.showerror("Erro de Entrada", "Por favor, insira valores numéricos válidos.")
        except Exception as e: messagebox.showerror("Erro na Predição", f"Ocorreu um erro: {e}")

    def plot_initial_data(self):
        self.ax.clear()
        if 'date' in self.df_original.columns and 'meantemp' in self.df_original.columns:
            self.ax.plot(self.df_original['date'], self.df_original['meantemp'], label='Histórico Temp. Média', color='cornflowerblue', alpha=0.8)
        else: self.ax.text(0.5, 0.5, "Colunas 'date' ou 'meantemp'\n não encontradas.", ha='center', va='center', transform=self.ax.transAxes)
        self.ax.set_xlabel("Data"); self.ax.set_ylabel("Temperatura Média (°C)")
        self.ax.set_title("Histórico de Temperatura Média Diária"); self.ax.legend(loc='upper left')
        self.fig.autofmt_xdate(); self.canvas_plot.draw()

    def update_plot(self, prediction_date, predicted_temp):
        self.plot_initial_data()
        self.ax.plot(prediction_date, predicted_temp, 'o', color='red', markersize=8, label=f'Predição ({prediction_date.strftime("%d/%m")}): {predicted_temp:.2f}°C')
        self.ax.legend(loc='upper left'); self.fig.autofmt_xdate(); self.canvas_plot.draw()

# --- Interface do Simulador de Abrigo ---
class ShelterSimulatorInterface:
    def __init__(self, master_frame):
        self.master = master_frame
        self.master.configure(style="Tab.TFrame")
        self.shelters = [dict(s) for s in SHELTERS_DATA] # Cópia para manipulação

        # --- Frames Principais ---
        control_frame = ttk.LabelFrame(self.master, text="Critérios de Busca e Controlo", padding="10", style="TLabelframe")
        control_frame.pack(pady=10, padx=10, fill="x")

        list_frame = ttk.LabelFrame(self.master, text="Lista de Abrigos Disponíveis", padding="10", style="TLabelframe")
        list_frame.pack(pady=10, padx=10, fill="both", expand=True)

        recommendation_frame = ttk.LabelFrame(self.master, text="Abrigo Recomendado", padding="10", style="TLabelframe")
        recommendation_frame.pack(pady=10, padx=10, fill="x")

        # --- Controlo e Inputs ---
        input_criteria_frame = ttk.Frame(control_frame, style="TFrame") # Subframe para inputs
        input_criteria_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        simulation_buttons_frame = ttk.Frame(control_frame, style="TFrame") # Subframe para botões de simulação
        simulation_buttons_frame.pack(side=tk.RIGHT, padx=5)

        ttk.Label(input_criteria_frame, text="Nº de Pessoas:", style="TLabel").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.num_people_var = tk.StringVar(value="1")
        self.num_people_entry = ttk.Entry(input_criteria_frame, textvariable=self.num_people_var, width=5, font=FONT_INPUT_NORMAL)
        self.num_people_entry.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        ttk.Label(input_criteria_frame, text="Prioridade Segurança:", style="TLabel").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.safety_priority_var = tk.StringVar(value="Qualquer")
        safety_options = ["Qualquer"] + list(SAFETY_LEVEL_MAP.keys())
        self.safety_priority_combo = ttk.Combobox(input_criteria_frame, textvariable=self.safety_priority_var, values=safety_options, state="readonly", width=12, font=FONT_INPUT_NORMAL)
        self.safety_priority_combo.grid(row=1, column=1, padx=5, pady=5, sticky="w")
        
        # Checkboxes para Recursos
        self.resource_vars = {}
        ttk.Label(input_criteria_frame, text="Recursos Essenciais:", style="TLabel").grid(row=2, column=0, columnspan=2, padx=5, pady=(10,0), sticky="w")
        resources_frame = ttk.Frame(input_criteria_frame, style="TFrame")
        resources_frame.grid(row=3, column=0, columnspan=2, padx=5, pady=2, sticky="w")
        for i, resource_name in enumerate(POSSIBLE_RESOURCES):
            var = tk.BooleanVar()
            cb = ttk.Checkbutton(resources_frame, text=resource_name.replace("_", " ").capitalize(), variable=var, style="TCheckbutton")
            cb.pack(side=tk.LEFT, padx=3)
            self.resource_vars[resource_name] = var
            if resource_name in ["água", "comida"]: var.set(True) # Pre-selecionar alguns comuns

        self.find_shelter_button = ttk.Button(simulation_buttons_frame, text="Encontrar Melhor Abrigo", command=self.find_best_shelter, style="Accent.TButton")
        self.find_shelter_button.pack(pady=5, fill=tk.X)
        self.simulate_occupancy_button = ttk.Button(simulation_buttons_frame, text="Simular Ocupação", command=self.simulate_shelter_occupancy)
        self.simulate_occupancy_button.pack(pady=5, fill=tk.X)


        # --- Lista de Abrigos (Treeview) ---
        cols = ("id", "name", "capacity_total", "capacity_current", "available", "safety_level", "resources")
        col_names = ("ID", "Nome do Abrigo", "Cap. Total", "Ocupação", "Vagas", "Segurança", "Recursos")
        col_widths = (50, 200, 80, 80, 60, 80, 200)

        self.shelter_tree = ttk.Treeview(list_frame, columns=cols, show="headings", height=7)
        for col, name, width in zip(cols, col_names, col_widths):
            self.shelter_tree.heading(col, text=name)
            self.shelter_tree.column(col, width=width, anchor='center')
        
        # Scrollbar para a Treeview
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.shelter_tree.yview)
        self.shelter_tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self.shelter_tree.pack(fill="both", expand=True)
        self.shelter_tree.tag_configure('recommended', background='lightgreen') # Tag para destacar

        # --- Abrigo Recomendado ---
        self.recommended_shelter_label = ttk.Label(recommendation_frame, text="Nenhum abrigo recomendado ainda. Ajuste os critérios e procure.", style="Result.TLabel", wraplength=600, justify="center", font=('Helvetica', 11))
        self.recommended_shelter_label.pack(pady=10)

        self.populate_shelter_tree()

    def populate_shelter_tree(self, recommended_id=None):
        for i in self.shelter_tree.get_children():
            self.shelter_tree.delete(i)
        for shelter in self.shelters:
            available = shelter["capacity_total"] - shelter["capacity_current"]
            resources_str = ", ".join(shelter["resources"]).capitalize()
            
            tags = ()
            if recommended_id and shelter["id"] == recommended_id:
                tags = ('recommended',)

            self.shelter_tree.insert("", "end", values=(
                shelter["id"],
                shelter["name"],
                shelter["capacity_total"],
                shelter["capacity_current"],
                max(0, available), # Não mostrar vagas negativas
                shelter["safety_level"].capitalize(),
                resources_str
            ), tags=tags)

    def simulate_shelter_occupancy(self):
        for shelter in self.shelters:
            # Simula uma mudança na ocupação, mas não excede a capacidade total
            change = random.randint(-shelter["capacity_total"] // 4, shelter["capacity_total"] // 4)
            shelter["capacity_current"] += change
            shelter["capacity_current"] = max(0, min(shelter["capacity_current"], shelter["capacity_total"]))
        self.populate_shelter_tree()
        self.recommended_shelter_label.config(text="Ocupação dos abrigos simulada. Procure novamente se necessário.")


    def find_best_shelter(self):
        try:
            num_people = int(self.num_people_var.get())
            if num_people <= 0:
                messagebox.showerror("Entrada Inválida", "Número de pessoas deve ser maior que zero.")
                return
        except ValueError:
            messagebox.showerror("Entrada Inválida", "Número de pessoas deve ser um valor numérico.")
            return

        required_resources = [res_name for res_name, var in self.resource_vars.items() if var.get()]
        safety_priority = self.safety_priority_var.get()

        candidate_shelters = []
        for shelter in self.shelters:
            available_capacity = shelter["capacity_total"] - shelter["capacity_current"]
            if available_capacity < num_people:
                continue # Não tem capacidade suficiente

            # Verificar recursos
            has_all_required_resources = True
            for req_res in required_resources:
                if req_res not in shelter["resources"]:
                    has_all_required_resources = False
                    break
            if not has_all_required_resources:
                continue

            # Verificar prioridade de segurança
            shelter_safety_val = SAFETY_LEVEL_MAP.get(shelter["safety_level"], 0)
            if safety_priority != "Qualquer":
                priority_val = SAFETY_LEVEL_MAP.get(safety_priority, 0)
                if shelter_safety_val < priority_val: # Se a prioridade é alta, segurança baixa/média não serve.
                    continue
                # Se a prioridade é média, segurança baixa não serve.
                if priority_val == SAFETY_LEVEL_MAP["média"] and shelter_safety_val == SAFETY_LEVEL_MAP["baixa"]:
                    continue


            # Se passou por todos os filtros, é um candidato
            # Adiciona uma pontuação para ordenação (maior é melhor)
            # Pontuação: Nível de segurança (peso maior) + Vagas disponíveis (normalizado, peso menor)
            score = shelter_safety_val * 100 + (available_capacity - num_people) 
            candidate_shelters.append((score, shelter))
        
        if not candidate_shelters:
            self.recommended_shelter_label.config(text="Nenhum abrigo encontrado que corresponda a todos os critérios.")
            self.populate_shelter_tree() # Atualiza a treeview sem destaque
            return

        # Ordenar por pontuação (maior primeiro)
        candidate_shelters.sort(key=lambda x: x[0], reverse=True)
        best_shelter_data = candidate_shelters[0][1]

        recommendation_text = (
            f"Melhor Abrigo Encontrado: {best_shelter_data['name']} (ID: {best_shelter_data['id']})\n"
            f"Vagas Disponíveis: {best_shelter_data['capacity_total'] - best_shelter_data['capacity_current']}\n"
            f"Nível de Segurança: {best_shelter_data['safety_level'].capitalize()}\n"
            f"Recursos: {', '.join(best_shelter_data['resources']).capitalize()}"
        )
        self.recommended_shelter_label.config(text=recommendation_text)
        self.populate_shelter_tree(recommended_id=best_shelter_data["id"])


# --- Aplicação Principal com Abas ---
class SensorDashboardApp:
    def __init__(self, root, model_pred, features_pred, df_pred):
        self.root = root
        self.root.title("Dashboard de Sensores, Previsão e Abrigos")
        self.root.geometry("950x750") # Aumentado para a nova aba
        self.root.configure(bg=COLOR_WINDOW_BG)
        self.root.minsize(900, 700)

        style = ttk.Style()
        try: style.theme_use('clam')
        except tk.TclError: style.theme_use(style.theme_names()[0])

        style.configure("TFrame", background=COLOR_WINDOW_BG)
        style.configure("Gauge.TFrame", background=COLOR_FRAME_BG, relief="groove", borderwidth=1)
        style.configure("TLabel", background=COLOR_WINDOW_BG, foreground=COLOR_LABEL_TEXT, font=FONT_LABEL_NORMAL)
        style.configure("MainTitle.TLabel", font=FONT_MAIN_TITLE, background=COLOR_WINDOW_BG, foreground="#333333")
        style.configure("GaugeTitle.TLabel", font=FONT_GAUGE_TITLE, background=COLOR_FRAME_BG, foreground="#424242")
        style.configure("GaugeValue.TLabel", font=FONT_GAUGE_VALUE, background=COLOR_FRAME_BG)
        style.configure("TNotebook", background=COLOR_WINDOW_BG, borderwidth=0)
        style.configure("TNotebook.Tab", font=FONT_TAB_HEADER, padding=[10, 5], background=COLOR_TAB_BG, foreground="#555555")
        style.map("TNotebook.Tab", background=[("selected", COLOR_FRAME_BG)], foreground=[("selected", "#00796B")])
        style.configure("Tab.TFrame", background=COLOR_FRAME_BG)
        style.configure("TLabelframe", background=COLOR_FRAME_BG, bordercolor="#BDBDBD", relief="solid")
        style.configure("TLabelframe.Label", background=COLOR_FRAME_BG, foreground="#455A64", font=('Helvetica', 11, 'bold'))
        style.configure("Accent.TButton", font=FONT_BUTTON_NORMAL, background="#00796B", foreground="white")
        style.map("Accent.TButton", background=[('active', '#004D40')])
        style.configure("Result.TLabel", background=COLOR_FRAME_BG, anchor="center")
        style.configure("TEntry", font=FONT_INPUT_NORMAL)
        style.configure("TCombobox", font=FONT_INPUT_NORMAL)
        style.configure("TCheckbutton", background=COLOR_FRAME_BG, font=FONT_LABEL_NORMAL)
        style.configure('Treeview.Heading', background=COLOR_TREEVIEW_HEADING, font=('Helvetica', 10, 'bold'))


        self.notebook = ttk.Notebook(self.root)

        self.sensors_tab_frame = ttk.Frame(self.notebook, style="Tab.TFrame")
        self.notebook.add(self.sensors_tab_frame, text=" Sensores em Tempo Real ")
        self._setup_sensors_ui(self.sensors_tab_frame)

        if model_pred is not None:
            self.predictor_tab_frame = ttk.Frame(self.notebook, style="Tab.TFrame")
            self.notebook.add(self.predictor_tab_frame, text=" Previsão de Temperatura ")
            self.temp_predictor_ui = TemperaturePredictorInterface(self.predictor_tab_frame, model_pred, features_pred, df_pred)
        else:
            error_tab_frame = ttk.Frame(self.notebook, style="Tab.TFrame")
            self.notebook.add(error_tab_frame, text=" Previsão Indisponível ")
            ttk.Label(error_tab_frame, text="A funcionalidade de previsão de temperatura não pôde ser carregada.", font=("Helvetica", 12), style="TLabel", wraplength=380, justify="center").pack(expand=True, padx=20, pady=20)

        # Aba Simulador de Abrigo
        self.shelter_sim_tab_frame = ttk.Frame(self.notebook, style="Tab.TFrame")
        self.notebook.add(self.shelter_sim_tab_frame, text=" Simulador de Abrigo ")
        self.shelter_sim_ui = ShelterSimulatorInterface(self.shelter_sim_tab_frame)

        self.notebook.pack(expand=True, fill='both', padx=10, pady=10)

    def _setup_sensors_ui(self, parent_frame):
        main_title_label = ttk.Label(parent_frame, text="Monitor de Sensores", style="MainTitle.TLabel")
        main_title_label.pack(pady=20)
        gauges_container = ttk.Frame(parent_frame, style="TFrame")
        gauges_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        for i in range(3): gauges_container.columnconfigure(i, weight=1)
        gauges_container.rowconfigure(0, weight=1)
        self.temp_gauge = Gauge(gauges_container, "Temperatura", "°C", TEMP_MIN, TEMP_MAX, TEMP_ALERT_THRESHOLD, is_float=True)
        self.temp_gauge.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self.occ_gauge = Gauge(gauges_container, "Ocupação", "%", OCC_MIN, OCC_MAX, OCC_ALERT_THRESHOLD)
        self.occ_gauge.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        self.hum_gauge = Gauge(gauges_container, "Umidade", "%", HUM_MIN_SENSOR, HUM_MAX_SENSOR, HUM_ALERT_THRESHOLD_SENSOR)
        self.hum_gauge.grid(row=0, column=2, padx=10, pady=10, sticky="nsew")
        self.update_all_sensors_periodically()

    def update_all_sensors_periodically(self):
        self.temp_gauge.update_value(get_random_temperature_sensor())
        self.occ_gauge.update_value(get_random_occupancy_sensor())
        self.hum_gauge.update_value(get_random_humidity_sensor())
        self.root.after(UPDATE_INTERVAL, self.update_all_sensors_periodically)

# --- Ponto de Entrada Principal ---
if __name__ == "__main__":
    model_ready = load_and_train_prediction_model()
    root_app = tk.Tk()
    app_instance = SensorDashboardApp(root_app, model_predictor, features_predictor, df_predictor_original)
    root_app.mainloop()
