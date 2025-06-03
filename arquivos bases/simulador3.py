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


UPDATE_INTERVAL = 2000  
TEMP_MIN, TEMP_MAX = 10.0, 40.0  
OCC_MIN, OCC_MAX = 0, 100    
HUM_MIN_SENSOR, HUM_MAX_SENSOR = 20, 100

TEMP_ALERT_THRESHOLD = 35.0
OCC_ALERT_THRESHOLD = 80
HUM_ALERT_THRESHOLD_SENSOR = 85 

COLOR_NORMAL_GAUGE = "#4CAF50"
COLOR_ALERT_GAUGE = "#F44336"
COLOR_GAUGE_BACKGROUND = "#E0E0E0"
COLOR_TEXT_ON_GAUGE = "#333333"
COLOR_WINDOW_BG = "#ECEFF1"
COLOR_FRAME_BG = "#FFFFFF" 
COLOR_TAB_BG = "#FAFAFA"  
COLOR_LABEL_TEXT = "#263238"

FONT_MAIN_TITLE = ("Helvetica", 18, "bold")
FONT_GAUGE_TITLE = ("Helvetica", 12)
FONT_GAUGE_VALUE = ("Helvetica", 20, "bold")
FONT_GAUGE_MINMAX = ("Helvetica", 9)
FONT_TAB_HEADER = ("Helvetica", 11, "bold")

DATA_FILE = 'DailyDelhiClimateTrain.csv'
model_predictor = None
features_predictor = []
df_predictor_original = None

def load_and_train_prediction_model():
    """Carrega os dados, pré-processa e treina o modelo de previsão de temperatura."""
    global model_predictor, features_predictor, df_predictor_original

    if not os.path.exists(DATA_FILE):
        messagebox.showerror("Erro de Arquivo",
                             f"O arquivo '{DATA_FILE}' não foi encontrado.\n"
                             "Por favor, baixe-o do Kaggle (Daily Delhi Climate) e coloque-o "
                             "no mesmo diretório do script.\n\nA aplicação não pode iniciar sem este arquivo.")
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
                                         "Verifique o arquivo CSV.")
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
        print(f"--- Avaliação do Modelo de Previsão (no console) ---")
        print(f"MSE: {mse:.2f}, R²: {r2:.2f}, RMSE: {np.sqrt(mse):.2f}")
        
        return True

    except Exception as e:
        messagebox.showerror("Erro no Processamento de Dados",
                             f"Ocorreu um erro durante o carregamento ou treinamento do modelo de previsão:\n{e}")
        return False

def get_random_temperature_sensor():
    return round(random.uniform(TEMP_MIN, TEMP_MAX), 1)

def get_random_occupancy_sensor():
    return random.randint(OCC_MIN, OCC_MAX)

def get_random_humidity_sensor():
    return random.randint(HUM_MIN_SENSOR, HUM_MAX_SENSOR)


class Gauge(ttk.Frame):
    def __init__(self, parent, title, unit, min_val, max_val, alert_threshold, is_float=False):
        super().__init__(parent, padding="10 10 10 10")
        self.configure(style="Gauge.TFrame")

        self.title_text = title
        self.unit = unit
        self.min_val = min_val
        self.max_val = max_val
        self.alert_threshold = alert_threshold
        self.current_value = min_val
        self.is_float = is_float

        self.canvas_width = 150
        self.gauge_thickness = 16
        self.canvas_height = (self.canvas_width / 2) + self.gauge_thickness + 15

        self.title_label = ttk.Label(self, text=self.title_text, font=FONT_GAUGE_TITLE, style="GaugeTitle.TLabel")
        self.title_label.pack(pady=(0, 6))

        self.canvas = tk.Canvas(self, width=self.canvas_width, height=self.canvas_height,
                                bg=COLOR_FRAME_BG, highlightthickness=0)
        self.canvas.pack()

        value_text_formatted = f"{self.current_value:.1f}{self.unit}" if self.is_float else f"{self.current_value}{self.unit}"
        self.value_label = ttk.Label(self, text=value_text_formatted, font=FONT_GAUGE_VALUE, style="GaugeValue.TLabel")
        self.value_label.pack(pady=(6,0))

        self._draw_gauge()

    def _draw_gauge(self):
        self.canvas.delete("all")
        arc_bbox_x1 = self.gauge_thickness
        arc_bbox_y1 = self.gauge_thickness
        arc_bbox_x2 = self.canvas_width - self.gauge_thickness
        arc_bbox_y2 = self.canvas_width - self.gauge_thickness

        self.canvas.create_arc(
            arc_bbox_x1, arc_bbox_y1, arc_bbox_x2, arc_bbox_y2,
            start=0, extent=180,
            outline=COLOR_GAUGE_BACKGROUND, width=self.gauge_thickness, style=tk.ARC
        )

        value_range = self.max_val - self.min_val
        percentage = (self.current_value - self.min_val) / value_range if value_range != 0 else 0.0
        percentage = max(0.0, min(1.0, percentage))
        value_sweep_angle = 180 * percentage

        current_gauge_color = COLOR_ALERT_GAUGE if self.current_value > self.alert_threshold else COLOR_NORMAL_GAUGE

        if value_sweep_angle > 0:
            self.canvas.create_arc(
                arc_bbox_x1, arc_bbox_y1, arc_bbox_x2, arc_bbox_y2,
                start=180, extent=-value_sweep_angle,
                outline=current_gauge_color, width=self.gauge_thickness, style=tk.ARC
            )

        arc_center_y = arc_bbox_y1 + (arc_bbox_y2 - arc_bbox_y1) / 2
        text_offset_y = 10 
        self.canvas.create_text(arc_bbox_x1 - 5, arc_center_y + text_offset_y, text=str(self.min_val),
                                anchor="ne", font=FONT_GAUGE_MINMAX, fill=COLOR_TEXT_ON_GAUGE)
        self.canvas.create_text(arc_bbox_x2 + 5, arc_center_y + text_offset_y, text=str(self.max_val),
                                anchor="nw", font=FONT_GAUGE_MINMAX, fill=COLOR_TEXT_ON_GAUGE)

    def update_value(self, new_value):
        self.current_value = new_value
        value_text_formatted = f"{self.current_value:.1f}{self.unit}" if self.is_float else f"{self.current_value}{self.unit}"
        self.value_label.config(text=value_text_formatted)
        self._draw_gauge()

class TemperaturePredictorInterface:
    def __init__(self, master_frame, model, features_list, df_hist_data):
        self.master = master_frame
        self.master.configure(style="Tab.TFrame") 

        self.model = model
        self.features = features_list
        self.df_original = df_hist_data.copy() if df_hist_data is not None else pd.DataFrame()

       
        self.input_frame = ttk.LabelFrame(self.master, text="Entrada de Dados para Predição", padding="15", style="TLabelframe")
        self.input_frame.pack(pady=10, padx=20, fill="x")

        self.create_input_widgets(self.input_frame)

        self.predict_button = ttk.Button(self.master, text="Prever Temperatura Média", command=self.predict_temperature, style="Accent.TButton")
        self.predict_button.pack(pady=8)

        self.result_frame = ttk.LabelFrame(self.master, text="Resultado da Predição", padding="15", style="TLabelframe")
        self.result_frame.pack(pady=8, padx=20, fill="x")

        self.result_label = ttk.Label(self.result_frame, text="Temperatura Média Predita: -- °C",
                                      font=('Helvetica', 14, 'bold'), foreground='#00695C', style="Result.TLabel") # Verde escuro
        self.result_label.pack(pady=8)

        self.chart_frame = ttk.LabelFrame(self.master, text="Visualização da Predição", padding="15", style="TLabelframe")
        self.chart_frame.pack(pady=10, padx=20, fill="both", expand=True)

        self.fig, self.ax = plt.subplots(figsize=(7, 4)) 
        plt.style.use('seaborn-v0_8-whitegrid') 
        self.canvas_plot = FigureCanvasTkAgg(self.fig, master=self.chart_frame)
        self.canvas_widget = self.canvas_plot.get_tk_widget()
        self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        if not self.df_original.empty:
            self.plot_initial_data()
        else:
            self.ax.text(0.5, 0.5, "Dados históricos não disponíveis.", horizontalalignment='center', verticalalignment='center', transform=self.ax.transAxes)
            self.canvas_plot.draw()


    def create_input_widgets(self, parent_frame):
        feature_labels = {
            'humidity': 'Umidade (%)', 'meanpressure': 'Pressão (hPa)',
            'day_of_year': 'Dia do Ano (1-366)', 'month': 'Mês (1-12)',
            'day_of_week': 'Dia da Semana (0=Seg, 6=Dom)'
        }
        self.entry_widgets = {}
        
        
        num_features = len(self.features)
        cols = 2
        rows_per_col = (num_features + cols -1) // cols


        for i, feature in enumerate(self.features):
            row_idx = i % rows_per_col
            col_idx = (i // rows_per_col) * 2 

            ttk.Label(parent_frame, text=f"{feature_labels.get(feature, feature.capitalize())}:").grid(
                row=row_idx, column=col_idx, padx=5, pady=5, sticky="w")
            entry = ttk.Entry(parent_frame, width=12, font=('Helvetica', 10))
            entry.grid(row=row_idx, column=col_idx + 1, padx=5, pady=5, sticky="ew")
            self.entry_widgets[feature] = entry

            
            if not self.df_original.empty:
                if feature in ['humidity', 'meanpressure'] and feature in self.df_original.columns:
                    mean_val = self.df_original[feature].mean()
                    entry.insert(0, f"{mean_val:.1f}")
            if feature == 'day_of_year': entry.insert(0, str(datetime.datetime.now().timetuple().tm_yday))
            elif feature == 'month': entry.insert(0, str(datetime.datetime.now().month))
            elif feature == 'day_of_week': entry.insert(0, str(datetime.datetime.now().weekday()))
        
        
        parent_frame.columnconfigure(1, weight=1)
        if num_features > rows_per_col:
             parent_frame.columnconfigure(3, weight=1)


        self.use_current_time_button = ttk.Button(parent_frame, text="Usar Data Atual", command=self.set_current_time)
        self.use_current_time_button.grid(row=rows_per_col, column=0, columnspan=max(2, cols*2), pady=10)


    def set_current_time(self):
        now = datetime.datetime.now()
        if 'day_of_year' in self.entry_widgets:
            self.entry_widgets['day_of_year'].delete(0, tk.END); self.entry_widgets['day_of_year'].insert(0, str(now.timetuple().tm_yday))
        if 'month' in self.entry_widgets:
            self.entry_widgets['month'].delete(0, tk.END); self.entry_widgets['month'].insert(0, str(now.month))
        if 'day_of_week' in self.entry_widgets:
            self.entry_widgets['day_of_week'].delete(0, tk.END); self.entry_widgets['day_of_week'].insert(0, str(now.weekday()))

    def predict_temperature(self):
        if self.model is None:
            messagebox.showerror("Erro de Modelo", "O modelo de predição não foi carregado ou treinado.")
            return

        input_values = {}
        try:
            for feature in self.features:
                value_str = self.entry_widgets[feature].get()
                if not value_str:
                    messagebox.showerror("Erro de Entrada", f"O campo '{feature}' não pode estar vazio.")
                    return
                input_values[feature] = float(value_str)
            
            
            if not (0 <= input_values.get('humidity', 50) <= 100):
                messagebox.showerror("Erro de Entrada", "Umidade deve estar entre 0 e 100.")
                return
            

            input_data = pd.DataFrame([input_values], columns=self.features)
            predicted_temp = self.model.predict(input_data)[0]
            self.result_label.config(text=f"Temperatura Média Predita: {predicted_temp:.2f} °C")

            if not self.df_original.empty:
                try:
                    current_year = self.df_original['date'].dt.year.max() if 'date' in self.df_original.columns else datetime.datetime.now().year
                    
                    day_of_year_input = int(input_values['day_of_year'])
                    if not (datetime.date(current_year, 1, 1) + datetime.timedelta(days=365)).year == current_year and day_of_year_input == 366: # Não é bissexto e dia 366
                        day_of_year_viz = 365 
                    else:
                        day_of_year_viz = day_of_year_input

                    simulated_date = datetime.datetime(current_year, 1, 1) + datetime.timedelta(days=day_of_year_viz - 1)
                    
                    
                    try:
                        simulated_date = simulated_date.replace(month=int(input_values['month']))
                    except ValueError: 
                        pass


                except ValueError as ve: 
                     messagebox.showwarning("Aviso de Data", f"Não foi possível gerar data exata para visualização ({ve}). Usando data aproximada.")
                     simulated_date = datetime.datetime(current_year, int(input_values['month']), 1) # Usa o primeiro do mês

                self.update_plot(simulated_date, predicted_temp)

        except ValueError:
            messagebox.showerror("Erro de Entrada", "Por favor, insira valores numéricos válidos.")
        except Exception as e:
            messagebox.showerror("Erro na Predição", f"Ocorreu um erro: {e}")

    def plot_initial_data(self):
        self.ax.clear()
        if 'date' in self.df_original.columns and 'meantemp' in self.df_original.columns:
            self.ax.plot(self.df_original['date'], self.df_original['meantemp'], label='Histórico Temp. Média', color='cornflowerblue', alpha=0.8)
        else:
             self.ax.text(0.5, 0.5, "Colunas 'date' ou 'meantemp'\n não encontradas nos dados históricos.", ha='center', va='center', transform=self.ax.transAxes)

        self.ax.set_xlabel("Data")
        self.ax.set_ylabel("Temperatura Média (°C)")
        self.ax.set_title("Histórico de Temperatura Média Diária")
        self.ax.legend(loc='upper left')
        self.fig.autofmt_xdate()
        self.canvas_plot.draw()

    def update_plot(self, prediction_date, predicted_temp):
        self.plot_initial_data() 
        self.ax.plot(prediction_date, predicted_temp, 'o', color='red', markersize=8,
                     label=f'Predição ({prediction_date.strftime("%d/%m")}): {predicted_temp:.2f}°C')
        self.ax.legend(loc='upper left')
        self.fig.autofmt_xdate()
        self.canvas_plot.draw()


class SensorDashboardApp:
    def __init__(self, root, model_pred, features_pred, df_pred):
        self.root = root
        self.root.title("Dashboard de Sensores e Previsão Climática")
        
        self.root.geometry("900x700") 
        self.root.configure(bg=COLOR_WINDOW_BG)
        self.root.minsize(800, 600) 

        
        style = ttk.Style()
        try:
            style.theme_use('clam')
        except tk.TclError:
            style.theme_use(style.theme_names()[0])

        style.configure("TFrame", background=COLOR_WINDOW_BG)
        style.configure("Gauge.TFrame", background=COLOR_FRAME_BG, relief="groove", borderwidth=1)
        style.configure("TLabel", background=COLOR_WINDOW_BG, foreground=COLOR_LABEL_TEXT)
        style.configure("MainTitle.TLabel", font=FONT_MAIN_TITLE, background=COLOR_WINDOW_BG, foreground="#333333")
        style.configure("GaugeTitle.TLabel", font=FONT_GAUGE_TITLE, background=COLOR_FRAME_BG, foreground="#424242")
        style.configure("GaugeValue.TLabel", font=FONT_GAUGE_VALUE, background=COLOR_FRAME_BG)
        
        # Estilos para abas e seus conteúdos
        style.configure("TNotebook", background=COLOR_WINDOW_BG, borderwidth=0)
        style.configure("TNotebook.Tab", font=FONT_TAB_HEADER, padding=[10, 5], background=COLOR_TAB_BG, foreground="#555555")
        style.map("TNotebook.Tab",
            background=[("selected", COLOR_FRAME_BG)], 
            foreground=[("selected", "#00796B")]) 
        
        style.configure("Tab.TFrame", background=COLOR_FRAME_BG) 
        style.configure("TLabelframe", background=COLOR_FRAME_BG, bordercolor="#BDBDBD", relief="solid")
        style.configure("TLabelframe.Label", background=COLOR_FRAME_BG, foreground="#455A64", font=('Helvetica', 11, 'bold'))
        style.configure("Accent.TButton", font=('Helvetica', 10, 'bold'), background="#00796B", foreground="white") # Botão de destaque
        style.map("Accent.TButton", background=[('active', '#004D40')])
        style.configure("Result.TLabel", background=COLOR_FRAME_BG)


        
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
            ttk.Label(error_tab_frame, text="A funcionalidade de previsão de temperatura não pôde ser carregada.",
                      font=("Helvetica", 12), style="TLabel", wraplength=380, justify="center").pack(expand=True, padx=20, pady=20)


        self.notebook.pack(expand=True, fill='both', padx=10, pady=10)

    def _setup_sensors_ui(self, parent_frame):
        main_title_label = ttk.Label(parent_frame, text="Monitor de Sensores", style="MainTitle.TLabel")
        main_title_label.pack(pady=20)

        gauges_container = ttk.Frame(parent_frame, style="TFrame")
        gauges_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        gauges_container.columnconfigure(0, weight=1)
        gauges_container.columnconfigure(1, weight=1)
        gauges_container.columnconfigure(2, weight=1)
        gauges_container.rowconfigure(0, weight=1)


        self.temp_gauge = Gauge(gauges_container, "Temperatura", "°C", TEMP_MIN, TEMP_MAX, TEMP_ALERT_THRESHOLD, is_float=True)
        self.temp_gauge.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        self.occ_gauge = Gauge(gauges_container, "Ocupação", "%", OCC_MIN, OCC_MAX, OCC_ALERT_THRESHOLD)
        self.occ_gauge.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        self.hum_gauge = Gauge(gauges_container, "Umidade", "%", HUM_MIN_SENSOR, HUM_MAX_SENSOR, HUM_ALERT_THRESHOLD_SENSOR)
        self.hum_gauge.grid(row=0, column=2, padx=10, pady=10, sticky="nsew")

        self.update_all_sensors_periodically()

    def update_all_sensors_periodically(self):
        temp_val = get_random_temperature_sensor()
        occ_val = get_random_occupancy_sensor()
        hum_val = get_random_humidity_sensor()

        self.temp_gauge.update_value(temp_val)
        self.occ_gauge.update_value(occ_val)
        self.hum_gauge.update_value(hum_val)

        self.root.after(UPDATE_INTERVAL, self.update_all_sensors_periodically)


if __name__ == "__main__":
    
    model_ready = load_and_train_prediction_model()

    
    root_app = tk.Tk()
   
    app_instance = SensorDashboardApp(root_app, model_predictor, features_predictor, df_predictor_original)
    root_app.mainloop()
