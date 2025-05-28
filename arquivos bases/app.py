import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, messagebox
import datetime


def generate_synthetic_weather_data(num_samples=1000):
    """Gera dados sintéticos de tempo para demonstração."""
    dates = pd.to_datetime(pd.date_range(start='2020-01-01', periods=num_samples, freq='H'))
    hour_of_day = dates.hour
    day_of_year = dates.dayofyear
    month = dates.month
    year = dates.year

    base_temp = 15 + 10 * np.sin(2 * np.pi * (hour_of_day - 8) / 24) # Variação diária
    seasonal_temp = 10 * np.sin(2 * np.pi * (day_of_year - 90) / 365) # Variação sazonal (baseado no dia do ano)
    noise = np.random.normal(0, 2, num_samples) # Ruído aleatório

    temperature = base_temp + seasonal_temp + noise

    data = pd.DataFrame({
        'Date': dates,
        'Hour': hour_of_day,
        'DayOfYear': day_of_year,
        'Month': month,
        'Year': year,
        'Temperature': temperature
    })
    return data

# Gerar dados
df = generate_synthetic_weather_data(num_samples=5000)
print("Amostra dos dados gerados:")
print(df.head())
print("\nEstatísticas descritivas:")
print(df.describe())

# --- PARTE 2: Pré-processamento e Treinamento do Modelo ---

# Selecionar features (variáveis independentes) e target (variável dependente)
features = ['Hour', 'DayOfYear', 'Month'] # 'Year' pode ser usado se o modelo precisar aprender tendências anuais
target = 'Temperature'

X = df[features]
y = df[target]

# Dividir os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTamanho do conjunto de treino: {len(X_train)}")
print(f"Tamanho do conjunto de teste: {len(X_test)}")

model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1) # n_jobs=-1 usa todos os núcleos
model.fit(X_train, y_train)

# Avaliar o modelo
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nErro Quadrático Médio (MSE): {mse:.2f}")
print(f"Coeficiente de Determinação (R²): {r2:.2f}")

# Visualizar algumas predições vs. valores reais
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.3)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2) # Linha de identidade
plt.xlabel("Temperatura Real")
plt.ylabel("Temperatura Predita")
plt.title("Temperatura Real vs. Predita")
plt.grid(True)
plt.show()


# --- PARTE 3: Simulador com Interface Gráfica (Tkinter) ---

class TemperaturePredictorApp:
    def __init__(self, master):
        self.master = master
        master.title("Simulador de Predição de Temperatura")
        master.geometry("800x700") # Aumentar o tamanho da janela
        master.configure(bg='#f0f0f0') # Cor de fundo

        self.model = model # O modelo treinado
        self.features = features # As features usadas no treinamento

        # Estilos
        self.style = ttk.Style()
        self.style.configure('TFrame', background='#f0f0f0')
        self.style.configure('TLabel', background='#f0f0f0', font=('Arial', 10))
        self.style.configure('TButton', font=('Arial', 10, 'bold'))
        self.style.configure('TEntry', font=('Arial', 10))
        self.style.configure('TCombobox', font=('Arial', 10))

        # Frame de Entrada
        self.input_frame = ttk.LabelFrame(master, text="Entrada de Dados para Predição", padding="15 15 15 15")
        self.input_frame.pack(pady=20, padx=20, fill="x")

        # Rótulos e Campos de Entrada
        self.create_input_widgets(self.input_frame)

        # Botão de Predição
        self.predict_button = ttk.Button(master, text="Prever Temperatura", command=self.predict_temperature)
        self.predict_button.pack(pady=10)

        # Frame de Resultado
        self.result_frame = ttk.LabelFrame(master, text="Resultado da Predição", padding="15 15 15 15")
        self.result_frame.pack(pady=10, padx=20, fill="x")

        self.result_label = ttk.Label(self.result_frame, text="Temperatura Predita: -- °C", font=('Arial', 14, 'bold'), foreground='blue')
        self.result_label.pack(pady=10)

        # Frame para Gráficos
        self.chart_frame = ttk.LabelFrame(master, text="Visualização", padding="15 15 15 15")
        self.chart_frame.pack(pady=10, padx=20, fill="both", expand=True)

        self.fig, self.ax = plt.subplots(figsize=(7, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.chart_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.plot_initial_data()

    def create_input_widgets(self, parent_frame):
        # Hora do dia
        ttk.Label(parent_frame, text="Hora do Dia (0-23):").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.hour_entry = ttk.Entry(parent_frame, width=10)
        self.hour_entry.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        self.hour_entry.insert(0, str(datetime.datetime.now().hour)) # Valor inicial

        # Dia do Ano
        ttk.Label(parent_frame, text="Dia do Ano (1-366):").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.dayofyear_entry = ttk.Entry(parent_frame, width=10)
        self.dayofyear_entry.grid(row=1, column=1, padx=5, pady=5, sticky="w")
        self.dayofyear_entry.insert(0, str(datetime.datetime.now().timetuple().tm_yday)) # Valor inicial

        # Mês
        ttk.Label(parent_frame, text="Mês (1-12):").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.month_entry = ttk.Entry(parent_frame, width=10)
        self.month_entry.grid(row=2, column=1, padx=5, pady=5, sticky="w")
        self.month_entry.insert(0, str(datetime.datetime.now().month)) # Valor inicial

        # Botão para usar data e hora atual
        self.use_current_time_button = ttk.Button(parent_frame, text="Usar Hora Atual", command=self.set_current_time)
        self.use_current_time_button.grid(row=3, column=0, columnspan=2, pady=10)


    def set_current_time(self):
        now = datetime.datetime.now()
        self.hour_entry.delete(0, tk.END)
        self.hour_entry.insert(0, str(now.hour))
        self.dayofyear_entry.delete(0, tk.END)
        self.dayofyear_entry.insert(0, str(now.timetuple().tm_yday))
        self.month_entry.delete(0, tk.END)
        self.month_entry.insert(0, str(now.month))

    def predict_temperature(self):
        try:
            hour = int(self.hour_entry.get())
            day_of_year = int(self.dayofyear_entry.get())
            month = int(self.month_entry.get())

            if not (0 <= hour <= 23):
                messagebox.showerror("Erro de Entrada", "A Hora do Dia deve estar entre 0 e 23.")
                return
            if not (1 <= day_of_year <= 366):
                messagebox.showerror("Erro de Entrada", "O Dia do Ano deve estar entre 1 e 366.")
                return
            if not (1 <= month <= 12):
                messagebox.showerror("Erro de Entrada", "O Mês deve estar entre 1 e 12.")
                return

            # Criar um DataFrame com os dados de entrada
            input_data = pd.DataFrame([[hour, day_of_year, month]], columns=self.features)

            # Fazer a predição
            predicted_temp = self.model.predict(input_data)[0]

            self.result_label.config(text=f"Temperatura Predita: {predicted_temp:.2f} °C")

            # Atualizar o gráfico
            self.update_plot(hour, day_of_year, month, predicted_temp)

        except ValueError:
            messagebox.showerror("Erro de Entrada", "Por favor, insira valores numéricos válidos.")
        except Exception as e:
            messagebox.showerror("Erro", f"Ocorreu um erro na predição: {e}")

    def plot_initial_data(self):
        self.ax.clear()
        self.ax.scatter(df['Hour'], df['Temperature'], alpha=0.1, label='Dados Históricos')
        self.ax.set_xlabel("Hora do Dia")
        self.ax.set_ylabel("Temperatura (°C)")
        self.ax.set_title("Variação da Temperatura ao Longo do Dia")
        self.ax.grid(True)
        self.ax.legend()
        self.canvas.draw()

    def update_plot(self, hour, day_of_year, month, predicted_temp):
        self.ax.clear()
        # Plotar os dados históricos (ou uma amostra deles para não sobrecarregar)
        self.ax.scatter(df['Hour'], df['Temperature'], alpha=0.1, label='Dados Históricos')
        # Plotar o ponto predito
        self.ax.plot(hour, predicted_temp, 'ro', markersize=10, label=f'Predição ({predicted_temp:.2f}°C)')
        self.ax.set_xlabel("Hora do Dia")
        self.ax.set_ylabel("Temperatura (°C)")
        self.ax.set_title("Variação da Temperatura ao Longo do Dia com Predição")
        self.ax.grid(True)
        self.ax.legend()
        self.canvas.draw()


if __name__ == "__main__":
    # Certifique-se de que o dataframe 'df' e o 'model' estão definidos globalmente ou passados para a classe
    # Já estão definidos no escopo global para este script.

    root = tk.Tk()
    app = TemperaturePredictorApp(root)
    root.mainloop()