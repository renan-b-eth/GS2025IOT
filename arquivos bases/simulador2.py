import tkinter as tk
from tkinter import ttk
import random
import math

# --- Configurações Globais ---
UPDATE_INTERVAL = 2000  # Intervalo de atualização em milissegundos (ex: 2000ms = 2 segundos)

# Faixas dos sensores
TEMP_MIN, TEMP_MAX = 22.0, 38.0  # Temperatura em °C
OCC_MIN, OCC_MAX = 0, 100      # Ocupação em %
HUM_MIN, HUM_MAX = 40, 90       # Umidade em %

# Limites para alertas (valores acima destes disparam alerta)
TEMP_ALERT_THRESHOLD = 35.0
OCC_ALERT_THRESHOLD = 80
HUM_ALERT_THRESHOLD = 80 # Limite para umidade alta

# Cores
COLOR_NORMAL_GAUGE = "#4CAF50"  # Verde para estado normal do medidor
COLOR_ALERT_GAUGE = "#F44336"   # Vermelho para estado de alerta do medidor
COLOR_GAUGE_BACKGROUND = "#E0E0E0" # Cor de fundo da trilha do medidor
COLOR_TEXT_ON_GAUGE = "#333333" # Cor para os textos (min/max) no canvas do medidor
COLOR_WINDOW_BG = "#ECEFF1"     # Cor de fundo da janela principal
COLOR_FRAME_BG = "#FFFFFF"      # Cor de fundo para os frames que contêm os medidores
COLOR_LABEL_TEXT = "#263238"    # Cor de texto para labels gerais

# Fontes
FONT_MAIN_TITLE = ("Helvetica", 18, "bold")
FONT_GAUGE_TITLE = ("Helvetica", 12)
FONT_GAUGE_VALUE = ("Helvetica", 20, "bold")
FONT_GAUGE_MINMAX = ("Helvetica", 9)

# --- Geração de Dados Aleatórios para Sensores ---
def get_random_temperature():
    """Gera um valor aleatório de temperatura."""
    return round(random.uniform(TEMP_MIN, TEMP_MAX), 1)

def get_random_occupancy():
    """Gera um valor aleatório de percentual de ocupação."""
    return random.randint(OCC_MIN, OCC_MAX)

def get_random_humidity():
    """Gera um valor aleatório de percentual de umidade."""
    return random.randint(HUM_MIN, HUM_MAX)

# --- Widget de Medidor (Gauge) ---
class Gauge(ttk.Frame):
    """
    Um widget de medidor customizado usando Tkinter Canvas.
    Exibe um valor de sensor em um medidor semicircular com alertas coloridos.
    """
    def __init__(self, parent, title, unit, min_val, max_val, alert_threshold, is_float=False):
        super().__init__(parent, padding="10 10 10 10")
        self.configure(style="Gauge.TFrame") # Aplica estilo customizado para borda, etc.

        self.title_text = title
        self.unit = unit
        self.min_val = min_val
        self.max_val = max_val
        self.alert_threshold = alert_threshold
        self.current_value = min_val
        self.is_float = is_float # Para formatar a exibição do valor corretamente (com ou sem decimal)

        self.canvas_width = 160  # Largura do canvas do medidor
        self.gauge_thickness = 18 # Espessura da linha do medidor
        
        # Altura do canvas: calculada para acomodar o semicírculo e os textos min/max
        self.canvas_height = (self.canvas_width / 2) + self.gauge_thickness + 10 

        # Label do Título do Medidor
        self.title_label = ttk.Label(self, text=self.title_text, font=FONT_GAUGE_TITLE, style="GaugeTitle.TLabel")
        self.title_label.pack(pady=(0, 8)) # Espaçamento abaixo do título

        # Canvas para desenhar o Medidor
        self.canvas = tk.Canvas(self, width=self.canvas_width, height=self.canvas_height,
                                bg=COLOR_FRAME_BG, highlightthickness=0)
        self.canvas.pack()

        # Label do Valor (exibe o valor numérico e a unidade)
        value_text_formatted = f"{self.current_value:.1f}{self.unit}" if self.is_float else f"{self.current_value}{self.unit}"
        self.value_label = ttk.Label(self, text=value_text_formatted, font=FONT_GAUGE_VALUE, style="GaugeValue.TLabel")
        self.value_label.pack(pady=(8,0)) # Espaçamento acima do valor

        self._draw_gauge() # Desenha o medidor inicialmente

    def _draw_gauge(self):
        """Desenha ou redesenha o medidor no canvas."""
        self.canvas.delete("all") # Limpa o canvas antes de redesenhar

        # Coordenadas da caixa delimitadora para o arco do medidor.
        # A linha do arco (com espessura `gauge_thickness`) é centralizada neste caminho.
        arc_bbox_x1 = self.gauge_thickness  # Afastamento da esquerda
        arc_bbox_y1 = self.gauge_thickness  # Afastamento do topo
        arc_bbox_x2 = self.canvas_width - self.gauge_thickness  # Afastamento da direita
        # A caixa delimitadora para o oval do qual o arco é cortado.
        # Para um semicírculo, a altura desta parte da caixa é (X2-X1)/2.
        arc_bbox_y2 = self.canvas_width - self.gauge_thickness # Torna o oval base quadrado

        # Desenha a trilha de fundo do medidor
        self.canvas.create_arc(
            arc_bbox_x1, arc_bbox_y1,
            arc_bbox_x2, arc_bbox_y2,
            start=0, extent=180,  # Semicírculo superior (0 graus = 3h, 180 graus = 9h)
            outline=COLOR_GAUGE_BACKGROUND, width=self.gauge_thickness, style=tk.ARC
        )

        # Calcula o ângulo de varredura para o valor atual
        value_range = self.max_val - self.min_val
        if value_range == 0: # Evita divisão por zero
            percentage = 0.0
        else:
            percentage = (self.current_value - self.min_val) / value_range
        
        percentage = max(0.0, min(1.0, percentage))  # Garante que a porcentagem esteja entre 0 e 1
        
        value_sweep_angle = 180 * percentage  # Ângulo para o arco do valor

        # Determina a cor do arco do valor com base no limite de alerta
        current_gauge_color = COLOR_ALERT_GAUGE if self.current_value > self.alert_threshold else COLOR_NORMAL_GAUGE

        # Desenha o arco do valor
        if value_sweep_angle > 0:  # Desenha apenas se houver um valor para mostrar
            self.canvas.create_arc(
                arc_bbox_x1, arc_bbox_y1,
                arc_bbox_x2, arc_bbox_y2,
                start=180, extent=-value_sweep_angle,  # Começa da esquerda (180 graus) e varre no sentido horário
                outline=current_gauge_color, width=self.gauge_thickness, style=tk.ARC
            )

        # Coordenada Y para o centro da linha de base do arco (diâmetro horizontal)
        arc_center_y = arc_bbox_y1 + (arc_bbox_y2 - arc_bbox_y1) / 2

        # Exibe labels de Mínimo e Máximo no canvas, abaixo do arco
        self.canvas.create_text(arc_bbox_x1 - 5, arc_center_y + 7, text=str(self.min_val),
                                anchor="ne", font=FONT_GAUGE_MINMAX, fill=COLOR_TEXT_ON_GAUGE)
        self.canvas.create_text(arc_bbox_x2 + 5, arc_center_y + 7, text=str(self.max_val),
                                anchor="nw", font=FONT_GAUGE_MINMAX, fill=COLOR_TEXT_ON_GAUGE)

    def update_value(self, new_value):
        """Atualiza o medidor com um novo valor de sensor."""
        self.current_value = new_value
        # Formata o texto do valor para exibição
        value_text_formatted = f"{self.current_value:.1f}{self.unit}" if self.is_float else f"{self.current_value}{self.unit}"
        self.value_label.config(text=value_text_formatted)
        self._draw_gauge() # Redesenha o medidor com o novo valor e cor

# --- Aplicação Principal ---
class SensorSimulatorApp:
    """
    A classe principal da aplicação Simulador de Sensores.
    Configura a interface do usuário e gerencia as atualizações dos sensores.
    """
    def __init__(self, root_window):
        self.root = root_window
        self.root.title("Simulador de Sensores com Medidores")
        self.root.geometry("620x420") # Tamanho da janela ajustado para melhor espaçamento
        self.root.configure(bg=COLOR_WINDOW_BG)
        self.root.resizable(False, False) # Opcional: desabilita redimensionamento da janela

        # --- Configuração de Estilo ttk ---
        style = ttk.Style()
        # Tenta usar um tema que permita mais customização e tenha uma aparência moderna
        try:
            style.theme_use('clam') 
        except tk.TclError:
            # 'clam' pode não estar disponível em todos os sistemas, usa um fallback
            style.theme_use(style.theme_names()[0])


        style.configure("TFrame", background=COLOR_WINDOW_BG)
        # Estilo para o frame de cada medidor individual
        style.configure("Gauge.TFrame", background=COLOR_FRAME_BG, relief="raised", borderwidth=2) 
        
        style.configure("TLabel", background=COLOR_WINDOW_BG, foreground=COLOR_LABEL_TEXT)
        style.configure("MainTitle.TLabel", font=FONT_MAIN_TITLE, background=COLOR_WINDOW_BG, foreground="#1A237E") # Azul escuro para o título
        style.configure("GaugeTitle.TLabel", font=FONT_GAUGE_TITLE, background=COLOR_FRAME_BG, foreground="#37474F") # Cinza escuro para títulos de medidores
        style.configure("GaugeValue.TLabel", font=FONT_GAUGE_VALUE, background=COLOR_FRAME_BG) # Cor do valor será definida pela lógica de alerta implicitamente


        # --- Layout da Interface do Usuário ---
        # Label do Título Principal
        main_title_label = ttk.Label(self.root, text="Dashboard de Sensores", style="MainTitle.TLabel")
        main_title_label.pack(pady=25) # Espaçamento vertical para o título

        # Frame para conter todos os medidores, permitindo que sejam centralizados e espaçados
        gauges_container_frame = ttk.Frame(self.root, style="TFrame")
        gauges_container_frame.pack(fill=tk.X, expand=True, padx=20, pady=10)
        
        # Configura as colunas do container para terem peso igual, para que se expandam igualmente
        gauges_container_frame.columnconfigure(0, weight=1)
        gauges_container_frame.columnconfigure(1, weight=1)
        gauges_container_frame.columnconfigure(2, weight=1)

        # Cria e posiciona as instâncias dos Medidores
        # Medidor de Temperatura
        self.temp_gauge = Gauge(gauges_container_frame, "Temperatura", "°C", TEMP_MIN, TEMP_MAX, TEMP_ALERT_THRESHOLD, is_float=True)
        self.temp_gauge.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        # Medidor de Ocupação
        self.occ_gauge = Gauge(gauges_container_frame, "Ocupação", "%", OCC_MIN, OCC_MAX, OCC_ALERT_THRESHOLD)
        self.occ_gauge.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        # Medidor de Umidade
        self.hum_gauge = Gauge(gauges_container_frame, "Umidade", "%", HUM_MIN, HUM_MAX, HUM_ALERT_THRESHOLD)
        self.hum_gauge.grid(row=0, column=2, padx=10, pady=10, sticky="nsew")

        # Inicia o loop de simulação para atualizar os sensores
        self.update_all_sensors_periodically()

    def update_all_sensors_periodically(self):
        """Busca novos dados aleatórios e atualiza todos os medidores de sensores."""
        # Obtém novos valores dos sensores
        temp_val = get_random_temperature()
        occ_val = get_random_occupancy()
        hum_val = get_random_humidity()

        # Atualiza os medidores com os novos valores
        self.temp_gauge.update_value(temp_val)
        self.occ_gauge.update_value(occ_val)
        self.hum_gauge.update_value(hum_val)

        # Agenda a próxima atualização
        self.root.after(UPDATE_INTERVAL, self.update_all_sensors_periodically)

# --- Ponto de Entrada Principal: Executa a Aplicação ---
if __name__ == "__main__":
    main_window = tk.Tk()  # Cria a janela principal
    app_instance = SensorSimulatorApp(main_window) # Cria a instância da aplicação
    main_window.mainloop() # Inicia o loop de eventos do Tkinter
