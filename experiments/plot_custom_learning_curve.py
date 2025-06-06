import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3.common.monitor import load_results

# Caminho para a pasta do experimento
LOG_DIR = "logs/ppo/food_delivery_gym-FoodDelivery-medium-obj1-v0_3/"

# Carrega e concatena os dados de todos os arquivos .monitor.csv
def load_all_monitor_files(log_dir):
    dfs = []
    for fname in os.listdir(log_dir):
        if fname.endswith(".monitor.csv"):
            df = pd.read_csv(os.path.join(log_dir, fname), skiprows=1)
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

log_data = load_all_monitor_files(LOG_DIR)

# Extração dos retornos por episódio
retornos = log_data["r"].values

# --- Gráfico 1: Retorno por episódio ---
plt.figure(figsize=(10, 5))
plt.plot(retornos, label="Recompensa")
plt.xlabel("Episódios")
plt.ylabel("Recompensa")
plt.title("Curva de Aprendizado - Recompensa por Episódio")
plt.legend()
plt.savefig(os.path.join(LOG_DIR, "curva_de_aprendizado.png"), dpi=300, bbox_inches='tight')
plt.show()

# --- Gráfico 2: Média e desvio padrão a cada 1000 episódios ---
media_100_episodios = []
desvio_100_episodios = []

for i in range(100, len(retornos), 100):
    bloco = retornos[i-100:i]
    media_100_episodios.append(np.mean(bloco))
    desvio_100_episodios.append(np.std(bloco))

media_100_episodios = np.array(media_100_episodios)
desvio_100_episodios = np.array(desvio_100_episodios)

plt.figure(figsize=(10, 5))
plt.plot(media_100_episodios, label="Média a cada 100 episódios")
plt.fill_between(range(len(media_100_episodios)), media_100_episodios - desvio_100_episodios, media_100_episodios + desvio_100_episodios, alpha=0.2, label="Desvio Padrão")
plt.title("Curva de Aprendizado (média e desvio padrão a cada 100 episódios)")
plt.xlabel("Episódios (x100)")
plt.ylabel("Retorno")
plt.legend()
plt.savefig(os.path.join(LOG_DIR, "curva_de_aprendizado_avg_std_100_ep.png"), dpi=300, bbox_inches='tight')
plt.show()
