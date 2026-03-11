"""
Análise dos logs de treinamento PPO com 4 ambientes paralelos.
Consome os 4 arquivos monitor.csv gerados por n_envs=4.

Uso:
    python analyze_monitor.py --log-folder logs/training/ppo/FoodDelivery-medium-obj11-v0_1
"""

import argparse
import os
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()


def find_monitor_files(folder: str) -> list[str]:
    """Encontra todos os arquivos monitor.csv na pasta do experimento."""
    files = []

    for i in range(4):
        candidate = os.path.join(folder, f"{i}.monitor.csv")
        if os.path.exists(candidate):
            files.append(candidate)

    if not files:
        single = os.path.join(folder, "monitor.csv")
        if os.path.exists(single):
            files.append(single)

    if not files:
        for i in range(4):
            candidate = os.path.join(folder, f"monitor.{i}.csv")
            if os.path.exists(candidate):
                files.append(candidate)

    if not files:
        pattern = os.path.join(folder, "**", "monitor.csv")
        files = glob.glob(pattern, recursive=True)

    return sorted(files)


def load_monitor_csv(filepath: str, env_id: int) -> pd.DataFrame:
    """Carrega um monitor.csv e adiciona metadados."""
    df = pd.read_csv(filepath, skiprows=1)
    df.columns = df.columns.str.strip()

    rename_map = {}
    for col in df.columns:
        if col.lower() == "r":
            rename_map[col] = "reward"
        elif col.lower() == "l":
            rename_map[col] = "length"
        elif col.lower() == "t":
            rename_map[col] = "time"
    df = df.rename(columns=rename_map)

    df["env_id"] = env_id
    df["episode"] = range(1, len(df) + 1)
    df["file"] = os.path.basename(filepath)

    return df


def print_per_env_stats(dfs: list[pd.DataFrame]):
    print("\n" + "=" * 60)
    print("ESTATÍSTICAS POR AMBIENTE")
    print("=" * 60)

    for df in dfs:
        env_id = df["env_id"].iloc[0]
        print(f"\n📦 Env {env_id} — {df['file'].iloc[0]}")
        print(f"   Episódios registrados : {len(df)}")
        print(f"   Reward  → média: {df['reward'].mean():.2f} | "
              f"min: {df['reward'].min():.2f} | "
              f"max: {df['reward'].max():.2f} | "
              f"std: {df['reward'].std():.2f}")
        print(f"   Passos  → média: {df['length'].mean():.1f} | "
              f"min: {df['length'].min()} | "
              f"max: {df['length'].max()} | "
              f"std: {df['length'].std():.1f}")


def print_combined_stats(combined: pd.DataFrame):
    print("\n" + "=" * 60)
    print("ESTATÍSTICAS COMBINADAS (todos os 4 envs)")
    print("=" * 60)
    print(f"Total de episódios : {len(combined)}")
    print(f"\nReward:")
    print(combined["reward"].describe().to_string())
    print(f"\nPassos por episódio (length):")
    print(combined["length"].describe().to_string())


def plot_bar_combo(dfs: list[pd.DataFrame], combined: pd.DataFrame, window: int, no_rolling: bool):
    """
    Gráfico combinado: reward e tamanho do episódio no mesmo gráfico usando barras.
    Reward no eixo Y esquerdo, passos no eixo Y direito (eixo duplo).
    Se --no-rolling não for passado, sobrepõe a média móvel como linhas.
    Gera um gráfico por ambiente.
    """
    sns.set(style="darkgrid")
    colors = sns.color_palette("tab10", n_colors=len(dfs))

    for df, color in zip(dfs, colors):
        env_id = df["env_id"].iloc[0]
        fig, ax1 = plt.subplots(figsize=(14, 5))

        rolling_label = "" if no_rolling else f" + MA({window})"
        fig.suptitle(
            f"Reward × Passos por Episódio — Env {env_id}{rolling_label}",
            fontsize=13, fontweight="bold"
        )

        episodes = df["episode"]
        bar_width = max(1.0, len(episodes) / 500)

        length_color = "#4FC3F7"
        reward_color = "#FF7043"

        # Barras de passos (eixo esquerdo) — plotadas primeiro
        ax1.bar(episodes, df["length"], width=bar_width,
                color=length_color, alpha=0.85, label="Passos")
        ax1.set_xlabel("Episódio")
        ax1.set_ylabel("Passos por Episódio", color=length_color)
        ax1.tick_params(axis="y", labelcolor=length_color)

        # Eixo direito para reward — plotado por cima
        ax2 = ax1.twinx()
        ax2.bar(episodes, df["reward"], width=bar_width,
                color=reward_color, alpha=0.75, label="Reward")
        ax2.set_ylabel("Reward", color=reward_color)
        ax2.tick_params(axis="y", labelcolor=reward_color)

        # Média móvel sobreposta — omitida se --no-rolling
        if not no_rolling and len(df) >= window:
            rolled_reward = df["reward"].rolling(window).mean()
            rolled_length = df["length"].rolling(window).mean()
            ax1.plot(episodes, rolled_length, color="white",
                     linewidth=2, linestyle="--", label=f"Passos MA({window})")
            ax2.plot(episodes, rolled_reward, color="yellow",
                     linewidth=2, label=f"Reward MA({window})")

        # Legenda combinada dos dois eixos
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="upper left")

        plt.tight_layout()
        out_name = f"bar_combo_env{env_id}.png"
        plt.savefig(out_name, dpi=150, bbox_inches="tight")
        print(f"✅ Gráfico salvo em: {out_name}")
        plt.show()


def plot_all(dfs: list[pd.DataFrame], combined: pd.DataFrame, window: int, no_rolling: bool):
    """Gera todos os gráficos de análise."""
    sns.set(style="darkgrid")
    colors = sns.color_palette("tab10", n_colors=len(dfs))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Análise de Treinamento PPO — 4 Ambientes Paralelos", fontsize=14, fontweight="bold")

    title_suffix = "" if no_rolling else f" — Média Móvel ({window} ep)"

    # ── 1. Reward por episódio ─────────────────────────────────────────
    ax = axes[0, 0]
    for df, color in zip(dfs, colors):
        env_id = df["env_id"].iloc[0]
        if no_rolling:
            ax.plot(df["episode"], df["reward"],
                    label=f"Env {env_id}", color=color, linewidth=1, alpha=0.7)
        elif len(df) >= window:
            ax.plot(df["episode"], df["reward"].rolling(window).mean(),
                    label=f"Env {env_id}", color=color, linewidth=1.5)
    ax.set_title(f"Reward{title_suffix}", fontsize=11)
    ax.set_xlabel("Episódio")
    ax.set_ylabel("Reward")
    ax.legend(fontsize=9)

    # ── 2. Passos por episódio ─────────────────────────────────────────
    ax = axes[0, 1]
    for df, color in zip(dfs, colors):
        env_id = df["env_id"].iloc[0]
        if no_rolling:
            ax.plot(df["episode"], df["length"],
                    label=f"Env {env_id}", color=color, linewidth=1, alpha=0.7)
        elif len(df) >= window:
            ax.plot(df["episode"], df["length"].rolling(window).mean(),
                    label=f"Env {env_id}", color=color, linewidth=1.5)
    ax.set_title(f"Passos por Episódio{title_suffix}", fontsize=11)
    ax.set_xlabel("Episódio")
    ax.set_ylabel("Passos")
    ax.legend(fontsize=9)

    # ── 3. Distribuição de reward por env (boxplot) ────────────────────
    ax = axes[1, 0]
    data_box = [df["reward"].values for df in dfs]
    labels = [f"Env {df['env_id'].iloc[0]}" for df in dfs]
    bp = ax.boxplot(data_box, tick_labels=labels, patch_artist=True)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_title("Distribuição de Reward por Env", fontsize=11)
    ax.set_ylabel("Reward")

    # ── 4. Histograma de reward combinado ─────────────────────────────
    ax = axes[1, 1]
    ax.hist(combined["reward"], bins=50, color="steelblue", edgecolor="white", alpha=0.85)
    ax.axvline(combined["reward"].mean(), color="red", linestyle="--", linewidth=1.5,
               label=f"Média: {combined['reward'].mean():.2f}")
    ax.axvline(combined["reward"].median(), color="orange", linestyle="--", linewidth=1.5,
               label=f"Mediana: {combined['reward'].median():.2f}")
    ax.set_title("Histograma de Reward (todos os envs)", fontsize=11)
    ax.set_xlabel("Reward")
    ax.set_ylabel("Frequência")
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig("training_analysis.png", dpi=150, bbox_inches="tight")
    print("\n✅ Gráfico salvo em: training_analysis.png")
    plt.show()


def export_csv(combined: pd.DataFrame):
    out = "all_episodes.csv"
    combined[["env_id", "episode", "reward", "length", "time"]].to_csv(out, index=False)
    print(f"✅ CSV exportado em: {out}")


def main():
    parser = argparse.ArgumentParser(description="Análise dos monitor.csv de treinamento PPO (n_envs=4)")
    parser.add_argument("--log-folder", "-f", type=str, required=True,
                        help="Pasta do experimento, ex: logs/training/ppo/FoodDelivery-medium-obj11-v0_1")
    parser.add_argument("--window", "-w", type=int, default=100,
                        help="Janela da média móvel (default: 100)")
    parser.add_argument("--export", action="store_true",
                        help="Exporta os dados combinados para all_episodes.csv")
    parser.add_argument("--no-plot", action="store_true",
                        help="Pula geração de gráficos")
    parser.add_argument("--bar-combo", action="store_true",
                        help="Exibe reward e tamanho do episódio no mesmo gráfico usando barras "
                             "(eixo Y duplo, um gráfico por ambiente)")
    parser.add_argument("--no-rolling", action="store_true",
                        help="Remove a média móvel dos gráficos e exibe os valores brutos")

    args = parser.parse_args()

    files = find_monitor_files(args.log_folder)

    if not files:
        print(f"❌ Nenhum arquivo monitor.csv encontrado em: {args.log_folder}")
        print("   Verifique o caminho ou a estrutura de pastas.")
        return

    print(f"📂 Arquivos encontrados ({len(files)}):")
    for f in files:
        print(f"   {f}")

    dfs = []
    for i, filepath in enumerate(files):
        try:
            df = load_monitor_csv(filepath, env_id=i)
            dfs.append(df)
            print(f"   ✓ Env {i}: {len(df)} episódios carregados")
        except Exception as e:
            print(f"   ✗ Erro ao carregar {filepath}: {e}")

    if not dfs:
        print("❌ Nenhum dado pôde ser carregado.")
        return

    combined = pd.concat(dfs, ignore_index=True)

    print_per_env_stats(dfs)
    print_combined_stats(combined)

    if args.export:
        export_csv(combined)

    if not args.no_plot:
        if args.bar_combo:
            plot_bar_combo(dfs, combined, window=args.window, no_rolling=args.no_rolling)
        else:
            plot_all(dfs, combined, window=args.window, no_rolling=args.no_rolling)


if __name__ == "__main__":
    main()