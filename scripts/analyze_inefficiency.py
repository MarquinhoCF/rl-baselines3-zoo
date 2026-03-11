"""
Análise de ineficiência de política PPO: steps - reward por episódio.

A métrica `steps - reward` representa o "desperdício" da política:
  - 0  → todos os pedidos entregues sem passos extras (política ótima)
  - >0 → quanto maior, pior a política (passos gastos além do necessário)

Uso:
    python analyze_inefficiency.py --log-folder logs/training/ppo/FoodDelivery-medium-obj11-v0_1
"""

import argparse
import os
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ──────────────────────────────────────────────────────────────────────────────
# I/O helpers (reutilizados do analyze_monitor.py)
# ──────────────────────────────────────────────────────────────────────────────

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

    df["env_id"]  = env_id
    df["episode"] = range(1, len(df) + 1)
    df["file"]    = os.path.basename(filepath)

    return df


# ──────────────────────────────────────────────────────────────────────────────
# Métrica de ineficiência
# ──────────────────────────────────────────────────────────────────────────────

def add_inefficiency(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona coluna `inefficiency = length - reward`."""
    df = df.copy()
    df["inefficiency"] = df["length"] - df["reward"]
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Estatísticas
# ──────────────────────────────────────────────────────────────────────────────

def print_stats(dfs: list[pd.DataFrame], combined: pd.DataFrame):
    print("\n" + "=" * 60)
    print("INEFICIÊNCIA (steps - reward) POR AMBIENTE")
    print("=" * 60)

    for df in dfs:
        env_id = df["env_id"].iloc[0]
        col    = df["inefficiency"]
        zeros  = (col == 0).sum()
        pct    = zeros / len(df) * 100
        print(f"\n📦 Env {env_id} — {df['file'].iloc[0]}")
        print(f"   Episódios     : {len(df)}")
        print(f"   Episódios com ineficiência = 0 (ótimos): {zeros} ({pct:.1f} %)")
        print(f"   Ineficiência → média: {col.mean():.2f} | "
              f"min: {col.min():.2f} | "
              f"max: {col.max():.2f} | "
              f"std: {col.std():.2f}")

    print("\n" + "=" * 60)
    print("COMBINADO (todos os envs)")
    print("=" * 60)
    col    = combined["inefficiency"]
    zeros  = (col == 0).sum()
    pct    = zeros / len(combined) * 100
    print(f"Total episódios               : {len(combined)}")
    print(f"Episódios ótimos (inef. = 0)  : {zeros} ({pct:.1f} %)")
    print(f"\nIneficiência:")
    print(col.describe().to_string())


# ──────────────────────────────────────────────────────────────────────────────
# Gráfico
# ──────────────────────────────────────────────────────────────────────────────

def plot_inefficiency(dfs: list[pd.DataFrame], combined: pd.DataFrame,
                      window: int, no_rolling: bool, title: str | None = None):
    """
    Gráfico de linhas: ineficiência (steps - reward) por episódio.

    - Uma linha por ambiente
    - Linha de referência em y = 0 (política ótima)
    - Média móvel opcional sobreposta
    """
    sns.set(style="darkgrid")
    colors = sns.color_palette("tab10", n_colors=len(dfs))

    fig, ax = plt.subplots(figsize=(14, 6))

    rolling_label = f" + MA({window})" if not no_rolling else ""
    default_title = f"Ineficiência da Política (steps − reward){rolling_label}"
    fig.suptitle(
        f"{title}\n{default_title}" if title else default_title,
        fontsize=13, fontweight="bold"
    )

    for df, color in zip(dfs, colors):
        env_id = df["env_id"].iloc[0]

        if no_rolling:
            ax.plot(df["episode"], df["inefficiency"],
                    label=f"Env {env_id}", color=color,
                    linewidth=1, alpha=0.6)
        else:
            # Sinal bruto em transparência
            ax.plot(df["episode"], df["inefficiency"],
                    color=color, linewidth=0.6, alpha=0.25)

            # Média móvel em destaque
            if len(df) >= window:
                rolled = df["inefficiency"].rolling(window).mean()
                ax.plot(df["episode"], rolled,
                        label=f"Env {env_id} MA({window})",
                        color=color, linewidth=2)

    # Linha de referência y = 0
    ax.axhline(0, color="white", linestyle="--", linewidth=1.5,
               label="Referência ótima (inef. = 0)", zorder=5)

    ax.set_xlabel("Episódio")
    ax.set_ylabel("Ineficiência (steps − reward)")
    ax.legend(fontsize=9, loc="upper right")

    plt.tight_layout()
    out_name = "inefficiency.png"
    plt.savefig(out_name, dpi=150, bbox_inches="tight")
    print(f"\n✅ Gráfico salvo em: {out_name}")
    plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# Exportação
# ──────────────────────────────────────────────────────────────────────────────

def export_csv(combined: pd.DataFrame):
    out = "inefficiency_episodes.csv"
    combined[["env_id", "episode", "reward", "length", "inefficiency", "time"]].to_csv(
        out, index=False
    )
    print(f"✅ CSV exportado em: {out}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Análise de ineficiência (steps - reward) dos monitor.csv de treinamento PPO"
    )
    parser.add_argument("--log-folder", "-f", type=str, required=True,
                        help="Pasta do experimento, ex: logs/training/ppo/FoodDelivery-medium-obj11-v0_1")
    parser.add_argument("--window", "-w", type=int, default=100,
                        help="Janela da média móvel (default: 100)")
    parser.add_argument("--export", action="store_true",
                        help="Exporta os dados para inefficiency_episodes.csv")
    parser.add_argument("--no-plot", action="store_true",
                        help="Pula a geração do gráfico")
    parser.add_argument("--no-rolling", action="store_true",
                        help="Exibe os valores brutos sem média móvel")
    parser.add_argument("--title", "-t", type=str, default=None,
                        help="Título principal do gráfico, ex: 'Experimento v3 — lr=3e-4'")

    args = parser.parse_args()

    # ── Carregamento ──────────────────────────────────────────────────
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
            df = add_inefficiency(df)
            dfs.append(df)
            print(f"   ✓ Env {i}: {len(df)} episódios carregados")
        except Exception as e:
            print(f"   ✗ Erro ao carregar {filepath}: {e}")

    if not dfs:
        print("❌ Nenhum dado pôde ser carregado.")
        return

    combined = pd.concat(dfs, ignore_index=True)

    # ── Análise ───────────────────────────────────────────────────────
    print_stats(dfs, combined)

    if args.export:
        export_csv(combined)

    if not args.no_plot:
        plot_inefficiency(dfs, combined, window=args.window, no_rolling=args.no_rolling, title=args.title)


if __name__ == "__main__":
    main()