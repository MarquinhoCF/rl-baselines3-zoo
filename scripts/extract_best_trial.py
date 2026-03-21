"""
extract_best_trial.py

Analisa os trials de uma otimização de hiperparâmetros (salvos localmente),
identifica o melhor trial e gera/atualiza o arquivo ppo.yml com os
hiperparâmetros otimizados.

A pasta de trials é sempre <exp-dir>/optimization/, onde <exp-dir> é o
diretório do experimento gerado pelo rl-baselines-zoo3 (ex: logs/ppo/CartPole-v1_1/).

A normalização é detectada automaticamente a partir dos arquivos presentes
no diretório do experimento (config.yml, vecnormalize.pkl, etc.).

Uso mínimo:
    python extract_best_trial.py \
        --exp-dir logs/ppo/FoodDelivery-medium-obj1-v1_3

Uso completo:
    python extract_best_trial.py \
        --exp-dir logs/ppo/FoodDelivery-medium-obj1-v1_3 \
        --env FoodDelivery-medium-obj1-v1 \
        --n-timesteps 18000000 \
        --n-envs 4 \
        --output-dir hyperparams/best_params_for_food_delivery_gym
"""

import argparse
import importlib.metadata
import json
import os
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


# ── Estrutura de resultado ────────────────────────────────────────────────────

@dataclass
class LocalTrialResult:
    """Representa o melhor trial identificado a partir dos arquivos locais."""
    number: int
    best_mean_reward: float
    last_timestep: int
    num_evaluations: int
    num_valid_trials: int = 0


# ── Detecção de normalização ──────────────────────────────────────────────────

def detect_normalization(exp_dir: Path) -> bool:
    """
    Detecta automaticamente se a otimização foi feita com normalização.

    O ExperimentManager salva um arquivo report_*.pkl na raiz do exp_dir
    ao final de hyperparameters_optimization(). A presença desse arquivo
    indica que a otimização foi concluída; o nome do arquivo carrega os
    metadados do estudo (env, n_trials, n_timesteps, sampler, pruner).

    Convenção adotada: se existir algum *.pkl na raiz do exp_dir,
    considera-se que a otimização usou normalização.
    """
    pkl_files = list(exp_dir.glob("*.pkl"))
    if pkl_files:
        print(f"[Normalização] Arquivo de report detectado: {pkl_files[0].name} → True")
        return True
    print("[Normalização] Nenhum report *.pkl encontrado na raiz do exp_dir → False")
    return False


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_food_delivery_gym_version() -> str:
    """Lê a versão instalada do pacote food_delivery_gym."""
    try:
        return importlib.metadata.version("food_delivery_gym")
    except importlib.metadata.PackageNotFoundError:
        return "0.0.0"


def version_to_dir(version: str) -> str:
    """
    Converte a versão semântica no nome do subdiretório.

    Exemplos:
        0.0.3  → v0.0.x
        1.1.5  → v1.1.x
    """
    parts = version.split(".")
    if len(parts) >= 2:
        return f"v{parts[0]}.{parts[1]}.x"
    return f"v{parts[0]}.x"


def infer_env_id(exp_dir: Path) -> str:
    """
    Infere o env_id a partir do nome do diretório do experimento.

    O rl-baselines-zoo3 gera pastas no formato  <env_name>_<run_id>[_<uuid>].
    Remove o sufixo numérico (e UUID opcional) para obter o env_name.

    Exemplos:
        FoodDelivery-medium-obj1-v1_3         → FoodDelivery-medium-obj1-v1
        FoodDelivery-medium-obj1-v1_3_abc123  → FoodDelivery-medium-obj1-v1
        CartPole-v1_1                         → CartPole-v1
    """
    name = exp_dir.name
    # Remove sufixo _<número> e possível _<uuid> no final
    match = re.match(r"^(.+?)_\d+(_[0-9a-f\-]+)?$", name)
    if match:
        return match.group(1)
    return name


def build_comment(n_timesteps_per_trial: int, num_trials: int, normalize: bool) -> str:
    """Gera o comentário automaticamente a partir dos dados disponíveis."""
    parts = []
    if normalize:
        parts.append("com normalização")
    ts_fmt = f"{n_timesteps_per_trial:,}".replace(",", ".")
    parts.append(f"{ts_fmt} de passos por trial")
    return " e ".join(parts)


def load_model_data_from_zip(zip_path: str) -> dict:
    """Extrai o arquivo 'data' de um best_model.zip e retorna como dict."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        if "data" not in zf.namelist():
            raise FileNotFoundError(f"Arquivo 'data' não encontrado em {zip_path}")
        with zf.open("data") as f:
            return json.loads(f.read().decode("utf-8"))


def extract_hyperparams_from_model_data(data: dict) -> dict:
    """Extrai os hiperparâmetros relevantes do dict de dados do modelo SB3."""
    hyperparams = {}

    for key in ["learning_rate", "ent_coef", "vf_coef", "max_grad_norm",
                "gamma", "gae_lambda", "n_steps", "n_epochs", "n_envs",
                "num_timesteps"]:
        if key in data:
            hyperparams[key] = data[key]

    if "batch_size" in data:
        hyperparams["batch_size"] = data["batch_size"]

    # clip_range: pode ser objeto serializado com value_schedule
    clip_range_raw = data.get("clip_range")
    if isinstance(clip_range_raw, dict):
        vs = clip_range_raw.get("value_schedule", "")
        match = re.search(r"val=([\d.]+)", vs)
        if match:
            hyperparams["clip_range"] = float(match.group(1))
    elif isinstance(clip_range_raw, (int, float)):
        hyperparams["clip_range"] = clip_range_raw

    # policy_kwargs: net_arch e activation_fn
    policy_kwargs_raw = data.get("policy_kwargs", {})
    if isinstance(policy_kwargs_raw, dict):
        net_arch = policy_kwargs_raw.get("net_arch")
        activation_fn = policy_kwargs_raw.get("activation_fn", "")

        if isinstance(activation_fn, str):
            fn_lower = activation_fn.lower()
            if "leakyrelu" in fn_lower or "leaky_relu" in fn_lower:
                activation_fn_name = "nn.LeakyReLU"
            elif "relu" in fn_lower:
                activation_fn_name = "nn.ReLU"
            elif "tanh" in fn_lower:
                activation_fn_name = "nn.Tanh"
            elif "elu" in fn_lower:
                activation_fn_name = "nn.ELU"
            else:
                activation_fn_name = activation_fn
        else:
            activation_fn_name = str(activation_fn)

        if net_arch:
            hyperparams["net_arch"] = net_arch
            hyperparams["activation_fn_name"] = activation_fn_name

    return hyperparams


def format_net_arch(net_arch) -> str:
    """Formata net_arch como string Python para o YAML."""
    if isinstance(net_arch, dict):
        pi = net_arch.get("pi", [])
        vf = net_arch.get("vf", [])
        return f"dict(pi={pi}, vf={vf})"
    if isinstance(net_arch, list):
        return str(net_arch)
    return str(net_arch)


def format_value(value) -> str:
    """Formata um valor escalar para representação YAML legível."""
    if isinstance(value, float):
        return repr(value)
    return str(value)


def build_yaml_block(
    env_id: str,
    hyperparams: dict,
    n_timesteps: int,
    n_envs: int,
    normalize: bool,
    comment: str,
    num_trials: int,
    best_trial_number: int,
) -> str:
    """Constrói o bloco YAML para os hiperparâmetros do melhor trial."""
    net_arch = hyperparams.get("net_arch")
    activation_fn_name = hyperparams.get("activation_fn_name", "nn.Tanh")
    objective = _extract_objective(env_id)

    lines = [
        f"# Otimização de hiperparâmetros — objetivo {objective}, "
        f"{comment}, {num_trials} trials (melhor: trial_{best_trial_number})",
        f"{env_id}:",
        f"  n_timesteps: {n_timesteps}",
        f"  policy: 'MultiInputPolicy'",
        f"  n_envs: {n_envs}",
        f"  learning_rate: {format_value(hyperparams.get('learning_rate', 3e-4))}",
        f"  ent_coef: {format_value(hyperparams.get('ent_coef', 0.0))}",
        f"  clip_range: {format_value(hyperparams.get('clip_range', 0.2))}",
        f"  n_steps: {hyperparams.get('n_steps', 2048)}",
        f"  batch_size: {hyperparams.get('batch_size', 64)}",
        f"  n_epochs: {hyperparams.get('n_epochs', 10)}",
        f"  vf_coef: {format_value(hyperparams.get('vf_coef', 0.5))}",
        f"  gamma: {format_value(hyperparams.get('gamma', 0.99))}",
        f"  gae_lambda: {format_value(hyperparams.get('gae_lambda', 0.95))}",
        f"  max_grad_norm: {format_value(hyperparams.get('max_grad_norm', 0.5))}",
    ]

    if net_arch:
        arch_str = format_net_arch(net_arch)
        lines.append(
            f"  policy_kwargs: \"dict(net_arch={arch_str}, activation_fn={activation_fn_name})\""
        )

    if normalize:
        lines.append("  normalize: true")

    return "\n".join(lines)


def _extract_objective(env_id: str) -> str:
    """Extrai o número do objetivo do env_id. Ex: '...obj3-v0' → '3'."""
    match = re.search(r"obj(\d+)", env_id)
    return match.group(1) if match else "?"


# ── Lógica principal ──────────────────────────────────────────────────────────

def scan_trials_locally(trials_dir: Path) -> LocalTrialResult:
    """
    Varre todos os trial_N/evaluations.npz e retorna o melhor trial.

    Critério: recompensa média do ÚLTIMO checkpoint de cada trial (results[-1]),
    que espelha exatamente o valor retornado por objective() ao Optuna via
    eval_callback.last_mean_reward.
    """
    trial_dirs = sorted(
        [d for d in trials_dir.iterdir() if d.is_dir() and re.match(r"trial_\d+$", d.name)],
        key=lambda d: int(d.name.split("_")[1]),
    )

    if not trial_dirs:
        raise FileNotFoundError(
            f"Nenhum diretório trial_N encontrado em '{trials_dir}'."
        )

    print(f"\n[Local] Varrendo {len(trial_dirs)} trials em '{trials_dir}'...")

    best_number = -1
    best_reward = float("-inf")
    best_last_timestep = 0
    best_num_evals = 0
    valid_count = 0
    skipped: list[str] = []

    for trial_dir in trial_dirs:
        npz_path = trial_dir / "evaluations.npz"
        if not npz_path.is_file():
            skipped.append(trial_dir.name)
            continue

        try:
            data = np.load(str(npz_path))
            results = data["results"]      # shape: (n_evals, n_episodes)
            timesteps = data["timesteps"]  # shape: (n_evals,)

            last_mean = float(np.mean(results[-1]))
            last_timestep = int(timesteps[-1])
            num_evals = len(timesteps)
            valid_count += 1

            trial_number = int(trial_dir.name.split("_")[1])

            if last_mean > best_reward:
                best_reward = last_mean
                best_number = trial_number
                best_last_timestep = last_timestep
                best_num_evals = num_evals

        except Exception as e:
            skipped.append(f"{trial_dir.name} (erro: {e})")

    if skipped:
        preview = ", ".join(skipped[:10]) + (" ..." if len(skipped) > 10 else "")
        print(f"[Local] Trials ignorados ({len(skipped)}): {preview}")

    print(f"[Local] Trials válidos analisados : {valid_count}")

    if best_number == -1:
        raise RuntimeError("Nenhum evaluations.npz válido encontrado nos trials.")

    print(
        f"[Local] Melhor trial              : trial_{best_number}  "
        f"(última recompensa média={best_reward:.6f},  timestep={best_last_timestep})"
    )

    return LocalTrialResult(
        number=best_number,
        best_mean_reward=best_reward,
        last_timestep=best_last_timestep,
        num_evaluations=best_num_evals,
        num_valid_trials=valid_count,
    )


def find_best_model_zip(trials_dir: Path, trial_number: int) -> Optional[Path]:
    """Procura o best_model.zip no diretório do trial."""
    candidate = trials_dir / f"trial_{trial_number}" / "best_model.zip"
    return candidate if candidate.is_file() else None


def print_trial_summary(trial: LocalTrialResult, hyperparams: dict, normalize: bool):
    """Exibe resumo do melhor trial no terminal."""
    sep = "─" * 60
    print(f"\n{'═' * 60}")
    print(f"  MELHOR TRIAL : trial_{trial.number}")
    print(f"{'═' * 60}")
    print(f"  Última recompensa média : {trial.best_mean_reward:.6f}")
    print(f"  Último timestep         : {trial.last_timestep}")
    print(f"  Número de avaliações    : {trial.num_evaluations}")
    print(f"  Normalização detectada  : {normalize}")
    print(f"\n{sep}")
    print("  HIPERPARÂMETROS EXTRAÍDOS DO best_model.zip")
    print(sep)
    for k, v in sorted(hyperparams.items()):
        print(f"    {k:35s} = {v}")
    print()


def append_yaml(output_path: Path, yaml_block: str):
    """Adiciona o bloco YAML ao arquivo sem sobrescrever entradas existentes."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    existing = output_path.read_text(encoding="utf-8") if output_path.is_file() else ""
    separator = "\n\n" if existing.strip() else ""
    output_path.write_text(existing + separator + yaml_block + "\n", encoding="utf-8")
    print(f"[YAML] Hiperparâmetros adicionados em: {output_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Extrai o melhor trial de uma otimização local e gera/atualiza o ppo.yml.\n\n"
            "Passa-se o diretório do experimento (<env_name>_<id>) gerado pelo\n"
            "rl-baselines-zoo3. A pasta optimization/ é localizada automaticamente\n"
            "dentro dele, assim como a detecção de normalização.\n\n"
            "Exemplo:\n"
            "  python extract_best_trial.py \\\n"
            "      --exp-dir logs/ppo/FoodDelivery-medium-obj1-v1_3"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "--exp-dir", "-e",
        required=True,
        help=(
            "Diretório do experimento gerado pelo rl-baselines-zoo3.\n"
            "Formato esperado: <log_folder>/<algo>/<env_name>_<id>\n"
            "Exemplo: logs/ppo/FoodDelivery-medium-obj1-v1_3"
        ),
    )
    parser.add_argument(
        "--env",
        default=None,
        help=(
            "ID do ambiente Gymnasium (ex: FoodDelivery-medium-obj1-v1).\n"
            "Se omitido, é inferido automaticamente a partir do nome do --exp-dir."
        ),
    )
    parser.add_argument(
        "--n-timesteps",
        type=int,
        default=18_000_000,
        help="Timesteps para o treinamento final (padrão: 18000000).",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=4,
        help="Número de ambientes paralelos (padrão: 4).",
    )
    parser.add_argument(
        "--output-dir",
        default="hyperparams/best_params_for_food_delivery_gym",
        help=(
            "Diretório base de saída. O subdiretório de versão é criado automaticamente.\n"
            "Padrão: hyperparams/best_params_for_food_delivery_gym"
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    exp_dir = Path(args.exp_dir).resolve()
    if not exp_dir.is_dir():
        raise SystemExit(f"[Erro] Diretório do experimento não encontrado: {exp_dir}")

    # Localiza a pasta optimization/
    trials_dir = exp_dir / "optimization"
    if not trials_dir.is_dir():
        raise SystemExit(
            f"[Erro] Pasta 'optimization/' não encontrada dentro de '{exp_dir}'.\n"
            "       Verifique se a otimização de hiperparâmetros foi executada neste diretório."
        )

    # Infere env_id
    env_id = args.env or infer_env_id(exp_dir)

    # Versão do pacote → subdiretório de saída
    version = get_food_delivery_gym_version()
    version_dir = version_to_dir(version)
    output_yaml = Path(args.output_dir) / version_dir / "ppo.yml"

    print(f"[Info] Diretório do experimento     : {exp_dir}")
    print(f"[Info] Pasta de trials              : {trials_dir}")
    print(f"[Info] env_id                       : {env_id}")
    print(f"[Info] Versão do food_delivery_gym  : {version}")
    print(f"[Info] Arquivo YAML de destino      : {output_yaml}")

    # Detecta normalização automaticamente
    normalize = detect_normalization(exp_dir)

    # Identifica o melhor trial
    best_trial = scan_trials_locally(trials_dir)
    num_trials = best_trial.num_valid_trials
    n_timesteps_per_trial = best_trial.last_timestep

    # Extrai hiperparâmetros do best_model.zip
    zip_path = find_best_model_zip(trials_dir, best_trial.number)
    if zip_path is None:
        raise FileNotFoundError(
            f"best_model.zip não encontrado em '{trials_dir}/trial_{best_trial.number}/'.\n"
            "Certifique-se de que o TrialEvalCallback salvou o melhor modelo."
        )

    print(f"[Info] Lendo hiperparâmetros de     : {zip_path}")
    model_data = load_model_data_from_zip(str(zip_path))
    hyperparams = extract_hyperparams_from_model_data(model_data)

    # Usa num_timesteps real do modelo como referência para o comentário
    if "num_timesteps" in hyperparams:
        n_timesteps_per_trial = hyperparams.pop("num_timesteps")

    # Gera o comentário
    comment = build_comment(n_timesteps_per_trial, num_trials, normalize)

    # Exibe resumo
    print_trial_summary(best_trial, hyperparams, normalize)

    # Gera o bloco YAML
    yaml_block = build_yaml_block(
        env_id=env_id,
        hyperparams=hyperparams,
        n_timesteps=args.n_timesteps,
        n_envs=args.n_envs,
        normalize=normalize,
        comment=comment,
        num_trials=num_trials,
        best_trial_number=best_trial.number,
    )

    print("─" * 60)
    print("BLOCO YAML GERADO:")
    print("─" * 60)
    print(yaml_block)
    print("─" * 60)

    # Adiciona ao ppo.yml sem sobrescrever
    append_yaml(output_yaml, yaml_block)
    print(f"\n[OK] Concluído.")


if __name__ == "__main__":
    main()