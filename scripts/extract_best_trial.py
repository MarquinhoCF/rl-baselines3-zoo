"""
extract_best_trial.py

Analisa os trials de uma otimização de hiperparâmetros do Optuna, identifica
o melhor trial e gera/atualiza o arquivo ppo.yml com os hiperparâmetros otimizados.

Modos de operação:
  - Com SQLite  (--storage + --study-name): usa o Optuna para identificar o melhor
    trial com precisão total, incluindo metadados de tempo e parâmetros amostrados.
    O --trials-dir é opcional (necessário apenas para ler o best_model.zip).

  - Sem SQLite  (--trials-dir obrigatório): varre todos os trial_N/evaluations.npz
    localmente e identifica o melhor pela recompensa média do último checkpoint,
    espelhando o critério do Optuna (eval_callback.last_mean_reward).

O comentário do bloco YAML é gerado automaticamente a partir dos dados disponíveis.

Uso mínimo com SQLite:
    python extract_best_trial.py \
        --storage sqlite:///optuna_studies.db \
        --study-name meu_estudo \
        --env food_delivery_gym/FoodDelivery-medium-obj1-v0

Uso mínimo sem SQLite:
    python extract_best_trial.py \
        --trials-dir logs/hyperparam_opt_ppo_food_delivery_medium_obj1 \
        --env food_delivery_gym/FoodDelivery-medium-obj1-v0

Uso completo:
    python extract_best_trial.py \
        --trials-dir logs/hyperparam_opt_ppo_food_delivery_medium_obj1 \
        --storage sqlite:///optuna_studies.db \
        --study-name meu_estudo \
        --env food_delivery_gym/FoodDelivery-medium-obj1-v0 \
        --n-timesteps 18000000 \
        --n-envs 4 \
        --normalize \
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


# ── Estrutura para resultado local (sem SQLite) ───────────────────────────────

@dataclass
class LocalTrialResult:
    """Representa o melhor trial identificado apenas por arquivos locais."""
    number: int
    best_mean_reward: float
    last_timestep: int
    num_evaluations: int
    num_valid_trials: int = 0
    # Campos opcionais presentes apenas no modo Optuna
    params: dict = field(default_factory=dict)
    user_attrs: dict = field(default_factory=dict)
    datetime_start: Optional[object] = None
    datetime_complete: Optional[object] = None
    state_name: str = "COMPLETE"


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
        2.0.1  → v2.0.x
    """
    parts = version.split(".")
    if len(parts) >= 2:
        return f"v{parts[0]}.{parts[1]}.x"
    return f"v{parts[0]}.x"



def build_comment(n_timesteps_per_trial: int, num_trials: int, normalize: bool) -> str:
    """
    Gera o comentário automaticamente a partir dos dados disponíveis.

    Exemplo: "com normalização e 1.000.000 de passos por trial"
    """
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
            if "relu" in fn_lower and "leaky" not in fn_lower:
                activation_fn_name = "nn.ReLU"
            elif "tanh" in fn_lower:
                activation_fn_name = "nn.Tanh"
            elif "elu" in fn_lower and "leaky" not in fn_lower:
                activation_fn_name = "nn.ELU"
            elif "leakyrelu" in fn_lower or "leaky_relu" in fn_lower:
                activation_fn_name = "nn.LeakyReLU"
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
        lines.append(f"  policy_kwargs: \"dict(net_arch={arch_str}, activation_fn={activation_fn_name})\"")

    if normalize:
        lines.append("  normalize: true")

    return "\n".join(lines)


def _extract_objective(env_id: str) -> str:
    """Extrai o número do objetivo do env_id. Ex: '...obj3-v0' → '3'."""
    match = re.search(r"obj(\d+)", env_id)
    return match.group(1) if match else "?"


# ── Lógica principal ──────────────────────────────────────────────────────────

def load_study(storage: str, study_name: str):
    """Carrega o estudo Optuna do banco SQLite. Importa optuna sob demanda."""
    import optuna as _optuna
    print(f"\n[Optuna] Carregando estudo '{study_name}' de '{storage}'...")
    return _optuna.load_study(study_name=study_name, storage=storage)


def scan_trials_locally(trials_dir: str) -> LocalTrialResult:
    """
    Varre todos os trial_N/evaluations.npz e retorna o melhor trial.

    Critério de comparação: recompensa média do ÚLTIMO checkpoint de cada trial
    (results[-1]), que é exatamente o valor que o Optuna recebe via
    eval_callback.last_mean_reward no retorno de objective().
    Isso garante que o modo local seleciona o mesmo trial que o Optuna
    selecionaria como best_trial.

    Nota: trials podados pelo Optuna terminam antes de 1.000.000 de steps,
    portanto terão menos checkpoints — results[-1] será o último avaliado
    antes do pruning, assim como ocorre no Optuna.
    """
    trials_path = Path(trials_dir)
    trial_dirs = sorted(
        [d for d in trials_path.iterdir() if d.is_dir() and re.match(r"trial_\d+$", d.name)],
        key=lambda d: int(d.name.split("_")[1]),
    )

    if not trial_dirs:
        raise FileNotFoundError(f"Nenhum diretório trial_N encontrado em '{trials_dir}'.")

    print(f"\n[Local] Varrendo {len(trial_dirs)} trials em '{trials_dir}'...")

    best_number = -1
    best_reward = float("-inf")
    best_last_timestep = 0
    best_num_evals = 0
    valid_count = 0
    skipped = []

    for trial_dir in trial_dirs:
        npz_path = trial_dir / "evaluations.npz"
        if not npz_path.is_file():
            skipped.append(trial_dir.name)
            continue

        try:
            data = np.load(str(npz_path))
            results = data["results"]       # shape: (n_evals, n_episodes)
            timesteps = data["timesteps"]   # shape: (n_evals,)

            # Recompensa média do último checkpoint — espelha eval_callback.last_mean_reward
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
        print(f"[Local] Trials ignorados ({len(skipped)}): {', '.join(skipped[:10])}"
              + (" ..." if len(skipped) > 10 else ""))

    print(f"[Local] Trials válidos analisados : {valid_count}")

    if best_number == -1:
        raise RuntimeError("Nenhum evaluations.npz válido encontrado nos trials.")

    print(f"[Local] Melhor trial              : trial_{best_number}  "
          f"(recompensa média no último checkpoint={best_reward:.6f}  "
          f"timestep={best_last_timestep})")

    return LocalTrialResult(
        number=best_number,
        best_mean_reward=best_reward,
        last_timestep=best_last_timestep,
        num_evaluations=best_num_evals,
        num_valid_trials=valid_count,
    )


def find_best_model_zip(trials_dir: str, trial_number: int) -> Optional[str]:
    """Procura o best_model.zip no diretório do trial."""
    trial_path = Path(trials_dir) / f"trial_{trial_number}"
    zip_path = trial_path / "best_model.zip"
    if zip_path.is_file():
        return str(zip_path)
    return None


def print_trial_summary(trial, hyperparams: dict):
    """Exibe resumo do melhor trial. Aceita FrozenTrial (Optuna) ou LocalTrialResult."""
    sep = "─" * 60
    is_local = isinstance(trial, LocalTrialResult)

    print(f"\n{'═' * 60}")
    print(f"  MELHOR TRIAL: trial_{trial.number}")
    print(f"{'═' * 60}")

    if is_local:
        print(f"  Modo                                : LOCAL (sem SQLite)")
        print(f"  Recompensa média (último checkpoint): {trial.best_mean_reward:.6f}")
        print(f"  Último timestep avaliado            : {trial.last_timestep}")
        print(f"  Número de avaliações                : {trial.num_evaluations}")
    else:
        print(f"  Modo    : OPTUNA (SQLite)")
        print(f"  Valor   : {trial.value:.6f}")
        print(f"  Estado  : {trial.state.name}")
        print(f"  Início  : {trial.datetime_start}")
        print(f"  Fim     : {trial.datetime_complete}")
        duration = (
            (trial.datetime_complete - trial.datetime_start)
            if trial.datetime_complete else None
        )
        if duration:
            print(f"  Duração : {duration}")

        if trial.params:
            print(f"\n{sep}")
            print("  PARÂMETROS AMOSTRADOS (Optuna)")
            print(sep)
            for k, v in sorted(trial.params.items()):
                print(f"    {k:35s} = {v}")

        if trial.user_attrs:
            print(f"\n{sep}")
            print("  ATRIBUTOS DE USUÁRIO (valores reais)")
            print(sep)
            for k, v in sorted(trial.user_attrs.items()):
                print(f"    {k:35s} = {v}")

    print(f"\n{sep}")
    print("  HIPERPARÂMETROS EXTRAÍDOS DO best_model.zip")
    print(sep)
    for k, v in sorted(hyperparams.items()):
        print(f"    {k:35s} = {v}")
    print()


def append_yaml(output_path: str, yaml_block: str):
    """Adiciona o bloco YAML ao arquivo, sem sobrescrever entradas existentes."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    existing = path.read_text(encoding="utf-8") if path.is_file() else ""

    separator = "\n\n" if existing.strip() else ""
    new_content = existing + separator + yaml_block + "\n"

    path.write_text(new_content, encoding="utf-8")
    print(f"[YAML] Hiperparâmetros adicionados em: {output_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Extrai o melhor trial de uma otimização Optuna e gera/atualiza o ppo.yml.\n\n"
            "Modos (mutuamente exclusivos):\n"
            "  Com SQLite  : --storage + --study-name (--trials-dir opcional)\n"
            "  Sem SQLite  : --trials-dir + --env (obrigatórios)\n\n"
            "O comentário do YAML é gerado automaticamente a partir dos dados disponíveis."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # Modo SQLite
    sqlite_group = parser.add_argument_group("Modo SQLite (--storage + --study-name)")
    sqlite_group.add_argument(
        "--storage", "-s",
        default=None,
        help="URI do banco Optuna (ex: sqlite:///optuna_studies.db).",
    )
    sqlite_group.add_argument(
        "--study-name",
        default=None,
        help=(
            "Nome do estudo no banco Optuna (ex: ppo_medium_obj1).\n"
            "Nome do estudo no banco Optuna (ex: meu_estudo_ppo_medium)."
        ),
    )

    # Modo local
    local_group = parser.add_argument_group("Modo local (sem SQLite)")
    local_group.add_argument(
        "--trials-dir", "-t",
        default=None,
        help=(
            "Diretório com as pastas trial_N.\n"
            "Obrigatório no modo local. No modo SQLite, usado para ler o best_model.zip\n"
            "(opcional — sem ele os hiperparâmetros são inferidos dos params Optuna)."
        ),
    )
    local_group.add_argument(
        "--env",
        required=True,
        help="ID completo do ambiente Gymnasium (ex: food_delivery_gym/FoodDelivery-medium-obj1-v0).",
    )

    # Parâmetros de saída
    out_group = parser.add_argument_group("Parâmetros de saída (opcionais)")
    out_group.add_argument(
        "--n-timesteps",
        type=int,
        default=18_000_000,
        help="Timesteps para o treinamento final (padrão: 18000000).",
    )
    out_group.add_argument(
        "--n-envs",
        type=int,
        default=4,
        help="Número de ambientes paralelos (padrão: 4).",
    )
    out_group.add_argument(
        "--normalize",
        action="store_true",
        help="Adiciona 'normalize: true' no YAML e menciona no comentário gerado.",
    )
    out_group.add_argument(
        "--output-dir",
        default="hyperparams/best_params_for_food_delivery_gym",
        help=(
            "Diretório base de saída. O subdiretório de versão é criado automaticamente.\n"
            "Padrão: hyperparams/best_params_for_food_delivery_gym"
        ),
    )

    return parser.parse_args()


def validate_args(args):
    """Valida a combinação de argumentos e retorna o modo de operação."""
    use_optuna = bool(args.storage or args.study_name)

    if use_optuna:
        if not args.storage:
            raise SystemExit("[Erro] --storage é obrigatório quando --study-name é fornecido.")
        if not args.study_name:
            raise SystemExit("[Erro] --study-name é obrigatório quando --storage é fornecido.")
    else:
        if not args.trials_dir:
            raise SystemExit(
                "[Erro] No modo local (sem SQLite), --trials-dir é obrigatório.\n"
                "       Use --storage e --study-name para o modo Optuna."
            )

    return use_optuna


def main():
    args = parse_args()
    use_optuna = validate_args(args)

    # 1. Versão do pacote → subdiretório de saída
    version = get_food_delivery_gym_version()
    version_dir = version_to_dir(version)
    output_dir = os.path.join(args.output_dir, version_dir)
    output_yaml = os.path.join(output_dir, "ppo.yml")

    print(f"[Info] Versão do food_delivery_gym : {version}")
    print(f"[Info] Subdiretório de saída        : {output_dir}")
    print(f"[Info] Arquivo YAML de destino      : {output_yaml}")
    print(f"[Info] Modo de análise              : {'Optuna (SQLite)' if use_optuna else 'Local (evaluations.npz)'}")

    # 2. Identifica o melhor trial
    if use_optuna:
        study = load_study(args.storage, args.study_name)
        completed = [t for t in study.trials if t.value is not None]
        num_trials = len(completed)
        print(f"[Optuna] Total de trials concluídos : {num_trials}")
        best_trial = study.best_trial
        print(f"[Optuna] Melhor trial               : trial_{best_trial.number}  (valor={best_trial.value:.6f})")
        # Timesteps por trial = valor do último timestep no evaluations.npz (se disponível)
        n_timesteps_per_trial = best_trial.last_step or 0
    else:
        best_trial = scan_trials_locally(args.trials_dir)
        num_trials = best_trial.num_valid_trials
        n_timesteps_per_trial = best_trial.last_timestep

    # 3. env_id fornecido diretamente pelo usuário
    env_id = args.env
    print(f"[Info] env_id                       : {env_id}")

    # 4. Extrai hiperparâmetros do best_model.zip do melhor trial
    trials_dir = args.trials_dir
    zip_path = find_best_model_zip(trials_dir, best_trial.number) if trials_dir else None

    if zip_path is None:
        if use_optuna:
            print(
                f"\n[AVISO] best_model.zip não encontrado "
                + (f"em '{trials_dir}/trial_{best_trial.number}/'" if trials_dir else "(--trials-dir não fornecido)")
                + ".\n         Os hiperparâmetros serão inferidos dos parâmetros amostrados pelo Optuna."
            )
            hyperparams = {**best_trial.params, **best_trial.user_attrs}
        else:
            raise FileNotFoundError(
                f"best_model.zip não encontrado em '{trials_dir}/trial_{best_trial.number}/'. "
                "Sem o SQLite e sem o best_model.zip não é possível extrair os hiperparâmetros."
            )
    else:
        print(f"[Info] Lendo hiperparâmetros de     : {zip_path}")
        model_data = load_model_data_from_zip(zip_path)
        hyperparams = extract_hyperparams_from_model_data(model_data)
        # Usa o num_timesteps real do modelo como referência para o comentário
        if "num_timesteps" in hyperparams:
            n_timesteps_per_trial = hyperparams.pop("num_timesteps")

    # 5. Gera o comentário automaticamente
    comment = build_comment(n_timesteps_per_trial, num_trials, args.normalize)

    # 6. Exibe resumo na tela
    print_trial_summary(best_trial, hyperparams)

    # 7. Gera o bloco YAML
    yaml_block = build_yaml_block(
        env_id=env_id,
        hyperparams=hyperparams,
        n_timesteps=args.n_timesteps,
        n_envs=args.n_envs,
        normalize=args.normalize,
        comment=comment,
        num_trials=num_trials,
        best_trial_number=best_trial.number,
    )

    print("─" * 60)
    print("BLOCO YAML GERADO:")
    print("─" * 60)
    print(yaml_block)
    print("─" * 60)

    # 8. Adiciona ao arquivo ppo.yml (sem sobrescrever)
    append_yaml(output_yaml, yaml_block)
    print(f"\n[OK] Concluído.")


if __name__ == "__main__":
    main()