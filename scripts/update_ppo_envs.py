import re
import gymnasium
import food_delivery_gym

MARKER = "# ========== FOOD DELIVERY GYM ENVS =========="
YAML_PATH = "./hyperparams/ppo.yml"

DEFAULTS = """.defaults: &defaults
  n_envs: 4
  n_timesteps: 100000
  policy: 'MultiInputPolicy'
  normalize: true
"""

def generate_env_entries(env_ids):
    lines = [DEFAULTS]
    for env_id in env_ids:
        lines.append(f"{env_id}:\n  <<: *defaults\n")
    return "\n".join(lines)

def update_yaml(path):
    with open(path, "r") as f:
        content = f.read()

    # Encontra o marcador e corta tudo abaixo
    marker_pos = content.find(MARKER)
    if marker_pos == -1:
        raise ValueError("Marcador não encontrado no arquivo!")

    # Mantém até o fim da linha do segundo comentário
    after_marker = content[marker_pos:]
    second_newline = after_marker.find("\n", after_marker.find("\n") + 1)
    header = content[:marker_pos + second_newline + 1]

    env_ids = sorted([
        env_id for env_id in gymnasium.envs.registry.keys()
        if env_id.startswith("food_delivery_gym/")
    ])

    new_content = header + "\n" + generate_env_entries(env_ids)

    with open(path, "w") as f:
        f.write(new_content)

    print(f"{len(env_ids)} envs escritos em {path}")

update_yaml(YAML_PATH)