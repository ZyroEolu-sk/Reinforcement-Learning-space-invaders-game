# Aprendizaje por Refuerzo aplicado a Space Invaders

Entrenamiento de agentes de aprendizaje por refuerzo sobre un clon propio de *Space Invaders*
desarrollado en Pygame. El proyecto compara dos formas de representar el estado del juego:

- **Enfoque vectorial** (`rl_vector/`): el agente observa un vector de caracteristicas numericas
  extraidas del estado del juego. Aprende rapido pero depende de un diseno manual de features.
- **Enfoque basado en vision** (`rl_vision/`): el agente observa directamente los pixeles del
  juego mediante una CNN. Mas costoso de entrenar, pero sin ingenieria de caracteristicas.

Sobre el enfoque de vision se han entrenado agentes con **PPO** y con **DQN**, ambos mediante
`stable-baselines3`.

El juego en si vive en un repositorio aparte, incluido aqui como submodulo de Git
(`space-invaders-game/`).

---

## Contenido del repositorio

```text
.
├─ rl_vector/                 # Enfoque vectorial (PPO sobre vector de caracteristicas)
│  ├─ gym_env.py              # Entorno Gymnasium: envuelve el juego y expone el vector de estado
│  ├─ train_rl.py             # Entrenamiento desde cero
│  ├─ continue_training.py    # Reanudacion desde un checkpoint
│  └─ play_rl.py              # Ejecucion/evaluacion de un modelo entrenado
│
├─ rl_vision/                 # Enfoque basado en vision (PPO y DQN sobre pixeles)
│  ├─ gym_env_vision.py       # Entorno para PPO: frames en escala de grises apilados
│  ├─ gym_env_dqn.py          # Entorno para DQN
│  ├─ custom_cnn.py           # Extractores de caracteristicas CNN (residual y simple)
│  ├─ train_rl_vision.py      # Entrenamiento PPO desde cero
│  ├─ train_dqn_vision.py     # Entrenamiento DQN desde cero
│  ├─ continue_training_vision.py  # Reanudacion PPO
│  ├─ continue_training_dqn.py     # Reanudacion DQN
│  ├─ play_rl_vision.py       # Ejecucion/evaluacion de un modelo PPO
│  └─ play_dqn_vision.py      # Ejecucion/evaluacion de un modelo DQN
│
├─ compare_models_ttest.py    # Comparacion estadistica (t-test) entre dos modelos
├─ train_and_compare.py       # Encadena entrenamiento continuo + comparacion automatica
├─ plot_learning_curves.py    # Genera curvas de aprendizaje a partir de los logs
│
├─ memoria.ipynb              # Memoria del proyecto (documento principal)
├─ MEMORIA_TECNICA.md         # Documento tecnico complementario
├─ OPTIMIZATION_GUIDE.md      # Notas sobre los ajustes de hiperparametros de vision
├─ TRAIN_OPTIMIZED.sh         # Atajo para lanzar el entrenamiento de vision ya configurado
│
├─ models/                    # Modelos entrenados, checkpoints y logs (ver seccion propia)
├─ space-invaders-game/       # Submodulo: el juego en Pygame
├─ pyproject.toml             # Dependencias (Python >= 3.11)
└─ uv.lock                    # Versiones fijadas
```

---

## Instalacion

### 1. Clonar con el submodulo

El juego es un submodulo, asi que hay que traerlo explicitamente. Sin este paso los entornos
no encuentran el codigo del juego y todos los scripts fallan al importar.

```bash
git clone --recurse-submodules https://github.com/ZyroEolu-sk/Reinforcement-Learning-space-invaders-game.git
cd Reinforcement-Learning-space-invaders-game
```

Si ya lo habias clonado sin el submodulo:

```bash
git submodule update --init --recursive
```

### 2. Instalar dependencias

El proyecto usa [`uv`](https://docs.astral.sh/uv/), que lee `pyproject.toml` y `uv.lock`:

```bash
uv sync
```

Esto crea `.venv/` con las versiones exactas del lockfile. Los comandos de las secciones
siguientes se ejecutan con `uv run <script>`, que usa ese entorno sin necesidad de activarlo.

Alternativa con `venv` y `pip`:

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
pip install -U pip
pip install gymnasium pygame scipy "stable-baselines3[extra]" tensorboard
```

Requisitos: **Python 3.11 o superior**.

---

## Ejecutar el juego manualmente

```bash
uv run space-invaders-game/src/main.py
```

Controles: flechas **izquierda/derecha** para moverse, **espacio** para disparar y **ESC**
para pausar. El record se guarda en `space-invaders-game/score.json`.

---

## Entrenamiento

Todos los scripts se lanzan desde la raiz del repositorio y aceptan `--help` para ver la lista
completa de opciones.

### Enfoque vectorial

```bash
uv run rl_vector/train_rl.py \
  --total-timesteps 2000000 \
  --num-envs 4 \
  --max-level 4 \
  --seed 42
```

Opciones principales: `--total-timesteps`, `--save-dir`, `--seed`, `--num-envs`, `--frame-skip`,
`--max-steps`, `--start-level`, `--max-level`, `--eval-freq`, `--checkpoint-freq`.

### Enfoque de vision con PPO

```bash
uv run rl_vision/train_rl_vision.py \
  --total-timesteps 2000000 \
  --num-envs 4 \
  --max-level 4 \
  --preset optimized \
  --lr-schedule cosine \
  --arch residual \
  --seed 42
```

Ademas de las opciones del caso vectorial, acepta:

| Opcion | Valores | Por defecto | Descripcion |
|---|---|---|---|
| `--preset` | `baseline`, `conservative`, `explore`, `optimized` | `baseline` | Conjunto de hiperparametros predefinido |
| `--lr-schedule` | `constant`, `linear`, `cosine`, `polynomial` | `cosine` | Planificacion del learning rate |
| `--arch` | `residual`, `simple` | `residual` | Arquitectura del extractor CNN |
| `--img-width` / `--img-height` | entero | — | Resolucion de la observacion |
| `--n-eval-episodes` | entero | — | Episodios por evaluacion periodica |
| `--obs-debug-freq` / `--obs-debug-dir` | — | — | Volcado de observaciones para depuracion |

El script `TRAIN_OPTIMIZED.sh` envuelve esta llamada con la configuracion recomendada:

```bash
./TRAIN_OPTIMIZED.sh [timesteps] [num_envs] [max_level]
```

### Enfoque de vision con DQN

```bash
uv run rl_vision/train_dqn_vision.py \
  --total-timesteps 2000000 \
  --num-envs 4 \
  --max-level 4 \
  --batch-size 256 \
  --seed 42
```

Acepta las mismas opciones que la version PPO mas `--batch-size`.

---

## Reanudar un entrenamiento

Los entrenamientos largos se reanudan desde un checkpoint en lugar de empezar de cero:

```bash
# Vectorial
uv run rl_vector/continue_training.py \
  --model-path models/vector/best_model/best_model \
  --additional-timesteps 1000000

# Vision (PPO)
uv run rl_vision/continue_training_vision.py \
  --model-path models/vision/best_model/best_model \
  --additional-timesteps 1000000

# Vision (DQN)
uv run rl_vision/continue_training_dqn.py \
  --model-path models/vision/best_model/best_model \
  --additional-timesteps 1000000
```

Los scripts de reanudacion permiten sobrescribir hiperparametros sin reentrenar desde cero:
`--override-learning-rate`, `--lr-schedule` y, segun el algoritmo,
`--override-ent-coef` y `--override-clip-range` (PPO) o
`--override-exploration-initial-eps` y `--override-exploration-final-eps` (DQN).

También aceptan `--comparison-episodes` para evaluar el modelo reanudado frente al de partida.

---

## Evaluar un modelo entrenado

```bash
# Vectorial
uv run rl_vector/play_rl.py --model-path models/vector/best_model/best_model --episodes 10

# Vision (PPO)
uv run rl_vision/play_rl_vision.py --model-path models/vision/best_model/best_model --episodes 10

# Vision (DQN)
uv run rl_vision/play_dqn_vision.py --model-path models/vision/best_model/best_model --episodes 10
```

Por defecto la politica actua de forma determinista. En los scripts de vision, `--stochastic`
muestrea de la distribucion de acciones; en el vectorial, el comportamiento se controla con
`--deterministic`.

---

## Analisis de resultados

```bash
# Comparacion estadistica entre dos modelos mediante t-test
uv run compare_models_ttest.py

# Curvas de aprendizaje a partir de los logs de entrenamiento
uv run plot_learning_curves.py

# Entrenamiento continuo seguido de comparacion automatica
uv run train_and_compare.py
```

Los logs son de TensorBoard, asi que tambien pueden inspeccionarse directamente:

```bash
uv run tensorboard --logdir models/vision/logs_continued
```

---

## Modelos, checkpoints y logs

Los artefactos de entrenamiento se organizan bajo `models/`, separados por enfoque:

```text
models/
├─ vector/
│  ├─ best_model/        # Mejor modelo segun la evaluacion periodica
│  ├─ logs/              # Logs de TensorBoard del entrenamiento inicial
│  └─ logs_continued/    # Logs de los entrenamientos reanudados
└─ vision/
   ├─ best_model/
   ├─ best_model_continued/
   ├─ logs/
   ├─ logs_continued/
   └─ level2_experiment/  # Experimento aislado sobre el nivel 2
```

Cada script crea sus directorios automaticamente, de modo que no hace falta prepararlos a mano.

**Los checkpoints intermedios no se versionan.** Cada uno ocupa unos 48 MB y una corrida larga
genera cientos, asi que `models/**/checkpoints/` esta excluido en `.gitignore`. En el repositorio
solo se conservan los modelos finales de `best_model/`. Si necesitas los checkpoints intermedios,
tendras que regenerarlos entrenando.

> **Nota sobre convenciones:** `train_dqn_vision.py` guarda en un subdirectorio propio por
> experimento (`models/vision/<identificador>/`), mientras que el resto de scripts escriben
> directamente en `models/vision/`. Conviene tenerlo en cuenta al buscar los resultados de una
> corrida concreta.

---

## Documentacion del proyecto

- **`memoria.ipynb`** — memoria principal: formalizacion del problema como MDP, diseno iterativo
  de la funcion de recompensa, estado del arte, arquitecturas, diseno experimental y resultados.
- **`MEMORIA_TECNICA.md`** — documento tecnico complementario.
- **`OPTIMIZATION_GUIDE.md`** — registro de los ajustes de hiperparametros aplicados al
  entrenamiento de vision y su motivacion.

---

## Reproducibilidad

- La version de Python queda fijada en `.python-version` (3.11) y las dependencias en `uv.lock`.
- Todos los scripts de entrenamiento aceptan `--seed`. Los resultados de la memoria se
  obtuvieron con `--seed 42`.
- Al anadir o actualizar dependencias, hay que reflejarlo en `pyproject.toml` y regenerar el
  lockfile con `uv sync`.

---

## Solucion de problemas

**`ModuleNotFoundError` al importar el juego (`settings`, `space_invaders`...)**
El submodulo no esta inicializado. Ejecuta `git submodule update --init --recursive`.

**`No module named 'pygame'` u otra dependencia**
El entorno no esta sincronizado. Ejecuta `uv sync`, o usa `uv run` en lugar de `python`
directamente.

**`FileNotFoundError` cargando imagenes o sonidos**
Los scripts resuelven las rutas respecto a la raiz del repositorio. Lanzalos desde ahi y no
desde subdirectorios.

---

## Creditos

Hecho por:

- [ZyroEolu-sk](https://github.com/ZyroEolu-sk)
- [pvinas23](https://github.com/pvinas23)

Proyecto de nuestros primeros pasos dentro de la programación creando un juego con Python y Pygame. Lo hemos ordenado respecto al código original.
