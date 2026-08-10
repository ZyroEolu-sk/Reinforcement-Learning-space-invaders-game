"""Script para comparar dos modelos RL usando t-test
Ejecuta múltiples episodios de cada modelo y realiza análisis estadístico
"""

import argparse
import sys
from pathlib import Path
import json
import numpy as np
from scipy import stats
from typing import List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Este script vive en scripts/, pero importa los entornos desde la raíz del
# repositorio, así que hay que añadirla al path antes de los imports locales.
#
# Además hay que añadir rl_vector/ y rl_vision/. Los modelos de visión se
# guardaron con el extractor SpaceInvadersResidualSiluCNN, y al deserializarlos
# cloudpickle busca 'custom_cnn' como módulo de primer nivel: sin esto,
# PPO.load falla con ModuleNotFoundError sobre cualquier modelo de visión.
for _p in (PROJECT_ROOT, PROJECT_ROOT / "rl_vector", PROJECT_ROOT / "rl_vision"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from stable_baselines3 import PPO
from rl_vector.gym_env import SpaceInvadersGymEnv
from rl_vision.gym_env_vision import SpaceInvadersVisionEnv


def _resolve_project_path(path: str) -> Path:
    """Resolver ruta relativa a la raíz del proyecto."""
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return (PROJECT_ROOT / candidate).resolve()


def evaluate_vector_model(model_path: str, episodes: int = 10, verbose: bool = True) -> List[float]:
    """Evalúa el modelo vectorial y retorna lista de puntuaciones.
    
    Args:
        model_path: Ruta al modelo PPO
        episodes: Número de episodios a jugar
        verbose: Mostrar información durante la evaluación
        
    Returns:
        Lista de puntuaciones obtenidas
    """
    scores = []
    model_path_resolved = str(_resolve_project_path(model_path))
    
    print(f"Cargando modelo vectorial: {model_path_resolved}")
    model = PPO.load(model_path_resolved)
    
    for episode in range(episodes):
        env = SpaceInvadersGymEnv(
            render_mode=None,
            max_steps=12000,
            frame_skip=1,
            start_level=1,
            max_level=4
        )
        
        obs, info = env.reset()
        terminated = False
        truncated = False
        episode_score = 0
        
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_score = info.get('score', 0)
        
        scores.append(episode_score)
        env.close()
        
        if verbose:
            print(f"Episodio {episode + 1}/{episodes}: Score = {episode_score}")
    
    return scores


def evaluate_vision_model(model_path: str, episodes: int = 10, verbose: bool = True) -> List[float]:
    """Evalúa el modelo de visión y retorna lista de puntuaciones.
    
    Args:
        model_path: Ruta al modelo PPO
        episodes: Número de episodios a jugar
        verbose: Mostrar información durante la evaluación
        
    Returns:
        Lista de puntuaciones obtenidas
    """
    scores = []
    model_path_resolved = str(_resolve_project_path(model_path))
    
    print(f"Cargando modelo de visión: {model_path_resolved}")
    model = PPO.load(model_path_resolved)
    
    # Inferir tamaño de imagen desde el modelo
    shape = model.observation_space.shape
    if shape[0] == 4:
        img_width, img_height = int(shape[2]), int(shape[1])
    else:
        img_width, img_height = int(shape[1]), int(shape[0])
    
    for episode in range(episodes):
        env = SpaceInvadersVisionEnv(
            render_mode=None,
            max_steps=24000,
            frame_skip=2,
            start_level=1,
            max_level=4,
            img_width=img_width,
            img_height=img_height,
            enable_combo_actions=True
        )
        
        obs, info = env.reset()
        terminated = False
        truncated = False
        episode_score = 0
        
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_score = info.get('score', 0)
        
        scores.append(episode_score)
        env.close()
        
        if verbose:
            print(f"Episodio {episode + 1}/{episodes}: Score = {episode_score}")
    
    return scores


def perform_ttest(scores1: List[float], scores2: List[float], model1_name: str = "Modelo 1", 
                  model2_name: str = "Modelo 2", alternative: str = "two-sided") -> dict:
    """Realiza t-test independiente entre dos muestras.
    
    Args:
        scores1: Puntuaciones del modelo 1
        scores2: Puntuaciones del modelo 2
        model1_name: Nombre del modelo 1
        model2_name: Nombre del modelo 2
        alternative: 'two-sided' (¿son diferentes?), 'greater' (model1 > model2), 
                    'less' (model1 < model2)
    
    Returns:
        Diccionario con resultados del análisis
    """
    scores1 = np.array(scores1)
    scores2 = np.array(scores2)
    
    # Estadísticas descriptivas
    mean1, std1 = scores1.mean(), scores1.std()
    mean2, std2 = scores2.mean(), scores2.std()
    
    # T-test independiente
    t_stat, p_value = stats.ttest_ind(scores1, scores2, alternative=alternative)
    
    # Tamaño del efecto (Cohen's d)
    pooled_std = np.sqrt(((len(scores1)-1)*std1**2 + (len(scores2)-1)*std2**2) / (len(scores1)+len(scores2)-2))
    cohens_d = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0
    
    # Intervalo de confianza (95%)
    se_diff = np.sqrt(std1**2/len(scores1) + std2**2/len(scores2))
    ci_lower = (mean1 - mean2) - 1.96 * se_diff
    ci_upper = (mean1 - mean2) + 1.96 * se_diff
    
    results = {
        "model1_name": model1_name,
        "model2_name": model2_name,
        "model1_mean": float(mean1),
        "model1_std": float(std1),
        "model1_n": len(scores1),
        "model2_mean": float(mean2),
        "model2_std": float(std2),
        "model2_n": len(scores2),
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "cohens_d": float(cohens_d),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "alternative": alternative,
        "is_significant": p_value < 0.05
    }
    
    return results


def print_results(results: dict):
    """Imprime los resultados del t-test de forma amigable."""
    print("\n" + "="*70)
    print("RESULTADOS DEL T-TEST".center(70))
    print("="*70)
    
    print(f"\n{results['model1_name']}:")
    print(f"Media: {results['model1_mean']:.2f} ± {results['model1_std']:.2f}")
    print(f"N: {results['model1_n']} episodios")
    
    print(f"\n{results['model2_name']}:")
    print(f"Media: {results['model2_mean']:.2f} ± {results['model2_std']:.2f}")
    print(f"N: {results['model2_n']} episodios")
    
    print(f"\nPrueba Estadística ({results['alternative']}):")
    print(f"Estadístico t: {results['t_statistic']:.4f}")
    print(f"Valor p: {results['p_value']:.6f}")
    print(f"Tamaño del efecto (Cohen's d): {results['cohens_d']:.4f}")
    print(f"IC 95% de la diferencia: [{results['ci_lower']:.2f}, {results['ci_upper']:.2f}]")
    
    print(f"\nResultado:" if results['is_significant'] else "\nResultado:")
    if results['is_significant']:
        print(f"La diferencia ES estadísticamente significativa (p < 0.05)")
        if results['alternative'] == 'greater':
            print(f"  {results['model1_name']} obtiene puntuaciones SIGNIFICATIVAMENTE MAYORES")
        elif results['alternative'] == 'less':
            print(f"  {results['model1_name']} obtiene puntuaciones SIGNIFICATIVAMENTE MENORES")
        else:
            print(f"Los modelos obtienen puntuaciones SIGNIFICATIVAMENTE DIFERENTES")
    else:
        print(f"NO hay diferencia estadísticamente significativa (p ≥ 0.05)")
        print(f"No podemos afirmar que un modelo es mejor que el otro")
    
    print("\n" + "="*70)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Comparar dos modelos RL usando t-test"
    )
    parser.add_argument(
        "--model1-path",
        type=str,
        default="models/vector/best_model/best_model",
        help="Ruta al primer modelo"
    )
    parser.add_argument(
        "--model1-name",
        type=str,
        default="Modelo Vectorial",
        help="Nombre del primer modelo"
    )
    parser.add_argument(
        "--model1-type",
        type=str,
        choices=["vector", "vision"],
        default="vector",
        help="Tipo del primer modelo (vector o vision)"
    )
    parser.add_argument(
        "--model2-path",
        type=str,
        default="models/vision/best_model/best_model",
        help="Ruta al segundo modelo"
    )
    parser.add_argument(
        "--model2-name",
        type=str,
        default="Modelo Visión",
        help="Nombre del segundo modelo"
    )
    parser.add_argument(
        "--model2-type",
        type=str,
        choices=["vector", "vision"],
        default="vision",
        help="Tipo del segundo modelo (vector o vision)"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Número de episodios para cada modelo"
    )
    parser.add_argument(
        "--alternative",
        type=str,
        choices=["two-sided", "greater", "less"],
        default="greater",
        help="Tipo de prueba: two-sided (¿diferentes?), greater (model1 > model2), less (model1 < model2)"
    )
    parser.add_argument(
        "--save-results",
        type=str,
        default=None,
        help="Guardar resultados en archivo JSON"
    )
    parser.add_argument(
        "--no-verbose",
        action="store_true",
        help="No mostrar progreso de cada episodio"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("\nIniciando comparación de modelos...\n")
    
    # Evaluar modelo 1
    print(f"Evaluando {args.model1_name}...")
    if args.model1_type == "vector":
        scores1 = evaluate_vector_model(
            args.model1_path,
            episodes=args.episodes,
            verbose=not args.no_verbose
        )
    else:
        scores1 = evaluate_vision_model(
            args.model1_path,
            episodes=args.episodes,
            verbose=not args.no_verbose
        )
    
    # Evaluar modelo 2
    print(f"\nEvaluando {args.model2_name}...")
    if args.model2_type == "vector":
        scores2 = evaluate_vector_model(
            args.model2_path,
            episodes=args.episodes,
            verbose=not args.no_verbose
        )
    else:
        scores2 = evaluate_vision_model(
            args.model2_path,
            episodes=args.episodes,
            verbose=not args.no_verbose
        )
    
    # Realizar t-test
    results = perform_ttest(
        scores1, scores2,
        model1_name=args.model1_name,
        model2_name=args.model2_name,
        alternative=args.alternative
    )
    
    # Mostrar resultados
    print_results(results)
    
    # Guardar resultados si se especifica
    if args.save_results:
        with open(args.save_results, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResultados guardados en: {args.save_results}")


if __name__ == "__main__":
    main()
