"""Script para entrenar continuamente y luego comparar con t-test
Ejecuta entrenamiento + evaluación automática al terminar
"""

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent


def run_training():
    """Ejecuta el script de entrenamiento continuo de visión."""
    print("\n" + "="*70)
    print("INICIANDO ENTRENAMIENTO CONTINUO DE VISIÓN".center(70))
    print("="*70 + "\n")
    
    script_path = PROJECT_ROOT / "rl_vision" / "continue_training_vision.py"
    
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=PROJECT_ROOT,
        capture_output=False
    )
    
    return result.returncode == 0


def run_ttest(episodes=15):
    """Ejecuta el t-test comparando modelo entrenado con el mejor modelo."""
    print("\n" + "="*70)
    print("INICIANDO T-TEST COMPARATIVO".center(70))
    print("="*70 + "\n")
    
    script_path = SCRIPTS_DIR / "compare_models_ttest.py"
    
    # Comparar modelo continuado con mejor modelo
    cmd = [
        sys.executable,
        str(script_path),
        "--model1-path", "models/vision/best_model_continued/best_model",
        "--model1-name", "Modelo Visión (Continuado)",
        "--model1-type", "vision",
        "--model2-path", "models/vision/best_model/best_model",
        "--model2-name", "Modelo Visión (Original - Mejor)",
        "--model2-type", "vision",
        "--episodes", str(episodes),
        "--alternative", "greater",  # ¿Es el continuado MEJOR?
        "--save-results", f"ttest_vision_comparison_n{episodes}.json"
    ]
    
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    
    return result.returncode == 0


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Entrenar y comparar modelos con t-test")
    parser.add_argument(
        "--episodes",
        type=int,
        default=15,
        help="Número de episodios para el t-test (15=rápido, 50=riguroso)"
    )
    args = parser.parse_args()
    
    print("\nPIPELINE: ENTRENAMIENTO + T-TEST COMPARATIVO")
    print(f"Tamaño de muestra: {args.episodes} episodios por modelo\n")
    
    # Ejecutar entrenamiento
    training_success = run_training()
    
    if not training_success:
        print("\nEl entrenamiento falló. Abortando pipeline.")
        return False
    
    print("\nEntrenamiento completado exitosamente!")
    
    # Ejecutar t-test
    ttest_success = run_ttest(episodes=args.episodes)
    
    if not ttest_success:
        print("\nEl t-test falló.")
        return False
    
    print("\nPipeline completado exitosamente!")
    print(f"Resultados guardados en: ttest_vision_comparison_n{args.episodes}.json")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
