# training_pipeline/register_best_model.py
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType


def _load_env() -> Optional[Path]:
    """
    Carga variables desde un .env aunque esté fuera de training_pipeline/.
    Estrategia:
      - Si existe ENV_PATH, usarlo.
      - Si no, buscar .env hacia arriba desde este archivo (2-3 niveles).
      - Si no, intentar usar python-dotenv find_dotenv si está instalado.
    """
    env_hint = os.getenv("ENV_PATH")
    if env_hint:
        p = Path(env_hint).expanduser().resolve()
        if p.exists():
            try:
                from dotenv import load_dotenv
                load_dotenv(dotenv_path=str(p), override=True)
                print(f"[env] Cargado desde ENV_PATH={p}")
                return p
            except Exception as e:
                print(f"[env] Aviso: no pude cargar ENV_PATH={p}: {e}")

    here = Path(__file__).resolve()
    candidates = [
        here.parents[1] / ".env",  
        here.parents[2] / ".env",  
        here.parents[3] / ".env",  
    ]
    for cand in candidates:
        if cand.exists():
            try:
                from dotenv import load_dotenv
                load_dotenv(dotenv_path=str(cand), override=True)
                print(f"[env] Cargado desde {cand}")
                return cand
            except Exception as e:
                print(f"[env] Aviso: no pude cargar {cand}: {e}")

    try:
        from dotenv import load_dotenv, find_dotenv
        env_file = find_dotenv(filename=".env", raise_error_if_not_found=False, usecwd=True)
        if env_file:
            load_dotenv(env_file, override=True)
            print(f"[env] Cargado con find_dotenv: {env_file}")
            return Path(env_file)
    except Exception:
        pass

    print("[env] No se encontró un .env (continuando si ya estás autenticado en Databricks).")
    return None



EXPERIMENT_NAME = os.getenv(
    "MLFLOW_EXPERIMENT_PATH",
    "/Users/esteban.berumen@iteso.mx/nyc-taxi-experiments",
)


MODEL_REGISTRY_NAME = os.getenv(
    "MODEL_REGISTRY_NAME",
    "time_series.default.nyc-taxi-model-prefect",  
)

METRIC = os.getenv("BEST_METRIC", "rmse")
HIGHER_IS_BETTER = os.getenv("HIGHER_IS_BETTER", "false").lower() in ("1", "true", "yes")
ARTIFACT_PATH = os.getenv("MODEL_ARTIFACT_PATH", "model")  # Debe coincidir con log_model(..., artifact_path="model")

WAIT_TIMEOUT_S = int(os.getenv("MODEL_WAIT_TIMEOUT", "180"))



def _ensure_tracking_databricks() -> None:
    mlflow.set_tracking_uri("databricks")
    print("tracking:", mlflow.get_tracking_uri())

    host = os.getenv("DATABRICKS_HOST")
    token = os.getenv("DATABRICKS_TOKEN")
    profile = os.getenv("DATABRICKS_CONFIG_PROFILE")
    if host and token:
        print(f"[auth] DATABRICKS_HOST y TOKEN presentes.")
    elif profile:
        print(f"[auth] Usando perfil de ~/.databrickscfg: {profile}")
    else:
        print("[auth] Aviso: no hay DATABRICKS_HOST/TOKEN ni DATABRICKS_CONFIG_PROFILE; "
              "asumo login previo (CLI/Databricks Connect).")


def _resolve_experiment_id(client: MlflowClient, experiment_path: str) -> str:
    exp = client.get_experiment_by_name(experiment_path)
    if exp is None:
        print(f"[exp] No existe '{experiment_path}', creando…")
        exp_id = client.create_experiment(experiment_path)
        print(f"[exp] Creado con id={exp_id}")
        return exp_id
    print(f"[exp] Usando experiment id={exp.experiment_id} | name={exp.name}")
    return exp.experiment_id


def _wait_ready(client: MlflowClient, model_name: str, version: str, timeout_s: int = 180) -> None:
    """
    Espera a que la versión del modelo esté en estado READY (Databricks puede demorar).
    """
    t0 = time.time()
    while True:
        cur = client.get_model_version(name=model_name, version=version)
        status = getattr(cur, "status", "READY")
        if status == "READY":
            return
        if time.time() - t0 > timeout_s:
            raise TimeoutError(f"La versión {version} de '{model_name}' no llegó a READY en {timeout_s}s.")
        time.sleep(2)



def register_best_model(
    experiment_name: str = EXPERIMENT_NAME,
    model_registry_name: str = MODEL_REGISTRY_NAME,
    metric: str = METRIC,
    higher_is_better: bool = HIGHER_IS_BETTER,
    artifact_path: str = ARTIFACT_PATH,
    wait_timeout_s: int = WAIT_TIMEOUT_S,
) -> dict:
    """
    - Selecciona/crea experimento en Databricks.
    - Busca el mejor run por métrica.
    - Registra runs:/<run_id>/<artifact_path> en Model Registry.
    - Asigna alias @champion.
    """
    _load_env()
    _ensure_tracking_databricks()
    client = MlflowClient()

    # 1) Resolver experimento
    exp_id = _resolve_experiment_id(client, experiment_name)

    # 2) Buscar mejor run
    order_dir = "DESC" if higher_is_better else "ASC"
    runs = client.search_runs(
        experiment_ids=[exp_id],
        filter_string="attributes.status = 'FINISHED'",
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=1,
        order_by=[f"metrics.{metric} {order_dir}"],
    )
    if not runs:
        recent = client.search_runs(
            experiment_ids=[exp_id],
            filter_string="",
            run_view_type=ViewType.ALL,
            max_results=5,
            order_by=["attributes.start_time DESC"],
        )
        msg = f"No se encontraron runs con la métrica '{metric}' en estado FINISHED."
        if recent:
            msg += " Hay runs recientes, pero revisa que logueen esa métrica y que el run terminó."
        raise RuntimeError(msg)

    best = runs[0]
    run_id = best.info.run_id
    best_val = best.data.metrics.get(metric)
    if best_val is None:
        raise RuntimeError(f"El mejor run ({run_id}) no tiene la métrica '{metric}'.")

    run_uri = f"runs:/{run_id}/{artifact_path}"
    print(f"🏆 Mejor run: {run_id} | {metric}={best_val:.6f}")
    print("run_uri:", run_uri)

    mv = mlflow.register_model(model_uri=run_uri, name=model_registry_name)

    _wait_ready(client, model_registry_name, mv.version, timeout_s=wait_timeout_s)

    client.set_registered_model_alias(
        name=model_registry_name,
        alias="champion",
        version=mv.version,
    )

    result = {
        "model_name": model_registry_name,
        "model_version": mv.version,
        "alias": "champion",
        "experiment_id": exp_id,
        "best_run_id": run_id,
        "metric": metric,
        "metric_value": best_val,
        "run_uri": run_uri,
    }
    print("✅ Registro completado:", result)
    return result


def main():
    register_best_model()


if __name__ == "__main__":
    main()
