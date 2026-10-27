"""
Paso 3b: EXPERIMENTOS con la red neuronal (ajuste de hiperparámetros), de forma justa.

    reports/experimentos_red.csv   una fila por (configuración, semilla)

¿POR QUÉ?
    La primera red funcionó (R² 0.878), pero mostró dos problemas:
      1. Sobreajuste: el error de train siguió bajando (0.14) y el de validación se quedó en 0.214.
      2. Los embeddings de barrios parecen ruido (correlación con el precio 0.07). Causa probable:
         PyTorch inicia cada embedding con números N(0, 1), grandes y al azar; con vectores de 41
         dimensiones la red puede usarlos como "códigos de identidad" sin ordenarlos por similitud.
    Los embeddings nos importan porque el RECOMENDADOR va a comparar viviendas con esos vectores.

REGLAS PARA QUE EL EXPERIMENTO SEA HONESTO
    - Cada configuración se entrena con 3 semillas distintas: si una mejora es menor que la
      variación entre semillas, no es una mejora real, es suerte.
    - Se ELIGE con VALIDACIÓN. El conjunto de prueba se mira solo al final, para la ganadora
      (si eligiéramos con prueba, la nota de prueba dejaría de ser una evaluación independiente).

CALIDAD DE LOS EMBEDDINGS ("coherencia de vecinos")
    Para cada barrio se buscan los 5 barrios con el vector más parecido (similitud coseno) y se
    compara su precio con el promedio de esos vecinos. Correlación cercana a 0 = vectores al azar;
    cercana a 1 = la red agrupó barrios de precio parecido.

Ejecutar:  python -u -m src.models.red_experimentos
"""

import time

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.models.comun import REPORTES, cargar, columnas_entrada, metricas
from src.models.red_precio import LOTE, LR, MAX_EPOCAS, PACIENCIA, RedPrecio, a_tensores, predecir

SEMILLAS = [42, 7, 2026]
CONFIGURACIONES = {
    #  nombre          tope dim. embedding, desv. inicial, dropout, weight decay
    "A_actual":       dict(tope=50, init=None, dropout=0.2, wd=1e-4),
    "B_init_pequeno": dict(tope=50, init=0.05, dropout=0.2, wd=1e-4),
    "C_emb16":        dict(tope=16, init=0.05, dropout=0.2, wd=1e-4),
    "D_emb16_reg":    dict(tope=16, init=0.05, dropout=0.3, wd=1e-3),
    "E_emb8_reg":     dict(tope=8,  init=0.05, dropout=0.3, wd=1e-3),
}


def entrenar(cfg: dict, semilla: int, datos: dict, n_num: int, cardinalidades: list[int], y_media, y_desv):
    torch.manual_seed(semilla)
    modelo = RedPrecio(n_num, cardinalidades, dropout=cfg["dropout"],
                       tope_embedding=cfg["tope"], init_embedding=cfg["init"]).to(datos["train"][0].device)
    opt = torch.optim.AdamW(modelo.parameters(), lr=LR, weight_decay=cfg["wd"])
    plan = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=8)
    perdida_fn = nn.HuberLoss(delta=1.0)
    x_tr, c_tr, y_tr = datos["train"]
    x_va, c_va, y_va_log = datos["val"]
    n = len(y_tr)
    mejor, mejor_estado, sin_mejora, ep = np.inf, None, 0, 0
    for ep in range(1, MAX_EPOCAS + 1):
        modelo.train()
        orden = torch.randperm(n, device=x_tr.device)
        for i in range(0, n, LOTE):
            lote = orden[i:i + LOTE]
            opt.zero_grad()
            perdida_fn(modelo(x_tr[lote], c_tr[lote]), y_tr[lote]).backward()
            opt.step()
        rmse_va = metricas(y_va_log, predecir(modelo, x_va, c_va, y_media, y_desv))["RMSE_log"]
        plan.step(rmse_va)
        if rmse_va < mejor - 1e-4:
            mejor, sin_mejora = rmse_va, 0
            mejor_estado = {k: v.detach().clone() for k, v in modelo.state_dict().items()}
        else:
            sin_mejora += 1
            if sin_mejora >= PACIENCIA:
                break
    modelo.load_state_dict(mejor_estado)
    return modelo, ep


def coherencia_vecinos(modelo, meta: dict, train: pd.DataFrame, k: int = 5) -> float:
    """Correlación entre log(precio mediano) de cada barrio y el promedio de sus k vecinos en el embedding."""
    i_barrio = meta["categoricas"].index("barrio_modelo")
    E = modelo.embeddings[i_barrio].weight.detach().cpu().numpy()[1:]          # sin "OTRO" (índice 0)
    precio = np.log(train.groupby("barrio_modelo_idx")["precio"].median().reindex(range(1, len(E) + 1)).to_numpy())
    ok = ~np.isnan(precio)
    E, precio = E[ok], precio[ok]
    En = E / np.linalg.norm(E, axis=1, keepdims=True)
    sim = En @ En.T
    np.fill_diagonal(sim, -np.inf)                                             # un barrio no es su propio vecino
    vecinos = np.argsort(-sim, axis=1)[:, :k]
    return float(np.corrcoef(precio, precio[vecinos].mean(axis=1))[0, 1])


def main() -> None:
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train, val, test, meta = cargar()
    num, cat = columnas_entrada(meta)
    cardinalidades = [len(meta["vocabulario"][c.replace("_idx", "")]) for c in cat]
    y_media, y_desv = float(train["log_precio"].mean()), float(train["log_precio"].std())
    datos = {}
    for nombre, df in [("train", train), ("val", val), ("test", test)]:
        x_num, x_cat = a_tensores(df, num, cat, meta, dispositivo)
        y = (torch.tensor(((df["log_precio"] - y_media) / y_desv).to_numpy(), dtype=torch.float32, device=dispositivo)
             if nombre == "train" else df["log_precio"].to_numpy())
        datos[nombre] = (x_num, x_cat, y)

    filas, modelos = [], {}
    print(f"{len(CONFIGURACIONES)} configuraciones x {len(SEMILLAS)} semillas en {dispositivo}\n")
    for nombre, cfg in CONFIGURACIONES.items():
        for s in SEMILLAS:
            t0 = time.time()
            modelo, epocas = entrenar(cfg, s, datos, len(num), cardinalidades, y_media, y_desv)
            pred_tr = predecir(modelo, *datos["train"][:2], y_media, y_desv)
            pred_va = predecir(modelo, *datos["val"][:2], y_media, y_desv)
            fila = {"config": nombre, "semilla": s, **cfg, "epocas": epocas,
                    "rmse_train": metricas(train["log_precio"], pred_tr)["RMSE_log"],
                    "rmse_val": metricas(val["log_precio"], pred_va)["RMSE_log"],
                    "mdape_val": metricas(val["log_precio"], pred_va)["MdAPE_%"],
                    "coherencia_barrios": coherencia_vecinos(modelo, meta, train),
                    "segundos": time.time() - t0}
            filas.append(fila)
            modelos[(nombre, s)] = modelo
            print(f"  {nombre:<15} semilla {s:<5} épocas {epocas:>3} | RMSE train {fila['rmse_train']:.4f} "
                  f"val {fila['rmse_val']:.4f} | MdAPE val {fila['mdape_val']:.1f}% | "
                  f"coherencia barrios {fila['coherencia_barrios']:.2f} | {fila['segundos']:.0f} s")

    res = pd.DataFrame(filas)
    res.to_csv(REPORTES / "experimentos_red.csv", index=False)
    resumen = (res.groupby("config")
               .agg(rmse_val=("rmse_val", "mean"), rmse_val_desv=("rmse_val", "std"),
                    brecha=("rmse_train", "mean"), mdape_val=("mdape_val", "mean"),
                    coherencia=("coherencia_barrios", "mean"), epocas=("epocas", "mean")))
    resumen["brecha"] = resumen["rmse_val"] - resumen["brecha"]          # val - train: tamaño del sobreajuste
    print("\nRESUMEN (promedio de 3 semillas; se elige por RMSE de VALIDACIÓN):")
    print(resumen.sort_values("rmse_val").round(4).to_string())

    ganadora = resumen["rmse_val"].idxmin()
    print(f"\nGanadora por validación: {ganadora}  {CONFIGURACIONES[ganadora]}")
    print("Su resultado en PRUEBA (cada semilla) -- se mira solo ahora:")
    for s in SEMILLAS:
        m = metricas(test["log_precio"], predecir(modelos[(ganadora, s)], *datos["test"][:2], y_media, y_desv))
        print(f"  semilla {s:<5} RMSE_log {m['RMSE_log']:.4f} | R2_log {m['R2_log']:.3f} | MdAPE {m['MdAPE_%']:.1f}% | "
              f"±20%: {m['dentro_20_%']:.1f}%")
    print(f"\nGuardado: {REPORTES / 'experimentos_red.csv'}")


if __name__ == "__main__":
    main()
