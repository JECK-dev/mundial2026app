# ============================================================
# Predictor Mundial 2026 - Cruces de selecciones
# Archivo base: mundial2026_ML_dataset.xlsx
# Autor: generado para análisis en Jupyter / PyCharm
# ============================================================

import math
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

EXCEL_PATH = "mundial2026_ML_dataset.xlsx"
SHEET_NAME = "Dataset Principal"

# -------------------------------
# 1. Cargar dataset
# -------------------------------
def cargar_dataset(path=EXCEL_PATH):
    # El Excel tiene título en la primera fila, por eso usamos header=1
    df = pd.read_excel(path, sheet_name=SHEET_NAME, header=1)

    # Eliminar columnas vacías si existieran
    df = df.dropna(axis=1, how="all")

    # Limpiar nombres de columnas
    df.columns = [str(c).strip() for c in df.columns]

    # Eliminar filas sin equipo
    df = df[df["Equipo"].notna()].copy()

    return df


# -------------------------------
# 2. Calcular score de poder
# -------------------------------
def construir_scores(df):
    df = df.copy()

    # Convertir variables numéricas clave
    numeric_cols = [
        "Ranking_FIFA", "Rating_Elo", "Valor_Mercado_M_EUR",
        "Victorias_U10", "Empates_U10", "Derrotas_U10", "Puntos_U10",
        "Goles_Anotados_U10", "Goles_Recibidos_U10", "Dif_Gol_U10",
        "Porterias_Cero_U10", "Promedio_Goles_U10", "xG_Promedio",
        "xGA_Promedio", "Tiros_Por_Partido", "Tiros_Al_Arco",
        "Posesion_Promedio", "Grandes_Ocasiones_Creadas",
        "Grandes_Ocasiones_Concedidas", "Suspensiones",
        "Dias_Descanso_Promedio", "Cuota_Ganar_Mundial",
        "Cuota_Octavos", "Cuota_Semifinales", "Forma_5_Puntos"
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Rellenar faltantes numéricos con la mediana
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # Variables invertidas: menor ranking FIFA y menor cuota = mejor
    df["Ranking_FIFA_inv"] = -df["Ranking_FIFA"]
    df["Cuota_Ganar_inv"] = -df["Cuota_Ganar_Mundial"]
    df["Cuota_Semifinales_inv"] = -df["Cuota_Semifinales"]
    df["xGA_inv"] = -df["xGA_Promedio"]
    df["Goles_Recibidos_inv"] = -df["Goles_Recibidos_U10"]
    df["Grandes_Ocasiones_Concedidas_inv"] = -df["Grandes_Ocasiones_Concedidas"]
    df["Derrotas_inv"] = -df["Derrotas_U10"]

    score_features = [
        "Rating_Elo",
        "Ranking_FIFA_inv",
        "Valor_Mercado_M_EUR",
        "Forma_5_Puntos",
        "Puntos_U10",
        "Goles_Anotados_U10",
        "Dif_Gol_U10",
        "xG_Promedio",
        "xGA_inv",
        "Tiros_Al_Arco",
        "Porterias_Cero_U10",
        "Grandes_Ocasiones_Creadas",
        "Grandes_Ocasiones_Concedidas_inv",
        "Cuota_Ganar_inv",
        "Cuota_Semifinales_inv",
        "Derrotas_inv"
    ]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[score_features])
    X_scaled = pd.DataFrame(X_scaled, columns=[f"{c}_z" for c in score_features])

    df = pd.concat([df.reset_index(drop=True), X_scaled.reset_index(drop=True)], axis=1)

    # Score ponderado: mayor = selección más fuerte
    df["Score_Poder"] = (
        df["Rating_Elo_z"] * 0.22 +
        df["Ranking_FIFA_inv_z"] * 0.08 +
        df["Valor_Mercado_M_EUR_z"] * 0.10 +
        df["Forma_5_Puntos_z"] * 0.10 +
        df["Puntos_U10_z"] * 0.10 +
        df["Goles_Anotados_U10_z"] * 0.07 +
        df["Dif_Gol_U10_z"] * 0.09 +
        df["xG_Promedio_z"] * 0.10 +
        df["xGA_inv_z"] * 0.08 +
        df["Tiros_Al_Arco_z"] * 0.04 +
        df["Porterias_Cero_U10_z"] * 0.04 +
        df["Grandes_Ocasiones_Creadas_z"] * 0.03 +
        df["Grandes_Ocasiones_Concedidas_inv_z"] * 0.03 +
        df["Cuota_Ganar_inv_z"] * 0.05 +
        df["Cuota_Semifinales_inv_z"] * 0.04 +
        df["Derrotas_inv_z"] * 0.03
    )

    return df


# -------------------------------
# 3. Función de probabilidad
# -------------------------------
def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def predecir_partido(df, equipo_a, equipo_b, ventaja_local=0.0):
    df = df.copy()

    equipos = df["Equipo"].str.lower().tolist()

    if equipo_a.lower() not in equipos:
        raise ValueError(f"No encontré el equipo: {equipo_a}")

    if equipo_b.lower() not in equipos:
        raise ValueError(f"No encontré el equipo: {equipo_b}")

    A = df[df["Equipo"].str.lower() == equipo_a.lower()].iloc[0]
    B = df[df["Equipo"].str.lower() == equipo_b.lower()].iloc[0]

    diff_score = (A["Score_Poder"] - B["Score_Poder"]) + ventaja_local

    # Probabilidad de empate: más alta cuando los equipos son parejos
    p_empate = 0.29 * math.exp(-abs(diff_score) * 0.45)
    p_empate = max(0.12, min(0.31, p_empate))

    restante = 1 - p_empate
    p_a = sigmoid(diff_score * 1.35) * restante
    p_b = restante - p_a

    resultado = pd.DataFrame({
        "Resultado": [f"Gana {A['Equipo']}", "Empate", f"Gana {B['Equipo']}"],
        "Probabilidad": [p_a, p_empate, p_b]
    })

    resultado["Probabilidad_%"] = (resultado["Probabilidad"] * 100).round(2)

    resumen = {
        "Equipo_A": A["Equipo"],
        "Equipo_B": B["Equipo"],
        "Score_A": round(A["Score_Poder"], 3),
        "Score_B": round(B["Score_Poder"], 3),
        "Diferencia_Score": round(diff_score, 3),
        "Perfil_A": A.get("Perfil_Competitivo", "N/D"),
        "Perfil_B": B.get("Perfil_Competitivo", "N/D"),
        "Elo_A": A["Rating_Elo"],
        "Elo_B": B["Rating_Elo"],
        "xG_A": A["xG_Promedio"],
        "xG_B": B["xG_Promedio"],
        "xGA_A": A["xGA_Promedio"],
        "xGA_B": B["xGA_Promedio"],
        "Forma_A": A["Forma_5_Puntos"],
        "Forma_B": B["Forma_5_Puntos"],
        "Cuota_A": A["Cuota_Ganar_Mundial"],
        "Cuota_B": B["Cuota_Ganar_Mundial"],
    }

    return resultado, resumen


# -------------------------------
# 4. Ranking de favoritos
# -------------------------------
def ranking_favoritos(df, top=10):
    columnas = [
        "Equipo", "Grupo", "Confederacion", "Perfil_Competitivo",
        "Ranking_FIFA", "Rating_Elo", "Valor_Mercado_M_EUR",
        "Forma_5_Puntos", "xG_Promedio", "xGA_Promedio",
        "Cuota_Ganar_Mundial", "Score_Poder"
    ]

    return df.sort_values("Score_Poder", ascending=False)[columnas].head(top)


# -------------------------------
# 5. Ejecución interactiva
# -------------------------------
if __name__ == "__main__":
    df = cargar_dataset(EXCEL_PATH)
    df = construir_scores(df)

    print("\nTOP 10 FAVORITOS SEGÚN SCORE DEL MODELO")
    print(ranking_favoritos(df, 10).to_string(index=False))

    print("\nEquipos disponibles:")
    print(", ".join(df["Equipo"].tolist()))

    equipo_a = input("\nIngresa selección A: ")
    equipo_b = input("Ingresa selección B: ")

    probs, resumen = predecir_partido(df, equipo_a, equipo_b)

    print("\nRESUMEN DEL CRUCE")
    for k, v in resumen.items():
        print(f"{k}: {v}")

    print("\nPROBABILIDADES")
    print(probs[["Resultado", "Probabilidad_%"]].to_string(index=False))
