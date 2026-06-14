from __future__ import annotations

import importlib.util
import math
import os
import re
import sys
import types
import zipfile
from functools import lru_cache
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request, send_from_directory


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PUBLIC_DIR = BASE_DIR / "public"
DEFAULT_EXTERNAL_DIR = Path(r"D:\5.Mundial 2026")
PCA_SHEET_NAME = "PCA y Clustering"

PREDICTOR_CANDIDATES = [
    os.getenv("MUNDIAL_PREDICTOR_PATH"),
    BASE_DIR / "predictor_mundial2026.py",
    DEFAULT_EXTERNAL_DIR / "predictor_mundial2026.py",
]

DATASET_CANDIDATES = [
    os.getenv("MUNDIAL_DATASET_PATH"),
    DATA_DIR / "mundial2026_ML_dataset.xlsx",
    BASE_DIR / "mundial2026_ML_dataset.xlsx",
    DEFAULT_EXTERNAL_DIR / "mundial2026_ML_dataset.xlsx",
]

app = Flask(__name__, static_folder=None)


DISPLAY_NAMES_ES = {
    "France": "Francia",
    "Spain": "España",
    "Argentina": "Argentina",
    "England": "Inglaterra",
    "Portugal": "Portugal",
    "Brazil": "Brasil",
    "Netherlands": "Países Bajos",
    "Morocco": "Marruecos",
    "Belgium": "Bélgica",
    "Germany": "Alemania",
    "Croatia": "Croacia",
    "Colombia": "Colombia",
    "Senegal": "Senegal",
    "Mexico": "México",
    "United States": "Estados Unidos",
    "Uruguay": "Uruguay",
    "Japan": "Japón",
    "Switzerland": "Suiza",
    "Iran": "Irán",
    "Austria": "Austria",
    "Ecuador": "Ecuador",
    "South Korea": "Corea del Sur",
    "Venezuela": "Venezuela",
    "Australia": "Australia",
    "Egypt": "Egipto",
    "Canada": "Canadá",
    "Ivory Coast": "Costa de Marfil",
    "Qatar": "Catar",
    "Algeria": "Argelia",
    "Sweden": "Suecia",
    "Tunisia": "Túnez",
    "Czechia": "Chequia",
    "Türkiye": "Turquía",
    "Norway": "Noruega",
    "Scotland": "Escocia",
    "DR Congo": "RD Congo",
    "Bosnia & Herzegovina": "Bosnia y Herzegovina",
    "Panama": "Panamá",
    "Saudi Arabia": "Arabia Saudita",
    "South Africa": "Sudáfrica",
    "Iraq": "Irak",
    "Ghana": "Ghana",
    "Paraguay": "Paraguay",
    "Jordan": "Jordania",
    "Uzbekistan": "Uzbekistán",
    "Cape Verde": "Cabo Verde",
    "Curaçao": "Curazao",
    "New Zealand": "Nueva Zelanda",
}

TEAM_FLAGS = {
    "France": "🇫🇷",
    "Spain": "🇪🇸",
    "Argentina": "🇦🇷",
    "England": "🇬🇧",
    "Portugal": "🇵🇹",
    "Brazil": "🇧🇷",
    "Netherlands": "🇳🇱",
    "Morocco": "🇲🇦",
    "Belgium": "🇧🇪",
    "Germany": "🇩🇪",
    "Croatia": "🇭🇷",
    "Colombia": "🇨🇴",
    "Senegal": "🇸🇳",
    "Mexico": "🇲🇽",
    "United States": "🇺🇸",
    "Uruguay": "🇺🇾",
    "Japan": "🇯🇵",
    "Switzerland": "🇨🇭",
    "Iran": "🇮🇷",
    "Austria": "🇦🇹",
    "Ecuador": "🇪🇨",
    "South Korea": "🇰🇷",
    "Venezuela": "🇻🇪",
    "Australia": "🇦🇺",
    "Egypt": "🇪🇬",
    "Canada": "🇨🇦",
    "Ivory Coast": "🇨🇮",
    "Qatar": "🇶🇦",
    "Algeria": "🇩🇿",
    "Sweden": "🇸🇪",
    "Tunisia": "🇹🇳",
    "Czechia": "🇨🇿",
    "Türkiye": "🇹🇷",
    "Norway": "🇳🇴",
    "Scotland": "🇬🇧",
    "DR Congo": "🇨🇩",
    "Bosnia & Herzegovina": "🇧🇦",
    "Panama": "🇵🇦",
    "Saudi Arabia": "🇸🇦",
    "South Africa": "🇿🇦",
    "Iraq": "🇮🇶",
    "Ghana": "🇬🇭",
    "Paraguay": "🇵🇾",
    "Jordan": "🇯🇴",
    "Uzbekistan": "🇺🇿",
    "Cape Verde": "🇨🇻",
    "Curaçao": "🇨🇼",
    "New Zealand": "🇳🇿",
}

PCA_CLUSTER_NAMES = ["Élite", "Candidatos", "Competitivos", "Outsiders"]
PCA_CLUSTER_COLORS = ["#35d486", "#58a6ff", "#f0c75e", "#ff7f5f"]

PCA_STRENGTH_METRICS = {
    "Rating_Elo_norm": "Elo",
    "Valor_Mercado_M_EUR_norm": "mercado",
    "Forma_5_Puntos_norm": "forma",
    "xG_Promedio_norm": "xG",
    "Tiros_Al_Arco_norm": "tiros al arco",
    "Posesion_Promedio_norm": "posesión",
    "Grandes_Ocasiones_Creadas_norm": "ocasiones creadas",
    "Dif_Gol_U10_norm": "diferencia de gol",
    "Puntos_U10_norm": "puntos recientes",
}


def _resolve_existing_path(candidates: list[str | Path | None], label: str) -> Path:
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        if path.exists():
            return path
    options = ", ".join(str(Path(c)) for c in candidates if c)
    raise FileNotFoundError(f"No se encontro {label}. Rutas revisadas: {options}")


def _install_minimal_sklearn_fallback() -> None:
    try:
        import sklearn.preprocessing  # noqa: F401

        return
    except ModuleNotFoundError:
        pass

    class StandardScaler:
        def fit_transform(self, values: Any) -> np.ndarray:
            arr = np.asarray(values, dtype=float)
            self.mean_ = np.nanmean(arr, axis=0)
            self.scale_ = np.nanstd(arr, axis=0)
            self.scale_ = np.where(self.scale_ == 0, 1, self.scale_)
            return (arr - self.mean_) / self.scale_

    sklearn_module = types.ModuleType("sklearn")
    preprocessing_module = types.ModuleType("sklearn.preprocessing")
    preprocessing_module.StandardScaler = StandardScaler
    sklearn_module.preprocessing = preprocessing_module
    sys.modules.setdefault("sklearn", sklearn_module)
    sys.modules.setdefault("sklearn.preprocessing", preprocessing_module)


def _load_predictor_module():
    _install_minimal_sklearn_fallback()
    predictor_path = _resolve_existing_path(PREDICTOR_CANDIDATES, "predictor_mundial2026.py")
    spec = importlib.util.spec_from_file_location("predictor_mundial2026", predictor_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"No se pudo importar el predictor desde {predictor_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["predictor_mundial2026"] = module
    spec.loader.exec_module(module)
    return module, predictor_path


def _read_shared_strings(zip_file: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in zip_file.namelist():
        return []

    root = ET.fromstring(zip_file.read("xl/sharedStrings.xml"))
    ns = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    strings: list[str] = []

    for item in root.findall("x:si", ns):
        parts = [node.text or "" for node in item.findall(".//x:t", ns)]
        strings.append("".join(parts))

    return strings


def _column_index(cell_reference: str) -> int:
    letters = re.sub(r"[^A-Z]", "", cell_reference.upper())
    index = 0
    for char in letters:
        index = index * 26 + (ord(char) - ord("A") + 1)
    return index - 1


def _coerce_xlsx_value(value: str | None, cell_type: str | None, shared_strings: list[str]) -> Any:
    if value is None:
        return None

    if cell_type == "s":
        return shared_strings[int(value)]

    if cell_type == "b":
        return value == "1"

    try:
        number = float(value)
        if number.is_integer():
            return int(number)
        return number
    except ValueError:
        return value


def _worksheet_path(zip_file: zipfile.ZipFile, sheet_name: str) -> str:
    wb_root = ET.fromstring(zip_file.read("xl/workbook.xml"))
    rels_root = ET.fromstring(zip_file.read("xl/_rels/workbook.xml.rels"))

    ns = {
        "x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
        "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
        "rel": "http://schemas.openxmlformats.org/package/2006/relationships",
    }

    relation_id = None
    for sheet in wb_root.findall("x:sheets/x:sheet", ns):
        if sheet.attrib.get("name") == sheet_name:
            relation_id = sheet.attrib.get(f"{{{ns['r']}}}id")
            break

    if relation_id is None:
        raise ValueError(f"No existe la hoja '{sheet_name}' en el Excel")

    for relationship in rels_root.findall("rel:Relationship", ns):
        if relationship.attrib.get("Id") == relation_id:
            target = relationship.attrib["Target"].lstrip("/")
            return target if target.startswith("xl/") else f"xl/{target}"

    raise ValueError(f"No se encontro la relacion interna para la hoja '{sheet_name}'")


def _read_xlsx_without_openpyxl(path: str | Path, sheet_name: str, header: int = 0) -> pd.DataFrame:
    ns = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}

    with zipfile.ZipFile(path) as zip_file:
        shared_strings = _read_shared_strings(zip_file)
        worksheet = _worksheet_path(zip_file, sheet_name)
        root = ET.fromstring(zip_file.read(worksheet))

    rows: list[list[Any]] = []
    for row in root.findall(".//x:sheetData/x:row", ns):
        values: list[Any] = []
        for cell in row.findall("x:c", ns):
            index = _column_index(cell.attrib.get("r", "A1"))
            while len(values) <= index:
                values.append(None)

            value_node = cell.find("x:v", ns)
            inline_node = cell.find("x:is/x:t", ns)
            raw_value = value_node.text if value_node is not None else inline_node.text if inline_node is not None else None
            values[index] = _coerce_xlsx_value(raw_value, cell.attrib.get("t"), shared_strings)

        rows.append(values)

    width = max((len(row) for row in rows), default=0)
    normalized_rows = [row + [None] * (width - len(row)) for row in rows]
    header_values = normalized_rows[header]
    data_rows = normalized_rows[header + 1 :]

    return pd.DataFrame(data_rows, columns=header_values)


def _load_dataset_with_predictor(predictor, dataset_path: Path) -> pd.DataFrame:
    try:
        return predictor.cargar_dataset(str(dataset_path))
    except ImportError as exc:
        if "openpyxl" not in str(exc):
            raise

        original_read_excel = predictor.pd.read_excel

        def read_excel_fallback(path: str | Path, sheet_name: str | None = None, header: int = 0, **_: Any):
            return _read_xlsx_without_openpyxl(path, sheet_name or "Dataset Principal", header=header)

        predictor.pd.read_excel = read_excel_fallback
        try:
            return predictor.cargar_dataset(str(dataset_path))
        finally:
            predictor.pd.read_excel = original_read_excel


def _read_excel_sheet(path: str | Path, sheet_name: str, header: int = 0) -> pd.DataFrame:
    try:
        return pd.read_excel(path, sheet_name=sheet_name, header=header)
    except ImportError as exc:
        if "openpyxl" not in str(exc):
            raise
        return _read_xlsx_without_openpyxl(path, sheet_name, header=header)


def _load_pca_sheet(dataset_path: Path) -> pd.DataFrame:
    df = _read_excel_sheet(dataset_path, PCA_SHEET_NAME, header=1)
    df = df.dropna(axis=1, how="all")
    df.columns = [str(c).strip() for c in df.columns]
    return df[df["Equipo"].notna()].copy()


def _run_kmeans(values: np.ndarray, clusters: int = 4, max_iter: int = 80) -> np.ndarray:
    if len(values) == 0:
        return np.array([], dtype=int)

    clusters = max(1, min(clusters, len(values)))
    order = np.argsort(values[:, 0])
    seed_positions = np.linspace(0, len(values) - 1, clusters).round().astype(int)
    centers = values[order[seed_positions]].copy()
    labels = np.zeros(len(values), dtype=int)

    for _ in range(max_iter):
        distances = np.linalg.norm(values[:, None, :] - centers[None, :, :], axis=2)
        new_labels = distances.argmin(axis=1)

        if np.array_equal(labels, new_labels):
            break

        labels = new_labels
        for cluster_id in range(clusters):
            members = values[labels == cluster_id]
            if len(members):
                centers[cluster_id] = members.mean(axis=0)
            else:
                farthest = np.argmax(distances.min(axis=1))
                centers[cluster_id] = values[farthest]

    return labels


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 2:
        return 0.0
    if np.std(a[mask]) == 0 or np.std(b[mask]) == 0:
        return 0.0
    return float(np.corrcoef(a[mask], b[mask])[0, 1])


def _pca_strengths(row: pd.Series) -> str:
    available = []
    for column, label in PCA_STRENGTH_METRICS.items():
        value = clean_scalar(row.get(column))
        if value is not None:
            available.append((float(value), label))

    if not available:
        return "perfil balanceado"

    top = [label for _, label in sorted(available, reverse=True)[:3]]
    return ", ".join(top)


def pca_map_payload(pca_df: pd.DataFrame, model_df: pd.DataFrame) -> dict[str, Any]:
    numeric_cols = [col for col in pca_df.columns if col.endswith("_norm")]
    if len(numeric_cols) < 2:
        return {"points": [], "clusters": [], "summary": {"error": "No hay suficientes variables normalizadas."}}

    merged = pca_df.merge(
        model_df[
            [
                "Equipo",
                "Ranking_FIFA",
                "Rating_Elo",
                "Valor_Mercado_M_EUR",
                "Forma_5_Puntos",
                "Forma_5_Partidos",
                "xG_Promedio",
                "xGA_Promedio",
                "Lesiones_Importantes",
                "Score_Poder",
            ]
        ],
        on="Equipo",
        how="left",
    )

    X = pca_df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median(numeric_only=True)).fillna(0.0)
    centered = X.to_numpy(dtype=float)
    centered = centered - centered.mean(axis=0)

    _, singular_values, components = np.linalg.svd(centered, full_matrices=False)
    coordinates = centered @ components[:2].T

    score_values = pd.to_numeric(merged["Score_Poder"], errors="coerce").to_numpy(dtype=float)
    if _safe_corr(coordinates[:, 0], score_values) < 0:
        coordinates[:, 0] *= -1

    profile_signal = (
        pd.to_numeric(pca_df.get("xG_Promedio_norm"), errors="coerce").fillna(0).to_numpy(dtype=float)
        + pd.to_numeric(pca_df.get("Tiros_Al_Arco_norm"), errors="coerce").fillna(0).to_numpy(dtype=float)
        - pd.to_numeric(pca_df.get("xGA_Promedio_norm"), errors="coerce").fillna(0).to_numpy(dtype=float)
    )
    if _safe_corr(coordinates[:, 1], profile_signal) < 0:
        coordinates[:, 1] *= -1

    explained = singular_values**2
    explained_ratio = explained / explained.sum() if explained.sum() else np.zeros_like(explained)
    raw_labels = _run_kmeans(coordinates[:, :2], clusters=4)

    cluster_scores = []
    for cluster_id in sorted(set(raw_labels.tolist())):
        cluster_score = np.nanmean(score_values[raw_labels == cluster_id])
        cluster_scores.append((cluster_id, cluster_score if np.isfinite(cluster_score) else -999))

    cluster_order = {
        cluster_id: rank
        for rank, (cluster_id, _) in enumerate(sorted(cluster_scores, key=lambda item: item[1], reverse=True))
    }

    score_min = float(np.nanmin(score_values))
    score_max = float(np.nanmax(score_values))
    score_spread = max(score_max - score_min, 1e-9)
    score_ranks = model_df["Score_Poder"].rank(ascending=False, method="min").astype(int)
    rank_lookup = dict(zip(model_df["Equipo"], score_ranks))

    points = []
    for index, row in merged.reset_index(drop=True).iterrows():
        team = str(row["Equipo"])
        cluster_rank = cluster_order[int(raw_labels[index])]
        score = float(clean_scalar(row.get("Score_Poder")) or 0.0)
        score_scaled = (score - score_min) / score_spread

        points.append(
            {
                "team": team,
                "display_name": display_name(team),
                "flag": team_flag(team),
                "confederation": str(clean_scalar(row.get("Confederacion")) or "N/D"),
                "profile": str(clean_scalar(row.get("Perfil_Competitivo")) or "N/D"),
                "cluster": cluster_rank,
                "cluster_label": PCA_CLUSTER_NAMES[cluster_rank],
                "cluster_color": PCA_CLUSTER_COLORS[cluster_rank],
                "pc1": round(float(coordinates[index, 0]), 4),
                "pc2": round(float(coordinates[index, 1]), 4),
                "marker_size": round(10 + score_scaled * 18, 2),
                "score": round(score, 3),
                "model_rank": int(rank_lookup.get(team, index + 1)),
                "fifa_rank": format_int(row.get("Ranking_FIFA")),
                "elo": format_int(row.get("Rating_Elo")),
                "market": format_market(row.get("Valor_Mercado_M_EUR")),
                "form": format_form(row),
                "xg": format_decimal(row.get("xG_Promedio"), 2),
                "xga": format_decimal(row.get("xGA_Promedio"), 2),
                "injuries": str(clean_scalar(row.get("Lesiones_Importantes")) or "Sin bajas relevantes"),
                "strengths": _pca_strengths(row),
            }
        )

    clusters_payload = []
    for cluster_rank, label in enumerate(PCA_CLUSTER_NAMES):
        members = [point for point in points if point["cluster"] == cluster_rank]
        if not members:
            continue
        avg_score = sum(point["score"] for point in members) / len(members)
        clusters_payload.append(
            {
                "id": cluster_rank,
                "label": label,
                "color": PCA_CLUSTER_COLORS[cluster_rank],
                "count": len(members),
                "avg_score": round(avg_score, 3),
            }
        )

    strongest = sorted(points, key=lambda item: item["score"], reverse=True)[:3]
    summary = {
        "pc1_variance": round(float(explained_ratio[0] * 100), 1),
        "pc2_variance": round(float(explained_ratio[1] * 100), 1),
        "variables": len(numeric_cols),
        "strongest": ", ".join(point["display_name"] for point in strongest),
    }

    return {"points": points, "clusters": clusters_payload, "summary": summary}


@lru_cache(maxsize=1)
def get_model_context() -> dict[str, Any]:
    predictor, predictor_path = _load_predictor_module()
    dataset_path = _resolve_existing_path(DATASET_CANDIDATES, "mundial2026_ML_dataset.xlsx")

    df = _load_dataset_with_predictor(predictor, dataset_path)
    df = predictor.construir_scores(df)

    teams = sorted(
        [team_payload(team) for team in df["Equipo"].dropna().astype(str).tolist()],
        key=lambda item: item["display_name"],
    )

    ranking_df = predictor.ranking_favoritos(df, 10)
    pca_df = _load_pca_sheet(dataset_path)
    pca_map = pca_map_payload(pca_df, df)

    return {
        "predictor": predictor,
        "predictor_path": predictor_path,
        "dataset_path": dataset_path,
        "df": df,
        "teams": teams,
        "ranking": ranking_payload(ranking_df, df),
        "pca_map": pca_map,
    }


def display_name(team: str) -> str:
    return DISPLAY_NAMES_ES.get(team, team)


def team_flag(team: str) -> str:
    return TEAM_FLAGS.get(team, "🏳️")


def team_payload(team: str) -> dict[str, str]:
    return {
        "id": team,
        "name": team,
        "display_name": display_name(team),
        "flag": team_flag(team),
    }


def clean_scalar(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def format_decimal(value: Any, digits: int = 1) -> str:
    value = clean_scalar(value)
    if value is None:
        return "N/D"
    return f"{float(value):,.{digits}f}"


def format_int(value: Any) -> str:
    value = clean_scalar(value)
    if value is None:
        return "N/D"
    return f"{int(round(float(value))):,}"


def format_market(value: Any) -> str:
    value = clean_scalar(value)
    if value is None:
        return "N/D"

    amount = float(value)
    if amount >= 1000:
        return f"€{amount / 1000:.2f}B"
    return f"€{amount:.0f}M"


def format_form(row: pd.Series) -> str:
    form = clean_scalar(row.get("Forma_5_Partidos"))
    points = clean_scalar(row.get("Forma_5_Puntos"))
    if form is None and points is None:
        return "N/D"
    if form is None:
        return f"{format_decimal(points, 0)} pts"
    return f"{form} · {format_decimal(points, 0)} pts"


def injury_count(value: Any) -> int:
    value = clean_scalar(value)
    if value is None:
        return 0
    text = str(value).strip().lower()
    if text in {"", "0", "nan", "none", "ninguna", "sin lesiones", "sin bajas", "n/d"}:
        return 0
    return max(1, len([part for part in re.split(r"[,;/]", text) if part.strip()]))


def advantage_side(value_a: Any, value_b: Any, higher_is_better: bool = True) -> str:
    a = clean_scalar(value_a)
    b = clean_scalar(value_b)
    if a is None or b is None:
        return "neutral"

    a_float = float(a)
    b_float = float(b)
    if math.isclose(a_float, b_float, rel_tol=0.005, abs_tol=0.005):
        return "neutral"

    if higher_is_better:
        return "home" if a_float > b_float else "away"
    return "home" if a_float < b_float else "away"


def comparison_payload(row_a: pd.Series, row_b: pd.Series) -> list[dict[str, Any]]:
    injuries_a = injury_count(row_a.get("Lesiones_Importantes"))
    injuries_b = injury_count(row_b.get("Lesiones_Importantes"))

    return [
        {
            "metric": "Ranking FIFA",
            "home": format_int(row_a.get("Ranking_FIFA")),
            "away": format_int(row_b.get("Ranking_FIFA")),
            "side": advantage_side(row_a.get("Ranking_FIFA"), row_b.get("Ranking_FIFA"), higher_is_better=False),
        },
        {
            "metric": "Rating Elo",
            "home": format_int(row_a.get("Rating_Elo")),
            "away": format_int(row_b.get("Rating_Elo")),
            "side": advantage_side(row_a.get("Rating_Elo"), row_b.get("Rating_Elo")),
        },
        {
            "metric": "Valor de Mercado",
            "home": format_market(row_a.get("Valor_Mercado_M_EUR")),
            "away": format_market(row_b.get("Valor_Mercado_M_EUR")),
            "side": advantage_side(row_a.get("Valor_Mercado_M_EUR"), row_b.get("Valor_Mercado_M_EUR")),
        },
        {
            "metric": "Forma reciente",
            "home": format_form(row_a),
            "away": format_form(row_b),
            "side": advantage_side(row_a.get("Forma_5_Puntos"), row_b.get("Forma_5_Puntos")),
        },
        {
            "metric": "xG",
            "home": format_decimal(row_a.get("xG_Promedio"), 2),
            "away": format_decimal(row_b.get("xG_Promedio"), 2),
            "side": advantage_side(row_a.get("xG_Promedio"), row_b.get("xG_Promedio")),
        },
        {
            "metric": "xGA",
            "home": format_decimal(row_a.get("xGA_Promedio"), 2),
            "away": format_decimal(row_b.get("xGA_Promedio"), 2),
            "side": advantage_side(row_a.get("xGA_Promedio"), row_b.get("xGA_Promedio"), higher_is_better=False),
        },
        {
            "metric": "Lesiones",
            "home": str(clean_scalar(row_a.get("Lesiones_Importantes")) or "Sin bajas relevantes"),
            "away": str(clean_scalar(row_b.get("Lesiones_Importantes")) or "Sin bajas relevantes"),
            "side": "neutral"
            if injuries_a == injuries_b
            else "home"
            if injuries_a < injuries_b
            else "away",
        },
    ]


def row_for_team(df: pd.DataFrame, team: str) -> pd.Series:
    matches = df[df["Equipo"].str.lower() == team.lower()]
    if matches.empty:
        raise ValueError(f"No se encontro el equipo: {team}")
    return matches.iloc[0]


def interpretation(
    row_a: pd.Series,
    row_b: pd.Series,
    p_home: float,
    p_draw: float,
    p_away: float,
    diff_score: float,
) -> str:
    name_a = display_name(str(row_a["Equipo"]))
    name_b = display_name(str(row_b["Equipo"]))

    if abs(p_home - p_away) < 0.04:
        favorite_side = "neutral"
        opener = f"{name_a} y {name_b} llegan prácticamente equilibrados"
    else:
        favorite_side = "home" if p_home > p_away else "away"
        favorite = name_a if p_home > p_away else name_b
        rival = name_b if favorite == name_a else name_a
        margin = "ligera" if abs(diff_score) < 0.45 else "clara"
        opener = f"{favorite} presenta una ventaja {margin} sobre {rival}"

    factors: list[str] = []
    if clean_scalar(row_a.get("Rating_Elo")) and clean_scalar(row_b.get("Rating_Elo")):
        elo_side = advantage_side(row_a.get("Rating_Elo"), row_b.get("Rating_Elo"))
        if elo_side == "home" and favorite_side in {"home", "neutral"}:
            factors.append(f"mejor Rating Elo de {name_a}")
        elif elo_side == "away" and favorite_side in {"away", "neutral"}:
            factors.append(f"mejor Rating Elo de {name_b}")

    if clean_scalar(row_a.get("xG_Promedio")) and clean_scalar(row_b.get("xG_Promedio")):
        xg_side = advantage_side(row_a.get("xG_Promedio"), row_b.get("xG_Promedio"))
        if xg_side == "home" and favorite_side in {"home", "neutral"}:
            factors.append(f"mayor volumen ofensivo de {name_a}")
        elif xg_side == "away" and favorite_side in {"away", "neutral"}:
            factors.append(f"mayor volumen ofensivo de {name_b}")

    xga_side = advantage_side(row_a.get("xGA_Promedio"), row_b.get("xGA_Promedio"), higher_is_better=False)
    if xga_side == "home" and favorite_side in {"home", "neutral"}:
        factors.append(f"defensa más eficiente de {name_a}")
    elif xga_side == "away" and favorite_side in {"away", "neutral"}:
        factors.append(f"defensa más eficiente de {name_b}")

    form_side = advantage_side(row_a.get("Forma_5_Puntos"), row_b.get("Forma_5_Puntos"))
    if form_side == "home" and favorite_side in {"home", "neutral"}:
        factors.append(f"mejor forma reciente de {name_a}")
    elif form_side == "away" and favorite_side in {"away", "neutral"}:
        factors.append(f"mejor forma reciente de {name_b}")

    if p_draw >= 0.27:
        factors.append("alta probabilidad de empate por cercanía estadística")

    if not factors:
        return f"{opener}. El Score de Poder integrado compensa métricas puntuales donde el rival también es fuerte."

    return f"{opener} gracias a {', '.join(factors[:3])}."


def probability_payload(probs: pd.DataFrame, team_a: str, team_b: str) -> list[dict[str, Any]]:
    p_home = float(probs.iloc[0]["Probabilidad_%"])
    p_draw = float(probs.iloc[1]["Probabilidad_%"])
    p_away = float(probs.iloc[2]["Probabilidad_%"])

    return [
        {
            "key": "home",
            "label": display_name(team_a),
            "short_label": display_name(team_a),
            "flag": team_flag(team_a),
            "percent": p_home,
        },
        {
            "key": "draw",
            "label": "Empate",
            "short_label": "Empate",
            "flag": "⚖️",
            "percent": p_draw,
        },
        {
            "key": "away",
            "label": display_name(team_b),
            "short_label": display_name(team_b),
            "flag": team_flag(team_b),
            "percent": p_away,
        },
    ]


def ranking_payload(ranking_df: pd.DataFrame, full_df: pd.DataFrame) -> list[dict[str, Any]]:
    min_score = float(full_df["Score_Poder"].min())
    max_score = float(full_df["Score_Poder"].max())
    spread = max(max_score - min_score, 1e-9)

    payload = []
    for index, row in ranking_df.reset_index(drop=True).iterrows():
        score = float(row["Score_Poder"])
        payload.append(
            {
                "position": index + 1,
                "team": str(row["Equipo"]),
                "display_name": display_name(str(row["Equipo"])),
                "flag": team_flag(str(row["Equipo"])),
                "score": round(score, 3),
                "progress": round(((score - min_score) / spread) * 100, 1),
                "profile": str(clean_scalar(row.get("Perfil_Competitivo")) or "N/D"),
            }
        )
    return payload


@app.get("/")
def index():
    load_error = None
    bootstrap_data: dict[str, Any] = {"teams": [], "ranking": []}
    try:
        context = get_model_context()
        bootstrap_data = {
            "teams": context["teams"],
            "ranking": context["ranking"],
            "pcaMap": context["pca_map"],
            "defaultHome": "Spain",
            "defaultAway": "France",
            "datasetPath": str(context["dataset_path"]),
            "predictorPath": str(context["predictor_path"]),
        }
    except Exception as exc:  # pragma: no cover - surfaced in UI for local setup issues
        load_error = str(exc)

    return render_template(
        "index.html",
        bootstrap_data=bootstrap_data,
        teams=bootstrap_data["teams"],
        ranking=bootstrap_data["ranking"],
        load_error=load_error,
    )


@app.get("/css/<path:filename>")
def public_css(filename: str):
    return send_from_directory(PUBLIC_DIR / "css", filename)


@app.get("/js/<path:filename>")
def public_js(filename: str):
    return send_from_directory(PUBLIC_DIR / "js", filename)


@app.get("/api/teams")
def api_teams():
    context = get_model_context()
    return jsonify(context["teams"])


@app.get("/api/pca-map")
def api_pca_map():
    context = get_model_context()
    return jsonify(context["pca_map"])


@app.post("/api/predict")
def api_predict():
    payload = request.get_json(silent=True) or {}
    team_a = str(payload.get("team_a", "")).strip()
    team_b = str(payload.get("team_b", "")).strip()

    if not team_a or not team_b:
        return jsonify({"error": "Selecciona dos selecciones para calcular el cruce."}), 400

    if team_a.lower() == team_b.lower():
        return jsonify({"error": "Elige dos selecciones diferentes."}), 400

    try:
        context = get_model_context()
        df = context["df"]
        predictor = context["predictor"]

        probs, summary = predictor.predecir_partido(df, team_a, team_b)
        row_a = row_for_team(df, team_a)
        row_b = row_for_team(df, team_b)
        probabilities = probability_payload(probs, str(row_a["Equipo"]), str(row_b["Equipo"]))
        p_home, p_draw, p_away = [item["percent"] / 100 for item in probabilities]

        response = {
            "match": {
                "home": team_payload(str(row_a["Equipo"])),
                "away": team_payload(str(row_b["Equipo"])),
                "title": f"{display_name(str(row_a['Equipo']))} vs {display_name(str(row_b['Equipo']))}",
            },
            "probabilities": probabilities,
            "scores": {
                "home": round(float(summary["Score_A"]), 3),
                "away": round(float(summary["Score_B"]), 3),
                "difference": round(float(summary["Diferencia_Score"]), 3),
            },
            "profiles": {
                "home": str(summary.get("Perfil_A", "N/D")),
                "away": str(summary.get("Perfil_B", "N/D")),
            },
            "comparison": comparison_payload(row_a, row_b),
            "interpretation": interpretation(
                row_a,
                row_b,
                p_home,
                p_draw,
                p_away,
                float(summary["Diferencia_Score"]),
            ),
        }
        return jsonify(response)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 404
    except Exception as exc:  # pragma: no cover - returned for local diagnostics
        return jsonify({"error": f"No se pudo calcular el partido: {exc}"}), 500


if __name__ == "__main__":
    debug_mode = (
        not os.getenv("VERCEL")
        and os.getenv("FLASK_DEBUG", "1").lower() not in {"0", "false", "no"}
    )
    app.run(debug=debug_mode, use_reloader=False)
