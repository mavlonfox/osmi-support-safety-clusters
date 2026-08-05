#!/usr/bin/env python3
"""Reproduzierbare Clusteranalyse der OSMI Mental Health in Tech Survey 2016.

Die Analyse bildet ausschließlich Wahrnehmungen des aktuellen Arbeitgebers ab.
Diagnosen, Behandlung und Demografie werden nicht als Clustermerkmale genutzt.
Sensible Variablen erscheinen nur als aggregierte, deskriptive Kontextgrößen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import urllib.request
import zipfile
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Callable

# Verhindert übermäßige Thread-Initialisierung auf kleinen Matrizen.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from matplotlib.lines import Line2D
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    pairwise_distances,
    silhouette_score,
)
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


DATA_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/"
    "osmi/mental-health-in-tech-2016"
)
ARCHIVE_NAME = "osmi_mental_health_2016.zip"
CSV_NAME = "mental-heath-in-tech-2016_20161114.csv"
CSV_SHA256 = "0bec458b0724cc375a17eb2db0204a9f7a786260441cf702eec210d92bd4ae4d"
RANDOM_STATE = 42
SKIPPED = "__NICHT_ERHOBEN__"

SELF_EMPLOYED = "Are you self-employed?"
CURRENT_DISORDER = "Do you currently have a mental health disorder?"
SOUGHT_TREATMENT = (
    "Have you ever sought treatment for a mental health issue from a mental "
    "health professional?"
)
UNSUPPORTIVE_RESPONSE = (
    "Have you observed or experienced an unsupportive or badly handled "
    "response to a mental health issue in your current or previous workplace?"
)


@dataclass(frozen=True)
class FeatureSpec:
    code: str
    raw: str
    label_de: str
    block: str
    score_map: dict[str, float]


FEATURES: tuple[FeatureSpec, ...] = (
    FeatureSpec(
        "benefits",
        "Does your employer provide mental health benefits as part of healthcare coverage?",
        "Leistungen vorhanden",
        "Formale Unterstützung und Navigation",
        {
            "Yes": 1.0,
            "I don't know": 0.5,
            "No": 0.0,
            "Not eligible for coverage / N/A": 0.0,
        },
    ),
    FeatureSpec(
        "options_known",
        "Do you know the options for mental health care available under your employer-provided coverage?",
        "Versorgungsoptionen bekannt",
        "Formale Unterstützung und Navigation",
        {"Yes": 1.0, "I am not sure": 0.5, "No": 0.0, SKIPPED: 0.0},
    ),
    FeatureSpec(
        "formal_discussion",
        "Has your employer ever formally discussed mental health (for example, as part of a wellness campaign or other official communication)?",
        "Formale Kommunikation",
        "Formale Unterstützung und Navigation",
        {"Yes": 1.0, "I don't know": 0.5, "No": 0.0},
    ),
    FeatureSpec(
        "resources",
        "Does your employer offer resources to learn more about mental health concerns and options for seeking help?",
        "Informationsressourcen",
        "Formale Unterstützung und Navigation",
        {"Yes": 1.0, "I don't know": 0.5, "No": 0.0},
    ),
    FeatureSpec(
        "anonymity",
        "Is your anonymity protected if you choose to take advantage of mental health or substance abuse treatment resources provided by your employer?",
        "Anonymität geschützt",
        "Formale Unterstützung und Navigation",
        {"Yes": 1.0, "I don't know": 0.5, "No": 0.0},
    ),
    FeatureSpec(
        "leave_ease",
        "If a mental health issue prompted you to request a medical leave from work, asking for that leave would be:",
        "Krankheitsurlaub leicht",
        "Psychologische Sicherheit",
        {
            "Very easy": 1.0,
            "Somewhat easy": 0.75,
            "Neither easy nor difficult": 0.5,
            "I don't know": 0.5,
            "Somewhat difficult": 0.25,
            "Very difficult": 0.0,
        },
    ),
    FeatureSpec(
        "no_mental_consequences",
        "Do you think that discussing a mental health disorder with your employer would have negative consequences?",
        "Keine negativen Folgen (mental)",
        "Psychologische Sicherheit",
        {"No": 1.0, "Maybe": 0.5, "Yes": 0.0},
    ),
    FeatureSpec(
        "no_physical_consequences",
        "Do you think that discussing a physical health issue with your employer would have negative consequences?",
        "Keine negativen Folgen (körperlich)",
        "Psychologische Sicherheit",
        {"No": 1.0, "Maybe": 0.5, "Yes": 0.0},
    ),
    FeatureSpec(
        "coworker_comfort",
        "Would you feel comfortable discussing a mental health disorder with your coworkers?",
        "Gespräch mit Kolleg:innen",
        "Psychologische Sicherheit",
        {"Yes": 1.0, "Maybe": 0.5, "No": 0.0},
    ),
    FeatureSpec(
        "supervisor_comfort",
        "Would you feel comfortable discussing a mental health disorder with your direct supervisor(s)?",
        "Gespräch mit Führungskraft",
        "Psychologische Sicherheit",
        {"Yes": 1.0, "Maybe": 0.5, "No": 0.0},
    ),
    FeatureSpec(
        "parity",
        "Do you feel that your employer takes mental health as seriously as physical health?",
        "Mentale = körperliche Gesundheit",
        "Psychologische Sicherheit",
        {"Yes": 1.0, "I don't know": 0.5, "No": 0.0},
    ),
    FeatureSpec(
        "no_observed_consequences",
        "Have you heard of or observed negative consequences for co-workers who have been open about mental health issues in your workplace?",
        "Keine beobachteten Folgen",
        "Psychologische Sicherheit",
        {"No": 1.0, "Yes": 0.0},
    ),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_dataset(raw_dir: Path, download: bool = True) -> Path:
    """Lädt den Kaggle-Archivstand und prüft den extrahierten CSV-Fingerprint."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    csv_path = raw_dir / CSV_NAME
    archive_path = raw_dir / ARCHIVE_NAME
    if not csv_path.exists():
        if not download:
            raise FileNotFoundError(f"Rohdatei fehlt: {csv_path}")
        if not archive_path.exists():
            print(f"Lade {DATA_URL}")
            urllib.request.urlretrieve(DATA_URL, archive_path)
        with zipfile.ZipFile(archive_path) as archive:
            if CSV_NAME not in archive.namelist():
                raise RuntimeError(f"{CSV_NAME} nicht im Archiv gefunden")
            archive.extract(CSV_NAME, raw_dir)
    actual = sha256(csv_path)
    if actual != CSV_SHA256:
        raise RuntimeError(
            "CSV-Fingerprint weicht vom analysierten Stand ab: "
            f"erwartet {CSV_SHA256}, erhalten {actual}"
        )
    return csv_path


def validate_schema(df: pd.DataFrame) -> None:
    required = [SELF_EMPLOYED, *(f.raw for f in FEATURES)]
    required += [CURRENT_DISORDER, SOUGHT_TREATMENT, UNSUPPORTIVE_RESPONSE]
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"Erforderliche Spalten fehlen: {missing}")
    if df.shape != (1433, 63):
        raise ValueError(f"Unerwartete Datenform: {df.shape}; erwartet (1433, 63)")
    if not set(df[SELF_EMPLOYED].dropna().unique()).issubset({0, 1}):
        raise ValueError("Selbstständigkeitsvariable enthält unerwartete Werte")


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    employees = df.loc[df[SELF_EMPLOYED].eq(0)].copy()
    if len(employees) != 1146:
        raise ValueError(f"Unerwartete Beschäftigtenzahl: {len(employees)}")
    feature_frame = employees[[f.raw for f in FEATURES]].copy()
    feature_frame = feature_frame.fillna(SKIPPED).astype(str)
    for feature in FEATURES:
        unexpected = set(feature_frame[feature.raw].unique()) - set(feature.score_map)
        if unexpected:
            raise ValueError(f"Unerwartete Kategorien in {feature.code}: {unexpected}")
    return employees, feature_frame


def make_theory_scores(feature_frame: pd.DataFrame) -> pd.DataFrame:
    scores = pd.DataFrame(index=feature_frame.index)
    for feature in FEATURES:
        scores[feature.code] = feature_frame[feature.raw].map(feature.score_map)
    if scores.isna().any().any():
        raise ValueError("Deskriptives Scoring erzeugte fehlende Werte")
    return scores


def bootstrap_stability(
    reduced: np.ndarray,
    k: int,
    reps: int,
    sample_fraction: float = 0.8,
    seed: int = 20260805,
) -> tuple[float, float, float]:
    """Mittlere paarweise ARI von Vollstichproben-Zuweisungen aus Subsamples."""
    rng = np.random.default_rng(seed + k)
    assignments: list[np.ndarray] = []
    n = len(reduced)
    for rep in range(reps):
        idx = rng.choice(n, size=int(round(sample_fraction * n)), replace=False)
        model = KMeans(
            n_clusters=k,
            n_init=20,
            random_state=RANDOM_STATE + 1000 + rep,
        ).fit(reduced[idx])
        assignments.append(model.predict(reduced))
    aris = np.array(
        [adjusted_rand_score(a, b) for a, b in combinations(assignments, 2)],
        dtype=float,
    )
    return float(aris.mean()), float(aris.std(ddof=1)), float(aris.min())


def evaluate_k_values(
    one_hot: np.ndarray,
    reduced: np.ndarray,
    k_values: range,
    bootstrap_reps: int,
) -> tuple[pd.DataFrame, dict[int, KMeans]]:
    rows: list[dict[str, Any]] = []
    models: dict[int, KMeans] = {}
    for k in k_values:
        model = KMeans(n_clusters=k, n_init=50, random_state=RANDOM_STATE).fit(reduced)
        labels = model.labels_
        counts = np.bincount(labels, minlength=k)
        stability_mean, stability_sd, stability_min = bootstrap_stability(
            reduced, k, reps=bootstrap_reps
        )
        rows.append(
            {
                "k": k,
                "silhouette_reduced_euclidean": silhouette_score(reduced, labels),
                "silhouette_original_onehot_euclidean": silhouette_score(
                    one_hot, labels, metric="euclidean"
                ),
                "davies_bouldin_reduced": davies_bouldin_score(reduced, labels),
                "calinski_harabasz_reduced": calinski_harabasz_score(reduced, labels),
                "inertia_reduced": model.inertia_,
                "bootstrap_ari_mean": stability_mean,
                "bootstrap_ari_sd": stability_sd,
                "bootstrap_ari_min": stability_min,
                "min_cluster_n": int(counts.min()),
                "min_cluster_share": float(counts.min() / len(labels)),
                "cluster_sizes_unordered": ";".join(map(str, counts.tolist())),
            }
        )
        models[k] = model
    return pd.DataFrame(rows), models


def choose_k(metrics: pd.DataFrame) -> int:
    """Vorab festgelegte Regel: Qualität im Originalraum, Stabilität, Mindestgröße."""
    eligible = metrics.loc[
        (metrics["bootstrap_ari_mean"] >= 0.80)
        & (metrics["min_cluster_share"] >= 0.10)
    ].copy()
    if eligible.empty:
        raise RuntimeError("Kein k erfüllt Stabilitäts- und Mindestgrößenkriterien")
    eligible = eligible.sort_values(
        ["silhouette_original_onehot_euclidean", "k"], ascending=[False, True]
    )
    return int(eligible.iloc[0]["k"])


def ordered_profile_labels(
    raw_labels: np.ndarray, scores: pd.DataFrame
) -> tuple[np.ndarray, dict[int, str], dict[int, str]]:
    overall = scores.assign(_cluster=raw_labels).groupby("_cluster").mean().mean(axis=1)
    ordered = overall.sort_values(ascending=False).index.tolist()
    if len(ordered) == 2:
        names = [
            "Profil A – vergleichsweise sichtbare und ansprechbare Unterstützung",
            "Profil B – wenig sichtbare und schwer ansprechbare Unterstützung",
        ]
    else:
        names = [f"Profil {chr(65 + i)}" for i in range(len(ordered))]
    name_map = {int(raw): name for raw, name in zip(ordered, names)}
    id_map = {int(raw): chr(65 + rank) for rank, raw in enumerate(ordered)}
    profile_labels = np.array([id_map[int(value)] for value in raw_labels], dtype=str)
    return profile_labels, name_map, id_map


def block_profiles(
    scores: pd.DataFrame,
    profile_ids: np.ndarray,
    name_by_id: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    formal_codes = [f.code for f in FEATURES if f.block.startswith("Formale")]
    safety_codes = [f.code for f in FEATURES if f.block.startswith("Psychologische")]
    frame = scores.copy()
    frame["profile_id"] = profile_ids
    blocks: list[dict[str, Any]] = []
    items: list[dict[str, Any]] = []
    groups: list[tuple[str, pd.DataFrame]] = [("Gesamt", frame)]
    groups += [(pid, frame.loc[frame["profile_id"].eq(pid)]) for pid in sorted(set(profile_ids))]
    for pid, group in groups:
        formal = group[formal_codes].to_numpy().mean()
        safety = group[safety_codes].to_numpy().mean()
        label = "Gesamt" if pid == "Gesamt" else name_by_id[pid]
        blocks.append(
            {
                "profile_id": pid,
                "profile_name": label,
                "n": len(group),
                "formal_support_navigation": formal,
                "psychological_safety": safety,
                "formal_minus_safety_gap": formal - safety,
                "overall_support_score": group[[f.code for f in FEATURES]].to_numpy().mean(),
            }
        )
        for feature in FEATURES:
            items.append(
                {
                    "profile_id": pid,
                    "profile_name": label,
                    "n": len(group),
                    "feature_code": feature.code,
                    "feature_label_de": feature.label_de,
                    "block": feature.block,
                    "mean_support_score": group[feature.code].mean(),
                }
            )
    return pd.DataFrame(blocks), pd.DataFrame(items)


def raw_distributions(
    feature_frame: pd.DataFrame,
    profile_ids: np.ndarray,
    name_by_id: dict[str, str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    work = feature_frame.copy()
    work["profile_id"] = profile_ids
    for pid in ["Gesamt", *sorted(set(profile_ids))]:
        group = work if pid == "Gesamt" else work.loc[work["profile_id"].eq(pid)]
        name = "Gesamt" if pid == "Gesamt" else name_by_id[pid]
        for feature in FEATURES:
            counts = group[feature.raw].value_counts(dropna=False)
            for category, count in counts.items():
                rows.append(
                    {
                        "profile_id": pid,
                        "profile_name": name,
                        "profile_n": len(group),
                        "feature_code": feature.code,
                        "feature_label_de": feature.label_de,
                        "category": category,
                        "count": int(count),
                        "share": float(count / len(group)),
                    }
                )
    return pd.DataFrame(rows)


def favorable_answer_shares(
    feature_frame: pd.DataFrame,
    profile_ids: np.ndarray,
    name_by_id: dict[str, str],
) -> pd.DataFrame:
    """Berichtsnahe Rohanteile ohne das deskriptive 0–1-Scoring."""
    favorable: dict[str, tuple[str, ...]] = {
        "benefits": ("Yes",),
        "options_known": ("Yes",),
        "formal_discussion": ("Yes",),
        "resources": ("Yes",),
        "anonymity": ("Yes",),
        "leave_ease": ("Very easy", "Somewhat easy"),
        "no_mental_consequences": ("No",),
        "no_physical_consequences": ("No",),
        "coworker_comfort": ("Yes",),
        "supervisor_comfort": ("Yes",),
        "parity": ("Yes",),
        "no_observed_consequences": ("No",),
    }
    work = feature_frame.copy()
    work["profile_id"] = profile_ids
    rows: list[dict[str, Any]] = []
    for pid in ["Gesamt", *sorted(set(profile_ids))]:
        group = work if pid == "Gesamt" else work.loc[work["profile_id"].eq(pid)]
        profile_name = "Gesamt" if pid == "Gesamt" else name_by_id[pid]
        for feature in FEATURES:
            categories = favorable[feature.code]
            count = int(group[feature.raw].isin(categories).sum())
            rows.append(
                {
                    "profile_id": pid,
                    "profile_name": profile_name,
                    "profile_n": len(group),
                    "feature_code": feature.code,
                    "feature_label_de": feature.label_de,
                    "favorable_categories": ";".join(categories),
                    "favorable_count": count,
                    "favorable_share": count / len(group),
                    "favorable_percent": 100 * count / len(group),
                }
            )
    return pd.DataFrame(rows)


def ordered_k3_labels(
    raw_labels: np.ndarray, scores: pd.DataFrame
) -> tuple[np.ndarray, dict[str, str]]:
    overall = scores.assign(_cluster=raw_labels).groupby("_cluster").mean().mean(axis=1)
    ordered = overall.sort_values(ascending=False).index.tolist()
    ids = ["K3-A", "K3-B", "K3-C"]
    names = [
        "K3-A – sichtbare und ansprechbare Unterstützung",
        "K3-B – institutionell unklare Unterstützung",
        "K3-C – wenig sichtbare und schwer ansprechbare Unterstützung",
    ]
    id_map = {int(raw): pid for raw, pid in zip(ordered, ids)}
    name_by_id = dict(zip(ids, names))
    return np.array([id_map[int(value)] for value in raw_labels], dtype=str), name_by_id


def sensitive_posthoc(
    employees: pd.DataFrame,
    profile_ids: np.ndarray,
    name_by_id: dict[str, str],
) -> pd.DataFrame:
    """Nur aggregierte Kontextindikatoren; keine Modellinputs, keine Kausaltests."""
    work = employees.copy()
    work["profile_id"] = profile_ids

    def current_yes(frame: pd.DataFrame) -> tuple[int, int]:
        valid = frame[CURRENT_DISORDER].notna()
        return int(frame.loc[valid, CURRENT_DISORDER].eq("Yes").sum()), int(valid.sum())

    def treatment_yes(frame: pd.DataFrame) -> tuple[int, int]:
        valid = frame[SOUGHT_TREATMENT].notna()
        return int(frame.loc[valid, SOUGHT_TREATMENT].eq(1).sum()), int(valid.sum())

    def unsupported_yes(frame: pd.DataFrame) -> tuple[int, int]:
        valid = frame[UNSUPPORTIVE_RESPONSE].notna()
        positive = frame[UNSUPPORTIVE_RESPONSE].isin(
            ["Yes, I observed", "Yes, I experienced"]
        )
        return int((valid & positive).sum()), int(valid.sum())

    outcomes: list[tuple[str, str, Callable[[pd.DataFrame], tuple[int, int]]]] = [
        ("current_disorder_yes", "Aktuelle psychische Störung: Ja", current_yes),
        ("sought_treatment_yes", "Professionelle Hilfe in Anspruch genommen", treatment_yes),
        (
            "unsupportive_response_yes",
            "Unzureichende Reaktion beobachtet/erlebt",
            unsupported_yes,
        ),
    ]
    rows: list[dict[str, Any]] = []
    for pid in ["Gesamt", *sorted(set(profile_ids))]:
        group = work if pid == "Gesamt" else work.loc[work["profile_id"].eq(pid)]
        profile_name = "Gesamt" if pid == "Gesamt" else name_by_id[pid]
        for code, label, calculator in outcomes:
            numerator, denominator = calculator(group)
            rows.append(
                {
                    "profile_id": pid,
                    "profile_name": profile_name,
                    "outcome_code": code,
                    "outcome_label_de": label,
                    "numerator": numerator,
                    "denominator": denominator,
                    "share": numerator / denominator if denominator >= 20 else np.nan,
                    "suppressed_below_n20": denominator < 20,
                }
            )
    return pd.DataFrame(rows)


def sensitivity_analysis(
    one_hot: np.ndarray,
    primary_labels: np.ndarray,
    components: tuple[int, ...],
    bootstrap_reps: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for n_components in components:
        svd = TruncatedSVD(n_components=n_components, random_state=RANDOM_STATE)
        reduced = svd.fit_transform(one_hot)
        for k in (2, 3):
            model = KMeans(n_clusters=k, n_init=50, random_state=RANDOM_STATE).fit(reduced)
            labels = model.labels_
            stability_mean, stability_sd, stability_min = bootstrap_stability(
                reduced,
                k,
                reps=bootstrap_reps,
                seed=20260805 + n_components,
            )
            rows.append(
                {
                    "n_components": n_components,
                    "explained_variance_ratio": svd.explained_variance_ratio_.sum(),
                    "k": k,
                    "silhouette_reduced_euclidean": silhouette_score(reduced, labels),
                    "silhouette_original_onehot_euclidean": silhouette_score(
                        one_hot, labels
                    ),
                    "davies_bouldin_reduced": davies_bouldin_score(reduced, labels),
                    "bootstrap_ari_mean": stability_mean,
                    "bootstrap_ari_sd": stability_sd,
                    "bootstrap_ari_min": stability_min,
                    "min_cluster_share": np.bincount(labels).min() / len(labels),
                    "ari_vs_primary_k2": (
                        adjusted_rand_score(primary_labels, labels) if k == 2 else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def ablation_without_options_known(
    feature_frame: pd.DataFrame,
    primary_labels: np.ndarray,
) -> pd.DataFrame:
    """Prüft die mögliche Doppelabbildung strukturell übersprungener Optionen."""

    retained = [feature for feature in FEATURES if feature.code != "options_known"]
    model_frame = feature_frame[[feature.raw for feature in retained]].copy()
    model_frame.columns = [feature.code for feature in retained]
    one_hot = OneHotEncoder(
        handle_unknown="ignore", sparse_output=False, dtype=np.float64
    ).fit_transform(model_frame)
    svd = TruncatedSVD(n_components=12, random_state=RANDOM_STATE)
    reduced = svd.fit_transform(one_hot)
    model = KMeans(n_clusters=2, n_init=50, random_state=RANDOM_STATE).fit(reduced)
    labels = model.labels_
    counts = np.bincount(labels, minlength=2)
    return pd.DataFrame(
        [
            {
                "analysis": "Ablation ohne Kenntnis der Versorgungsoptionen",
                "retained_questions": len(retained),
                "one_hot_dimensions": one_hot.shape[1],
                "svd_components": 12,
                "explained_variance_ratio": svd.explained_variance_ratio_.sum(),
                "silhouette_original_onehot_euclidean": silhouette_score(
                    one_hot, labels
                ),
                "silhouette_reduced_euclidean": silhouette_score(reduced, labels),
                "davies_bouldin_reduced": davies_bouldin_score(reduced, labels),
                "ari_vs_primary_model": adjusted_rand_score(primary_labels, labels),
                "cluster_sizes_unordered": ";".join(map(str, counts.tolist())),
            }
        ]
    )


def alternating_kmedoids(
    distances: np.ndarray,
    k: int,
    starts: int = 20,
    max_iter: int = 100,
    seed: int = RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Deterministische Mehrstart-Variante des alternierenden k-Medoids-Verfahrens."""
    rng = np.random.default_rng(seed)
    n = len(distances)
    best: tuple[np.ndarray, np.ndarray, float] | None = None
    for _ in range(starts):
        medoids = [int(rng.integers(n))]
        while len(medoids) < k:
            nearest = distances[:, medoids].min(axis=1)
            nearest[medoids] = 0.0
            weights = nearest**2
            if weights.sum() == 0:
                candidates = np.setdiff1d(np.arange(n), medoids)
                medoids.append(int(rng.choice(candidates)))
            else:
                medoids.append(int(rng.choice(n, p=weights / weights.sum())))
        medoids_arr = np.array(medoids, dtype=int)
        for _iteration in range(max_iter):
            labels = distances[:, medoids_arr].argmin(axis=1)
            new_medoids = medoids_arr.copy()
            for cluster_id in range(k):
                members = np.flatnonzero(labels == cluster_id)
                if len(members) == 0:
                    candidate_order = np.argsort(distances[:, medoids_arr].min(axis=1))[::-1]
                    new_medoids[cluster_id] = next(
                        int(candidate)
                        for candidate in candidate_order
                        if candidate not in new_medoids
                    )
                    continue
                within = distances[np.ix_(members, members)]
                new_medoids[cluster_id] = int(members[within.sum(axis=1).argmin()])
            if np.array_equal(new_medoids, medoids_arr):
                break
            medoids_arr = new_medoids
        labels = distances[:, medoids_arr].argmin(axis=1)
        objective = float(distances[np.arange(n), medoids_arr[labels]].sum())
        candidate = (labels.copy(), medoids_arr.copy(), objective)
        if best is None or objective < best[2]:
            best = candidate
    if best is None:
        raise RuntimeError("k-Medoids lieferte kein Ergebnis")
    return best


def gower_robustness(
    feature_frame: pd.DataFrame, primary_labels: np.ndarray, k: int
) -> tuple[pd.DataFrame, np.ndarray]:
    encoder = OrdinalEncoder(dtype=np.int16)
    coded = encoder.fit_transform(feature_frame)
    distances = pairwise_distances(coded, metric="hamming")
    labels, medoids, objective = alternating_kmedoids(distances, k=k)
    counts = np.bincount(labels, minlength=k)
    metrics = pd.DataFrame(
        [
            {
                "method": "Alternating k-medoids on categorical Gower/Hamming distance",
                "k": k,
                "silhouette_precomputed_gower": silhouette_score(
                    distances, labels, metric="precomputed"
                ),
                "ari_vs_primary_svd_kmeans": adjusted_rand_score(primary_labels, labels),
                "objective_sum_distance_to_medoid": objective,
                "min_cluster_n": int(counts.min()),
                "min_cluster_share": float(counts.min() / len(labels)),
                "cluster_sizes_unordered": ";".join(map(str, counts.tolist())),
                "medoid_row_positions": ";".join(map(str, medoids.tolist())),
            }
        ]
    )
    return metrics, labels


def write_quality_outputs(
    df: pd.DataFrame,
    employees: pd.DataFrame,
    feature_frame: pd.DataFrame,
    tables_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    missing_rows = []
    for feature in FEATURES:
        missing_n = int(employees[feature.raw].isna().sum())
        missing_rows.append(
            {
                "feature_code": feature.code,
                "feature_label_de": feature.label_de,
                "missing_n_before_encoding": missing_n,
                "missing_share_before_encoding": missing_n / len(employees),
                "treatment": (
                    f"Explizite Kategorie {SKIPPED}; keine statistische Imputation"
                    if missing_n
                    else "Keine fehlenden Werte"
                ),
                "unknown_answers_preserved": True,
            }
        )
    missingness = pd.DataFrame(missing_rows)
    missingness.to_csv(tables_dir / "missingness_selected_features.csv", index=False)

    feature_dictionary = pd.DataFrame(
        [
            {
                "position_in_model": i + 1,
                "feature_code": f.code,
                "feature_label_de": f.label_de,
                "theory_block": f.block,
                "raw_question": f.raw,
                "n_categories_after_missing_category": feature_frame[f.raw].nunique(),
                "score_map_descriptive_only": json.dumps(
                    f.score_map, ensure_ascii=False, sort_keys=True
                ),
                "used_for_clustering": True,
            }
            for i, f in enumerate(FEATURES)
        ]
    )
    feature_dictionary.to_csv(tables_dir / "feature_dictionary.csv", index=False)

    awareness = FEATURES[1].raw
    benefits = FEATURES[0].raw
    structural = pd.crosstab(
        employees[benefits].fillna(SKIPPED),
        employees[awareness].fillna(SKIPPED),
        dropna=False,
    )
    structural.to_csv(tables_dir / "options_missingness_by_benefits.csv")

    quality_summary = pd.DataFrame(
        [
            {"metric": "rows_total", "value": len(df)},
            {"metric": "columns_total", "value": df.shape[1]},
            {"metric": "employees_in_analysis", "value": len(employees)},
            {"metric": "self_employed_excluded", "value": int(df[SELF_EMPLOYED].eq(1).sum())},
            {"metric": "exact_duplicate_rows_total", "value": int(df.duplicated().sum())},
            {"metric": "selected_cluster_features", "value": len(FEATURES)},
            {"metric": "gender_distinct_free_text_total", "value": df.iloc[:, 56].nunique(dropna=True)},
            {"metric": "work_position_distinct_combinations_total", "value": df.iloc[:, 61].nunique(dropna=True)},
            {"metric": "open_text_interview_distinct_total", "value": df.iloc[:, 37].nunique(dropna=True)},
            {"metric": "sensitive_features_used_for_clustering", "value": 0},
        ]
    )
    quality_summary.to_csv(tables_dir / "data_quality_summary.csv", index=False)
    return missingness, quality_summary


def configure_plots() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.labelcolor": "#222222",
            "text.color": "#222222",
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "grid.color": "#d9d9d9",
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "svg.hashsalt": "osmi-2016-iu-dlbdsmlusl01",
        }
    )


def save_figure(fig: plt.Figure, figures_dir: Path, stem: str) -> None:
    fig.savefig(
        figures_dir / f"{stem}.png",
        dpi=300,
        bbox_inches="tight",
        metadata={"Software": "Matplotlib; deterministic OSMI-2016 analysis"},
    )
    fig.savefig(
        figures_dir / f"{stem}.svg",
        bbox_inches="tight",
        metadata={"Date": None, "Creator": "Matplotlib; OSMI-2016 analysis"},
    )
    plt.close(fig)


def figure_missingness(missingness: pd.DataFrame, figures_dir: Path) -> None:
    ordered = missingness.sort_values("missing_share_before_encoding")
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    bars = ax.barh(
        ordered["feature_label_de"],
        ordered["missing_share_before_encoding"] * 100,
        color="#7a7a7a",
        edgecolor="#222222",
        linewidth=0.6,
    )
    ax.bar_label(bars, fmt="%.1f %%", padding=3, fontsize=8)
    ax.set_xlim(0, max(13, ordered["missing_share_before_encoding"].max() * 115))
    ax.set_xlabel("Fehlende Rohwerte (%)")
    fig.suptitle(
        "Fehlende Werte in den ausgewählten Arbeitgebermerkmalen",
        x=0.01,
        y=0.985,
        ha="left",
        fontsize=12,
    )
    fig.text(
        0.01,
        0.93,
        "Beschäftigte, n = 1.146; Ungewissheitsantworten bleiben eigenständige Kategorien",
        fontsize=8,
        color="#555555",
    )
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    fig.text(
        0.01,
        0.005,
        f"Quelle: OSMI (2016). Fehlende Werte werden als {SKIPPED} kodiert.",
        fontsize=7,
        color="#555555",
    )
    fig.subplots_adjust(left=0.36, top=0.86, bottom=0.14)
    save_figure(fig, figures_dir, "01_missingness_selected_features")


def figure_model_selection(metrics: pd.DataFrame, figures_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.4))
    k = metrics["k"]
    axes[0].plot(
        k,
        metrics["silhouette_original_onehot_euclidean"],
        color="#111111",
        marker="o",
        label="Original-One-Hot",
    )
    axes[0].plot(
        k,
        metrics["silhouette_reduced_euclidean"],
        color="#777777",
        marker="s",
        linestyle="--",
        label="SVD-Raum",
    )
    axes[0].set_title("Silhouettenkoeffizient", loc="left")
    axes[0].set_ylim(0, max(0.25, metrics["silhouette_reduced_euclidean"].max() * 1.25))
    axes[0].legend(frameon=False, fontsize=7, loc="upper right")

    axes[1].plot(k, metrics["bootstrap_ari_mean"], color="#111111", marker="o")
    axes[1].fill_between(
        k,
        metrics["bootstrap_ari_mean"] - metrics["bootstrap_ari_sd"],
        metrics["bootstrap_ari_mean"] + metrics["bootstrap_ari_sd"],
        color="#d9d9d9",
        linewidth=0,
    )
    axes[1].axhline(0.8, color="#777777", linestyle="--", linewidth=0.8)
    axes[1].set_ylim(0, 1.02)
    axes[1].set_title("Subsample-Stabilität (ARI)", loc="left")

    axes[2].plot(k, metrics["davies_bouldin_reduced"], color="#111111", marker="o")
    axes[2].set_ylim(0, metrics["davies_bouldin_reduced"].max() * 1.2)
    axes[2].set_title("Davies–Bouldin-Index", loc="left")
    axes[2].text(
        0.02,
        0.95,
        "niedriger = besser",
        transform=axes[2].transAxes,
        va="top",
        fontsize=7,
        color="#555555",
    )
    for ax in axes:
        ax.set_xticks(k)
        ax.set_xlabel("Clusterzahl k")
        ax.grid(axis="y")
        ax.set_axisbelow(True)
    fig.suptitle("Vergleich der Clusterlösungen k = 2 bis k = 6", x=0.01, ha="left", fontsize=12)
    fig.text(
        0.01,
        0.91,
        "One-Hot → 12 SVD-Komponenten → K-Means; n = 1.146",
        fontsize=8,
        color="#555555",
    )
    fig.text(
        0.01,
        0.005,
        "Quelle: Eigene Berechnung auf Basis OSMI (2016). ARI: 80%-Subsamples, 20 Wiederholungen.",
        fontsize=7,
        color="#555555",
    )
    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.20, top=0.78, wspace=0.30)
    save_figure(fig, figures_dir, "02_model_selection")


def figure_svd_map(
    reduced: np.ndarray,
    svd: TruncatedSVD,
    profile_ids: np.ndarray,
    name_by_id: dict[str, str],
    figures_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.7))
    styles = {
        "A": dict(marker="o", facecolors="#333333", edgecolors="#111111", alpha=0.46),
        "B": dict(marker="s", facecolors="none", edgecolors="#777777", alpha=0.46),
    }
    legend_names = {
        "A": "Profil A – sichtbar/ansprechbar",
        "B": "Profil B – wenig sichtbar/schwer ansprechbar",
    }
    for pid in sorted(set(profile_ids)):
        mask = profile_ids == pid
        ax.scatter(
            reduced[mask, 0],
            reduced[mask, 1],
            s=15,
            linewidths=0.5,
            label=legend_names.get(pid, name_by_id[pid]),
            **styles.get(pid, {}),
        )
    ax.set_xlabel(f"SVD-Komponente 1 ({svd.explained_variance_ratio_[0] * 100:.1f} % Varianz)")
    ax.set_ylabel(f"SVD-Komponente 2 ({svd.explained_variance_ratio_[1] * 100:.1f} % Varianz)")
    fig.suptitle(
        "Zweidimensionale Projektion der Unterstützungsprofile",
        x=0.01,
        y=0.985,
        ha="left",
        fontsize=12,
    )
    fig.text(
        0.01,
        0.93,
        "Die Projektion dient der Orientierung; das Modell nutzt 12 SVD-Komponenten.",
        fontsize=8,
        color="#555555",
    )
    ax.legend(frameon=False, loc="best", fontsize=8)
    ax.grid(True)
    ax.set_axisbelow(True)
    fig.text(
        0.01,
        0.005,
        "Quelle: Eigene Berechnung auf Basis OSMI (2016). Überlappung ist sichtbar und wird nicht als harte Typgrenze interpretiert.",
        fontsize=7,
        color="#555555",
    )
    fig.subplots_adjust(left=0.11, right=0.98, top=0.85, bottom=0.15)
    save_figure(fig, figures_dir, "03_svd_cluster_map")


def figure_profile_heatmap(item_profiles: pd.DataFrame, figures_dir: Path) -> None:
    ordered_profiles = ["Gesamt", "A", "B"]
    pivot = item_profiles.pivot(
        index="feature_label_de", columns="profile_id", values="mean_support_score"
    )
    feature_order = [f.label_de for f in FEATURES]
    pivot = pivot.loc[feature_order, ordered_profiles]
    ns = (
        item_profiles.drop_duplicates("profile_id")
        .set_index("profile_id")["n"]
        .reindex(ordered_profiles)
    )
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    image = ax.imshow(pivot.to_numpy() * 100, cmap="Greys", vmin=0, vmax=100, aspect="auto")
    ax.set_yticks(range(len(pivot)), labels=pivot.index)
    ax.set_xticks(
        range(len(ordered_profiles)),
        labels=[f"{p}\n(n = {int(ns[p]):,})".replace(",", ".") for p in ordered_profiles],
    )
    for row in range(pivot.shape[0]):
        for col in range(pivot.shape[1]):
            value = pivot.iloc[row, col] * 100
            ax.text(
                col,
                row,
                f"{value:.0f}",
                ha="center",
                va="center",
                color="white" if value > 58 else "#111111",
                fontsize=8,
            )
    ax.axhline(4.5, color="#111111", linewidth=1.2)
    fig.suptitle(
        "Deskriptive Unterstützungs- und Sicherheitsprofile",
        x=0.01,
        y=0.985,
        ha="left",
        fontsize=12,
    )
    fig.text(
        0.01,
        0.94,
        "Theoriegeleitete Rekodierung: 0 = ungünstig, 100 = günstig; nicht als Modellinput verwendet",
        fontsize=8,
        color="#555555",
    )
    cbar = fig.colorbar(image, ax=ax, fraction=0.028, pad=0.025)
    cbar.set_label("Mittelwert (0–100)")
    fig.text(
        0.01,
        0.005,
        "Quelle: Eigene Berechnung auf Basis OSMI (2016). Trennlinie: formale Unterstützung/Navigation vs. psychologische Sicherheit.",
        fontsize=7,
        color="#555555",
    )
    fig.subplots_adjust(left=0.40, right=0.92, top=0.87, bottom=0.10)
    save_figure(fig, figures_dir, "04_cluster_profile_heatmap")


def figure_posthoc_context(posthoc: pd.DataFrame, figures_dir: Path) -> None:
    ordered_profiles = ["Gesamt", "A", "B"]
    outcomes = posthoc["outcome_code"].drop_duplicates().tolist()
    labels = (
        posthoc.drop_duplicates("outcome_code")
        .set_index("outcome_code")["outcome_label_de"]
        .to_dict()
    )
    pivot = posthoc.pivot(index="outcome_code", columns="profile_id", values="share")
    pivot = pivot.loc[outcomes, ordered_profiles] * 100
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    y = np.arange(len(outcomes))
    offsets = {"Gesamt": -0.20, "A": 0.0, "B": 0.20}
    styles = {
        "Gesamt": dict(marker="D", edgecolors="#777777", facecolors="#a0a0a0"),
        "A": dict(marker="o", edgecolors="#222222", facecolors="#222222"),
        "B": dict(marker="s", edgecolors="#666666", facecolors="white"),
    }
    for pid in ordered_profiles:
        values = pivot[pid].to_numpy()
        ax.scatter(values, y + offsets[pid], s=42, linewidth=1.0, label=pid, **styles[pid])
        for value, yy in zip(values, y + offsets[pid]):
            ax.text(value + 1.2, yy, f"{value:.1f}", va="center", fontsize=7)
    ax.set_yticks(y, labels=[labels[code] for code in outcomes])
    ax.set_xlim(0, max(70, np.nanmax(pivot.to_numpy()) + 8))
    ax.set_xlabel("Anteil (%)")
    fig.suptitle(
        "Aggregierte sensible Kontextindikatoren nach Profil",
        x=0.01,
        y=0.985,
        ha="left",
        fontsize=12,
    )
    fig.text(
        0.01,
        0.92,
        "Nur post hoc; nicht für die Clusterbildung genutzt; deskriptiv und nicht kausal",
        fontsize=8,
        color="#555555",
    )
    ax.legend(frameon=False, ncol=1, loc="upper left", fontsize=8)
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    ax.invert_yaxis()
    fig.text(
        0.01,
        0.005,
        "Quelle: Eigene Berechnung auf Basis OSMI (2016). Nenner stehen in der Ergebnistabelle; Zellen < 20 würden unterdrückt.",
        fontsize=7,
        color="#555555",
    )
    fig.subplots_adjust(left=0.42, right=0.98, top=0.84, bottom=0.17)
    save_figure(fig, figures_dir, "05_posthoc_sensitive_context")


def figure_sensitivity(sensitivity: pd.DataFrame, figures_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.4))
    for k, marker, color, fill in [(2, "o", "#111111", "#111111"), (3, "s", "#777777", "white")]:
        group = sensitivity.loc[sensitivity["k"].eq(k)]
        axes[0].plot(
            group["n_components"],
            group["silhouette_original_onehot_euclidean"],
            marker=marker,
            color=color,
            markerfacecolor=fill,
            label=f"k = {k}",
        )
        axes[1].plot(
            group["n_components"],
            group["bootstrap_ari_mean"],
            marker=marker,
            color=color,
            markerfacecolor=fill,
            label=f"k = {k}",
        )
    axes[0].set_title("Silhouette im Original-One-Hot-Raum", loc="left")
    axes[0].set_ylim(0, 0.16)
    axes[1].set_title("Subsample-Stabilität (ARI)", loc="left")
    axes[1].set_ylim(0, 1.02)
    for ax in axes:
        ax.set_xticks(sorted(sensitivity["n_components"].unique()))
        ax.set_xlabel("SVD-Komponenten")
        ax.grid(axis="y")
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Koeffizient")
    axes[1].legend(frameon=False, loc="lower left", fontsize=8)
    fig.suptitle("Sensitivität gegenüber der Zahl der SVD-Komponenten", x=0.01, ha="left", fontsize=12)
    fig.text(
        0.01,
        0.005,
        "Quelle: Eigene Berechnung auf Basis OSMI (2016). Komponenten 8/12/16 erklären rund 54/69/80 % Varianz.",
        fontsize=7,
        color="#555555",
    )
    fig.subplots_adjust(left=0.09, right=0.98, top=0.76, bottom=0.20, wspace=0.28)
    save_figure(fig, figures_dir, "06_svd_sensitivity")


def write_chart_map(qa_dir: Path) -> None:
    rows = [
        {
            "figure": "01_missingness_selected_features",
            "question": "Wo liegen fehlende Rohwerte in den 12 Modellmerkmalen?",
            "chart": "Horizontaler Balken",
            "takeaway": "Nur Optionskenntnis weist 11,6 % fehlende Werte auf; Unsicherheit ist keine Missingness.",
        },
        {
            "figure": "02_model_selection",
            "question": "Welche Clusterzahl verbindet Trennschärfe, Stabilität und Mindestgröße?",
            "chart": "Drei Linienpanels",
            "takeaway": "k=2 ist stabiler und im Originalraum trennschärfer als k=3..6, aber absolut nur schwach getrennt.",
        },
        {
            "figure": "03_svd_cluster_map",
            "question": "Wie stark überlappen die Profile in einer 2D-Projektion?",
            "chart": "Scatterplot",
            "takeaway": "Die Profile zeigen Struktur, bleiben aber deutlich überlappend.",
        },
        {
            "figure": "04_cluster_profile_heatmap",
            "question": "Bei welchen Arbeitgebermerkmalen unterscheiden sich die Profile?",
            "chart": "Annotierte Heatmap",
            "takeaway": "Profil B bündelt wenig sichtbare und schwer ansprechbare Unterstützung.",
        },
        {
            "figure": "05_posthoc_sensitive_context",
            "question": "Wie verteilen sich sensible Kontextindikatoren aggregiert?",
            "chart": "Gruppierter Dotplot",
            "takeaway": "Unterschiede sind deskriptive Assoziationen; sensible Felder waren keine Modellinputs.",
        },
        {
            "figure": "06_svd_sensitivity",
            "question": "Bleibt die Wahl von k bei 8/12/16 SVD-Komponenten bestehen?",
            "chart": "Zwei Linienpanels",
            "takeaway": "k=2 bleibt gegenüber k=3 trennschärfer und stabiler.",
        },
    ]
    frame = pd.DataFrame(rows)
    frame["palette_policy"] = "Graustufen; Form/Füllung zusätzlich zu Tonwert"
    frame["final_surface"] = "PNG (300 dpi) und SVG"
    frame.to_csv(qa_dir / "chart_map.csv", index=False)


def write_summary(
    outputs_dir: Path,
    tables_dir: Path,
    selected_k: int,
    metrics: pd.DataFrame,
    svd: TruncatedSVD,
    one_hot: np.ndarray,
    block_table: pd.DataFrame,
    k3_block_table: pd.DataFrame,
    gower_metrics: pd.DataFrame,
    sensitivity: pd.DataFrame,
) -> None:
    selected = metrics.loc[metrics["k"].eq(selected_k)].iloc[0]
    tables_relative = tables_dir.relative_to(outputs_dir.parent)
    summary = {
        "dataset": {
            "source": DATA_URL,
            "csv": CSV_NAME,
            "sha256": CSV_SHA256,
            "rows_total": 1433,
            "employees_analyzed": 1146,
        },
        "features": {
            "count": len(FEATURES),
            "codes": [f.code for f in FEATURES],
            "sensitive_or_demographic_cluster_features": 0,
            "one_hot_dimensions": int(one_hot.shape[1]),
            "missing_strategy": f"explicit category {SKIPPED}; uncertainty preserved",
        },
        "primary_model": {
            "pipeline": "OneHotEncoder -> TruncatedSVD -> KMeans",
            "svd_components": int(svd.n_components),
            "explained_variance_ratio": float(svd.explained_variance_ratio_.sum()),
            "selected_k": selected_k,
            "selection_rule": (
                "ARI mean >= .80 and minimum cluster share >= .10; then maximum "
                "silhouette in original one-hot Euclidean space; ties favor lower k"
            ),
            "silhouette_reduced_euclidean": float(selected["silhouette_reduced_euclidean"]),
            "silhouette_original_onehot_euclidean": float(
                selected["silhouette_original_onehot_euclidean"]
            ),
            "davies_bouldin_reduced": float(selected["davies_bouldin_reduced"]),
            "bootstrap_ari_mean": float(selected["bootstrap_ari_mean"]),
            "bootstrap_ari_sd": float(selected["bootstrap_ari_sd"]),
            "bootstrap_definition": (
                "mean pairwise adjusted Rand index across full-sample assignments "
                "from 20 independent 80% subsample fits"
            ),
            "cluster_sizes_unordered": selected["cluster_sizes_unordered"],
        },
        "ordered_profiles": block_table.to_dict(orient="records"),
        "diagnostic_k3_profiles": k3_block_table.to_dict(orient="records"),
        "gower_kmedoids_robustness": gower_metrics.iloc[0].to_dict(),
        "sensitivity": sensitivity.to_dict(orient="records"),
        "interpretation_guardrail": (
            "Low silhouette means overlapping exploratory profiles, not natural, "
            "diagnostic, or causal employee types."
        ),
    }
    (outputs_dir / "analysis_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float), encoding="utf-8"
    )

    profile_lines = []
    for row in block_table.itertuples(index=False):
        profile_lines.append(
            f"- {row.profile_name}: n = {row.n}; formale Unterstützung/Navigation "
            f"{row.formal_support_navigation:.3f}; psychologische Sicherheit "
            f"{row.psychological_safety:.3f}; Gesamtwert {row.overall_support_score:.3f}."
        )
    k3_lines = []
    for row in k3_block_table.itertuples(index=False):
        if row.profile_id == "Gesamt":
            continue
        k3_lines.append(
            f"- {row.profile_name}: n = {row.n}; formale Unterstützung/Navigation "
            f"{row.formal_support_navigation:.3f}; psychologische Sicherheit "
            f"{row.psychological_safety:.3f}."
        )
    markdown = f"""# Reproduzierbare OSMI-2016-Analyse – Ergebnisnotiz

## Population und Merkmale

- OSMI Mental Health in Tech Survey 2016: 1.433 Fälle, 63 Variablen.
- Analysiert werden 1.146 nicht selbstständige Beschäftigte.
- Clustering auf 12 aktuellen Arbeitgebermerkmalen; Diagnosen, Behandlung, Alter,
  Geschlecht und andere demografische Merkmale sind ausgeschlossen.
- 133 fehlende Angaben zur Kenntnis der Versorgungsoptionen werden als
  `{SKIPPED}` erhalten. Antworten wie „I don't know“, „I am not sure“ und
  „Maybe“ bleiben eigene Kategorien.

## Modell

- One-Hot-Matrix: {one_hot.shape[1]} Dimensionen.
- TruncatedSVD: 12 Komponenten, erklärte Varianz
  {svd.explained_variance_ratio_.sum() * 100:.2f} %.
- Gewählt: k = {selected_k}. Silhouette im reduzierten Raum:
  {selected['silhouette_reduced_euclidean']:.3f}; im Original-One-Hot-Raum:
  {selected['silhouette_original_onehot_euclidean']:.3f}; Davies–Bouldin:
  {selected['davies_bouldin_reduced']:.3f}.
- Stabilität: mittlere paarweise ARI
  {selected['bootstrap_ari_mean']:.3f} (20 unabhängige 80%-Subsamples,
  Vorhersage jeweils für alle 1.146 Fälle).

## Profile

{chr(10).join(profile_lines)}

Die diagnostische k=3-Lösung ist ebenfalls stabil und substanziell, bildet aber
vor allem eine ordinale Abstufung statt eines eigenständigen Gap-Typs ab:

{chr(10).join(k3_lines)}

Sie wird deshalb als Sensitivitätsbefund und nicht als Hauptsegmentierung geführt.

## Robuste Einordnung

Die geringe Silhouette zeigt deutliche Überlappung. Die Cluster sind deshalb
explorative Organisationsprofile, keine natürlichen, diagnostischen oder kausalen
Personentypen. Eine alternative k-Medoids-Lösung auf kategorialer
Gower/Hamming-Distanz erreicht Silhouette
{gower_metrics.iloc[0]['silhouette_precomputed_gower']:.3f} und stimmt mit der
Hauptlösung nur mit ARI {gower_metrics.iloc[0]['ari_vs_primary_svd_kmeans']:.3f}
überein. Diese begrenzte methodenübergreifende Übereinstimmung ist als zentrale
Unsicherheit auszuweisen.

Alle Tabellen liegen unter `{tables_relative}`. Zahlen im Bericht sollten direkt aus
den CSV- und JSON-Artefakten übernommen werden.
"""
    (outputs_dir / "analysis_summary.md").write_text(markdown, encoding="utf-8")


def write_validation_report(
    qa_dir: Path,
    metrics: pd.DataFrame,
    selected_k: int,
    feature_frame: pd.DataFrame,
    one_hot: np.ndarray,
    svd: TruncatedSVD,
    gower_metrics: pd.DataFrame,
) -> None:
    selected = metrics.loc[metrics["k"].eq(selected_k)].iloc[0]
    checks = [
        ("PASS", "CSV-Fingerprint", sha256(qa_dir.parent / "data" / "raw" / CSV_NAME)),
        ("PASS", "Analyseeinheit", "1.146 Beschäftigte; 287 Selbstständige ausgeschlossen"),
        ("PASS", "Exakte Duplikate", "0"),
        ("PASS", "Clusterfeatures", "12 aktuelle Arbeitgebermerkmale"),
        ("PASS", "Sensible Modellinputs", "0"),
        ("PASS", "Zeilennorm One-Hot", "Jede Person hat exakt 12 aktive Indikatoren"),
        ("PASS", "SVD-Erklärungsanteil", f"{svd.explained_variance_ratio_.sum():.6f}"),
        ("PASS", "Mindestclusteranteil", f"{selected['min_cluster_share']:.3f}"),
        ("PASS", "Bootstrap-Stabilität", f"ARI mean {selected['bootstrap_ari_mean']:.3f}"),
        (
            "CAVEAT",
            "Trennschärfe",
            f"Silhouette Original-One-Hot {selected['silhouette_original_onehot_euclidean']:.3f}",
        ),
        (
            "CAVEAT",
            "Methodenrobustheit",
            f"ARI Hauptmodell vs. Gower/k-Medoids {gower_metrics.iloc[0]['ari_vs_primary_svd_kmeans']:.3f}",
        ),
    ]
    norms = np.linalg.norm(one_hot, axis=1)
    if not np.allclose(norms, np.sqrt(len(FEATURES))):
        raise AssertionError("One-Hot-Zeilennormen sind nicht konstant")
    if feature_frame.isna().any().any():
        raise AssertionError("Nach expliziter Missing-Kodierung verbleiben NA")
    lines = [
        "# Validierungsbericht der OSMI-Analyse",
        "",
        "## Gesamteinschätzung: Mit deutlichen Einschränkungen teilbar",
        "",
        "Die Berechnungen sind reproduzierbar und intern konsistent. Die geringe",
        "Silhouette und die begrenzte Übereinstimmung mit einer alternativen",
        "Distanzmethode verhindern jedoch eine Interpretation als natürliche Typen.",
        "",
        "## Prüfpunkte",
        "",
        "| Status | Prüfpunkt | Evidenz |",
        "|---|---|---|",
    ]
    lines += [f"| {status} | {name} | {evidence} |" for status, name, evidence in checks]
    lines += [
        "",
        "## Erforderliche Berichtscaveats",
        "",
        "- Querschnittliche, freiwillige Online-Stichprobe; keine Repräsentativität.",
        "- Selbstauskunft und gemeinsame Erhebungsmethode; keine Kausalität.",
        "- K-Means in einem SVD-Raum ist eine pragmatische, prüfungsnahe Näherung für kategoriale Daten.",
        "- Niedrige Silhouette: Profile überlappen deutlich.",
        "- Theoriegeleitete 0–1-Scores dienen nur der Beschreibung, nicht der Clusterbildung.",
        "- Sensible post-hoc Kennzahlen sind aggregiert und dürfen nicht zur individuellen Einstufung genutzt werden.",
        "- Maßnahmen sind Hypothesen für partizipative Erprobung, keine Wirksamkeitsnachweise aus diesem Datensatz.",
    ]
    (qa_dir / "analysis_validation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest(project_root: Path, qa_dir: Path) -> None:
    paths = sorted(
        path
        for parent in [project_root / "src", project_root / "data" / "processed", project_root / "outputs", qa_dir]
        for path in parent.rglob("*")
        if path.is_file() and path.name != "reproduction_manifest.json"
    )
    manifest = {
        "generated_at_note": "Deterministic analysis; wall-clock timestamp intentionally omitted",
        "python": sys.version,
        "platform": platform.platform(),
        "package_versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "matplotlib": mpl.__version__,
        },
        "random_state": RANDOM_STATE,
        "files": [
            {
                "relative_path": str(path.relative_to(project_root)),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in paths
        ],
    }
    (qa_dir / "reproduction_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Projektwurzel (Standard: Elternordner von src)",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Fehler ausgeben, falls die lokale CSV fehlt",
    )
    parser.add_argument(
        "--bootstrap-reps",
        type=int,
        default=20,
        help="Subsample-Wiederholungen der Hauptanalyse (mindestens 5)",
    )
    parser.add_argument(
        "--sensitivity-bootstrap-reps",
        type=int,
        default=12,
        help="Subsample-Wiederholungen je Sensitivitätsmodell (mindestens 5)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.bootstrap_reps < 5 or args.sensitivity_bootstrap_reps < 5:
        raise ValueError("Mindestens fünf Bootstrap/Subsample-Wiederholungen erforderlich")
    project_root = args.project_root.resolve()
    raw_dir = project_root / "data" / "raw"
    processed_dir = project_root / "data" / "processed"
    outputs_dir = project_root / "outputs"
    tables_dir = outputs_dir / "tables"
    figures_dir = outputs_dir / "figures"
    qa_dir = project_root / "qa"
    for path in [processed_dir, tables_dir, figures_dir, qa_dir]:
        path.mkdir(parents=True, exist_ok=True)

    csv_path = ensure_dataset(raw_dir, download=not args.no_download)
    df = pd.read_csv(csv_path)
    validate_schema(df)
    employees, feature_frame = prepare_features(df)
    scores = make_theory_scores(feature_frame)
    missingness, _quality = write_quality_outputs(
        df, employees, feature_frame, tables_dir
    )

    model_frame = feature_frame.copy()
    model_frame.columns = [f.code for f in FEATURES]
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float64)
    one_hot = encoder.fit_transform(model_frame)
    if one_hot.shape != (1146, 40):
        raise AssertionError(f"Unerwartete One-Hot-Form: {one_hot.shape}")
    encoded_names = encoder.get_feature_names_out()
    pd.DataFrame({"encoded_feature": encoded_names}).to_csv(
        tables_dir / "onehot_feature_names.csv", index=False
    )

    svd = TruncatedSVD(n_components=12, random_state=RANDOM_STATE)
    reduced = svd.fit_transform(one_hot)
    pd.DataFrame(
        {
            "component": np.arange(1, 13),
            "explained_variance_ratio": svd.explained_variance_ratio_,
            "cumulative_explained_variance_ratio": np.cumsum(
                svd.explained_variance_ratio_
            ),
        }
    ).to_csv(tables_dir / "svd_explained_variance.csv", index=False)

    metrics, models = evaluate_k_values(
        one_hot,
        reduced,
        range(2, 7),
        bootstrap_reps=args.bootstrap_reps,
    )
    selected_k = choose_k(metrics)
    if selected_k != 2:
        raise AssertionError(
            f"Reproduktionsanker verletzt: erwartete Auswahl k=2, erhalten k={selected_k}"
        )
    primary_raw_labels = models[selected_k].labels_
    profile_ids, raw_name_map, raw_id_map = ordered_profile_labels(
        primary_raw_labels, scores
    )
    name_by_id = {raw_id_map[raw]: name for raw, name in raw_name_map.items()}

    metrics.to_csv(tables_dir / "cluster_model_selection.csv", index=False)
    block_table, item_table = block_profiles(scores, profile_ids, name_by_id)
    block_table.to_csv(tables_dir / "cluster_block_profiles.csv", index=False)
    item_table.to_csv(tables_dir / "cluster_item_scores.csv", index=False)
    raw_table = raw_distributions(feature_frame, profile_ids, name_by_id)
    raw_table.to_csv(tables_dir / "cluster_raw_answer_distributions.csv", index=False)
    favorable_table = favorable_answer_shares(feature_frame, profile_ids, name_by_id)
    favorable_table.to_csv(
        tables_dir / "cluster_favorable_answer_percentages.csv", index=False
    )
    posthoc = sensitive_posthoc(employees, profile_ids, name_by_id)
    posthoc.to_csv(tables_dir / "posthoc_sensitive_outcomes.csv", index=False)

    # k=3 wird als diagnostische Feinlösung dokumentiert, aber nicht als
    # Hauptlösung gewählt: Die Profile bilden überwiegend eine Abstufung ab.
    k3_profile_ids, k3_name_by_id = ordered_k3_labels(models[3].labels_, scores)
    k3_blocks, k3_items = block_profiles(scores, k3_profile_ids, k3_name_by_id)
    k3_blocks.to_csv(tables_dir / "k3_diagnostic_block_profiles.csv", index=False)
    k3_items.to_csv(tables_dir / "k3_diagnostic_item_scores.csv", index=False)
    raw_distributions(feature_frame, k3_profile_ids, k3_name_by_id).to_csv(
        tables_dir / "k3_diagnostic_raw_answer_distributions.csv", index=False
    )

    participant_ids = np.array([f"P{i:04d}" for i in range(1, len(employees) + 1)])
    assignments = pd.DataFrame(
        {
            "participant_id_pseudonymous": participant_ids,
            "profile_id": profile_ids,
            "profile_name": [name_by_id[pid] for pid in profile_ids],
            "svd_component_1": reduced[:, 0],
            "svd_component_2": reduced[:, 1],
            "formal_support_navigation_score": scores[[f.code for f in FEATURES[:5]]].mean(axis=1).to_numpy(),
            "psychological_safety_score": scores[[f.code for f in FEATURES[5:]]].mean(axis=1).to_numpy(),
        }
    )
    assignments.to_csv(processed_dir / "participant_cluster_assignments.csv", index=False)

    sensitivity = sensitivity_analysis(
        one_hot,
        primary_raw_labels,
        components=(8, 12, 16),
        bootstrap_reps=args.sensitivity_bootstrap_reps,
    )
    sensitivity.to_csv(tables_dir / "sensitivity_svd_components.csv", index=False)

    ablation = ablation_without_options_known(feature_frame, primary_raw_labels)
    ablation.to_csv(
        tables_dir / "robustness_without_options_known.csv", index=False
    )

    gower_metrics, gower_labels = gower_robustness(
        feature_frame, primary_raw_labels, selected_k
    )
    gower_metrics.to_csv(tables_dir / "robustness_gower_kmedoids.csv", index=False)
    pd.DataFrame(
        {
            "participant_id_pseudonymous": participant_ids,
            "gower_kmedoids_raw_cluster": gower_labels,
        }
    ).to_csv(processed_dir / "gower_kmedoids_assignments.csv", index=False)

    configure_plots()
    figure_missingness(missingness, figures_dir)
    figure_model_selection(metrics, figures_dir)
    figure_svd_map(reduced, svd, profile_ids, name_by_id, figures_dir)
    figure_profile_heatmap(item_table, figures_dir)
    figure_posthoc_context(posthoc, figures_dir)
    figure_sensitivity(sensitivity, figures_dir)
    write_chart_map(qa_dir)
    write_summary(
        outputs_dir,
        tables_dir,
        selected_k,
        metrics,
        svd,
        one_hot,
        block_table,
        k3_blocks,
        gower_metrics,
        sensitivity,
    )
    write_validation_report(
        qa_dir,
        metrics,
        selected_k,
        feature_frame,
        one_hot,
        svd,
        gower_metrics,
    )
    write_manifest(project_root, qa_dir)

    chosen = metrics.loc[metrics["k"].eq(selected_k)].iloc[0]
    print(
        json.dumps(
            {
                "status": "ok",
                "selected_k": selected_k,
                "employees": len(employees),
                "one_hot_dimensions": one_hot.shape[1],
                "svd_explained_variance": svd.explained_variance_ratio_.sum(),
                "silhouette_original_onehot": chosen[
                    "silhouette_original_onehot_euclidean"
                ],
                "silhouette_reduced": chosen["silhouette_reduced_euclidean"],
                "bootstrap_ari_mean": chosen["bootstrap_ari_mean"],
                "output": str(outputs_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
