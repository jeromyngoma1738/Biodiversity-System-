from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from models import Observation


# ============================================================
# CONFIGURATION
# ============================================================

# A change smaller than this is called "Stable".
TREND_BAND_PCT = 10.0

# Below this count, percentage changes are noisy and get damped.
LOW_COUNT_THRESHOLD = 5
SAME_SURVEY_POLICY = "separate"

# A count this many times the median of previous counts is flagged as
# a possible data-entry error rather than silently driving the result.
OUTLIER_RATIO = 20.0

# Forecasting
MIN_FORECAST_POINTS = 3
DAYS_PER_YEAR = 365.25
MAX_FORECAST_YEARS = 50

ENVIRONMENT_COLUMNS = [
    "area_hectares",
    "temperature",
    "rainfall",
    "soil_ph",
    "soil_moisture",
    "soil_quality",
    "water_ph",
    "water_turbidity",
    "water_quality",
    "vegetation_cover",
    "vegetation_density",
    "grass_availability",
    "tree_density",
]

# suitable_low, suitable_high, marginal_low, marginal_high
SUITABILITY_RANGES = {
    "temperature":    (15, 32, 10, 36),
    "water_quality":  (60, 100, 30, 100),
    "water_turbidity": (0, 40, 0, 70),
    "water_ph":       (6.5, 8.5, 6.0, 9.0),
    "soil_quality":   (60, 100, 30, 100),
    "soil_moisture":  (30, 100, 15, 100),
    "soil_ph":        (5.0, 7.5, 4.5, 8.0),
}

PERCENT_FACTORS = {
    "water_quality",
    "water_turbidity",
    "soil_quality",
    "soil_moisture",
}

FACTOR_GROUPS = {
    "temperature": ["temperature"],
    "water": ["water_quality", "water_turbidity", "water_ph"],
    "soil": ["soil_quality", "soil_moisture", "soil_ph"],
}

FACTOR_LABELS = {
    "temperature": "temperature",
    "water_quality": "water quality",
    "water_turbidity": "water turbidity",
    "water_ph": "water pH",
    "soil_quality": "soil quality",
    "soil_moisture": "soil moisture",
    "soil_ph": "soil pH",
}

GROUP_WEIGHTS = {
    "Producer":   {"temperature": 0.30, "water": 0.20, "soil": 0.50},
    "Decomposer": {"temperature": 0.30, "water": 0.20, "soil": 0.50},
    "default":    {"temperature": 0.30, "water": 0.40, "soil": 0.30},
}

RAINFALL_CLASSES = [
    (10, "Very Low"),
    (50, "Low"),
    (100, "Moderate"),
    (200, "High"),
]

# Reference grazing-unit equivalents. Project values, not CBU-calibrated.
WILDLIFE_UNIT_FACTORS = {
    "impala": 0.18,
    "zebra": 1.49,
    "waterbuck": 1.15,
    "puku": 0.30,
    "kudu": 0.70,
    "sable antelope": 1.20,
    "bushbuck": 0.40,
    "duiker": 0.20,
    "warthog": 0.60,
    "buffalo": 1.80,
}

# Reference grazing units per hectare.
RESOURCE_PRESSURE_THRESHOLDS = {"low": 0.50, "moderate": 1.00, "high": 2.00}

IMPACT_LABELS = {
    0: "Low Concern",
    1: "Watch",
    2: "Moderate Concern",
    3: "High Concern",
    4: "Critical Concern",
}


# ============================================================
# NAME HANDLING
# ============================================================

SPECIES_ALIASES = {
    "mice": "mouse",
    "rats": "rat",
    "frogs": "frog",
    "snakes": "snake",
    "lizards": "lizard",
    "spiders": "spider",
    "grasshoppers": "grasshopper",
    "caterpillars": "caterpillar",
    "trees": "tree",
    "plants": "plant",
    "grasses": "grass",
    "waterbucks": "waterbuck",
    "impalas": "impala",
    "zebras": "zebra",
    "pukus": "puku",
    "doves": "dove",
    "pigeons": "pigeon",
    "crow": "african pied crow",
}


def normalize_name(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip().lower().replace("_", " ").replace("-", " ")
    return " ".join(text.split())


def canonical_species_name(value: Any) -> str:
    name = normalize_name(value)
    return SPECIES_ALIASES.get(name, name)


def pretty_name(value: Any) -> str:
    return "" if value is None else str(value).title()


def names_text(names: List[str]) -> str:
    return ", ".join(pretty_name(n) for n in names)


def safe_number(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(number) or np.isinf(number):
        return None
    return number


def canonical_set(items: List[str]) -> set:
    return {canonical_species_name(i) for i in items}


# ============================================================
# TROPHIC ROLES
# ============================================================

PRODUCERS = canonical_set([
    "Tree", "Grass", "Flowering Plant", "Acacia", "Miombo Tree", "Mopane",
    "Grass Species", "Plant", "Brachystegia", "Julbernardia", "Isoberlinia",
    "Faidherbia Albida", "Khasi Pine", "Apple-ring Acacia", "Ana Tree",
    "Wild loquate", "Wild custard apple", "Msasa",
])

HERBIVORES = canonical_set([
    "Impala", "Waterbuck", "Puku", "Kudu", "Sable Antelope", "Bushbuck",
    "Duiker", "Warthog", "Zebra", "Buffalo", "Rabbit", "Grasshopper",
    "Caterpillar", "Snail", "Village Weaver", "Southern Red Bishop",
    "Speckled Mousebird", "Cape Turtle Dove", "Laughing Dove",
])

OMNIVORES = canonical_set([
    "Mouse", "Rat", "Rodent", "African Pied Crow", "Dove", "Pigeon",
    "African Grey Hornbill", "Southern Yellow-billed Hornbill",
    "Crested Barbet", "Civet", "Hadada Ibis", "Helmeted Guineafowl",
])

SMALL_PREDATORS = canonical_set([
    "Spider", "Praying Mantis", "Frog", "Lizard", "Insectivorous Bird",
    "African Hoopoe", "Fork-tailed Drongo", "Common Fiscal",
    "African Paradise Flycatcher",
])

PREDATORS = canonical_set(["Snake", "Mongoose", "Cat"])

# Classification only. Not assumed present in CBU Nature Park unless
# an approved observation exists.
LARGE_PREDATORS = canonical_set(["Monitor Lizard", "Eagle", "Crocodile"])

DECOMPOSERS = canonical_set(["Fungi", "Bacteria", "Termite", "Earthworm"])

LARGE_HERBIVORES = canonical_set([
    "Impala", "Waterbuck", "Puku", "Kudu", "Sable Antelope", "Bushbuck",
    "Duiker", "Warthog", "Zebra", "Buffalo",
])

PREDATOR_ROLES = ("Small Predator", "Predator", "Large Predator")


def get_role(species_name: Any) -> str:
    name = canonical_species_name(species_name)
    if name in PRODUCERS:
        return "Producer"
    if name in HERBIVORES:
        return "Herbivore"
    if name in OMNIVORES:
        return "Omnivore"
    if name in SMALL_PREDATORS:
        return "Small Predator"
    if name in PREDATORS:
        return "Predator"
    if name in LARGE_PREDATORS:
        return "Large Predator"
    if name in DECOMPOSERS:
        return "Decomposer"
    return "Unknown"


ROLE_TEXT = {
    "Producer": "It provides food and habitat for other species.",
    "Herbivore": "It feeds on plants and is food for predators.",
    "Omnivore": "It uses both plant and animal food.",
    "Small Predator": "It helps control insect and small prey numbers.",
    "Predator": "It helps control prey numbers.",
    "Large Predator": "Large-predator classification; presence is not "
                      "assumed without an approved observation.",
    "Decomposer": "It recycles nutrients back into the soil.",
    "Unknown": "Its ecological role is not yet defined in the model.",
}


# ============================================================
# DIET / FOOD WEB
# ============================================================

_RAW_DIET = {
    "Frog": ["Grasshopper", "Caterpillar", "Spider", "Snail"],
    "Lizard": ["Grasshopper", "Caterpillar", "Spider"],
    "Spider": ["Grasshopper", "Caterpillar"],
    "Praying Mantis": ["Grasshopper", "Caterpillar"],
    "Insectivorous Bird": ["Grasshopper", "Caterpillar", "Spider"],
    "African Hoopoe": ["Grasshopper", "Caterpillar"],
    "Fork-tailed Drongo": ["Grasshopper", "Caterpillar", "Spider"],
    "Common Fiscal": ["Grasshopper", "Caterpillar", "Lizard"],
    "African Paradise Flycatcher": ["Grasshopper", "Caterpillar"],
    "Snake": ["Frog", "Mouse", "Rat", "Lizard", "Helmeted Guineafowl"],
    "Mongoose": ["Snake", "Mouse", "Rat", "Frog", "Grasshopper"],
    "Monitor Lizard": ["Frog", "Snake", "Lizard", "Mouse", "Rat",
                       "Helmeted Guineafowl", "Grasshopper"],
    "Eagle": ["Snake", "Mouse", "Rat", "Lizard", "Dove"],
    "Crocodile": ["Impala", "Waterbuck", "Puku"],
    "Impala": ["Grass", "Grass Species", "Miombo Tree", "Acacia"],
    "Zebra": ["Grass", "Grass Species"],
    "Puku": ["Grass", "Plant"],
    "Kudu": ["Acacia", "Miombo Tree", "Brachystegia"],
    "Waterbuck": ["Grass", "Grass Species"],
    "Buffalo": ["Grass", "Grass Species"],
    "Sable Antelope": ["Grass", "Grass Species", "Miombo Tree"],
    "Bushbuck": ["Acacia", "Miombo Tree"],
    "Duiker": ["Grass", "Grass Species"],
    "Warthog": ["Grass", "Grass Species", "Plant"],
    "Rabbit": ["Grass", "Grass Species"],
    "Snail": ["Grass", "Flowering Plant"],
    "Grasshopper": ["Grass", "Grass Species"],
    "Caterpillar": ["Flowering Plant", "Grass"],
    "Helmeted Guineafowl": ["Grass Species"],
    "Cape Turtle Dove": ["Grass Species"],
    "Laughing Dove": ["Grass Species"],
    "Dove": ["Grass Species"],
    "Village Weaver": ["Grass Species"],
    "Southern Red Bishop": ["Grass Species"],
    "Speckled Mousebird": ["Flowering Plant"],
    "African Pied Crow": ["Grasshopper", "Flowering Plant"],
    "Hadada Ibis": ["Earthworm", "Grasshopper"],
    "African Grey Hornbill": ["Grasshopper", "Flowering Plant"],
    "Southern Yellow-billed Hornbill": ["Grasshopper", "Flowering Plant"],
    "Crested Barbet": ["Grasshopper", "Flowering Plant"],
    "Mouse": ["Grass", "Grass Species", "Grasshopper"],
    "Rat": ["Grass", "Grass Species", "Grasshopper"],
    "Rodent": ["Grass", "Grass Species", "Grasshopper"],
    "Civet": ["Mouse", "Rat", "Grasshopper", "Flowering Plant"],
}

SPECIES_DIET = {
    canonical_species_name(species): [canonical_species_name(f) for f in foods]
    for species, foods in _RAW_DIET.items()
}

# Which environmental measurement stands in for each producer.
PRODUCER_SOURCES: Dict[str, Tuple[str, ...]] = {}

for _name in ["Grass", "Grass Species"]:
    PRODUCER_SOURCES[canonical_species_name(_name)] = ("grass_availability",)

for _name in ["Plant", "Flowering Plant"]:
    PRODUCER_SOURCES[canonical_species_name(_name)] = (
        "vegetation_cover", "vegetation_density")

for _name in ["Tree", "Acacia", "Miombo Tree", "Brachystegia", "Julbernardia",
              "Isoberlinia", "Mopane", "Msasa", "Faidherbia Albida",
              "Apple-ring Acacia", "Ana Tree", "Khasi Pine", "Wild loquate",
              "Wild custard apple"]:
    PRODUCER_SOURCES[canonical_species_name(_name)] = ("tree_density",)


# ============================================================
# DATA LOADING
# ============================================================

def get_observation_data() -> pd.DataFrame:
    """Load approved observations, joined to environmental data if present."""
    observations = (Observation.query.filter_by(status="Approved").order_by(Observation.observation_date.asc()).all())

    rows = []
    for obs in observations:
        species = obs.species
        if not species:
            continue
        env = getattr(obs, "environmental_observation", None)
        if env is not None:
            site = env.monitoring_site
            section_key = f"site:{site.id}"
            section_name = site.name
        else:
            site = None
            section_name = species.location or "Unknown Section"
            section_key = "loc:" + normalize_name(section_name)

        area = safe_number(getattr(env, "area_hectares", None)) if env else None
        if (area is None or area <= 0) and site is not None:
            area = safe_number(getattr(site, "area_hectares", None))

        date = (env.observation_date
            if env and getattr(env, "observation_date", None)
            else obs.observation_date
        )

        row = {
            "observation_id": obs.id,
            "environmental_observation_id": env.id if env else None,
            "species_id": species.id,
            "species": species.specie_Common_Name,
            "canonical_species": canonical_species_name(
                species.specie_Common_Name),
            "scientific_name": getattr(species, "scientificName", None),
            "habitat": species.specie_Habitat,
            "section_key": section_key,
            "section_name": section_name,
            "location": section_name,
            "recorded_location_text": getattr(env, "location", None) if env else None,
            "population": safe_number(obs.population_count),
            "date": date,
            "notes": getattr(obs, "notes", None),
        }

        for column in ENVIRONMENT_COLUMNS:
            row[column] = safe_number(getattr(env, column, None)) if env else None

        row["area_hectares"] = area

        source = getattr(env, "water_source_available", None) if env else None
        row["water_source_available"] = None if source is None else bool(source)
        row["water_body_hectares"] = ( safe_number(getattr(env, "water_body_hectares", None)) if env else None)

        rows.append(row)

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def classify_rainfall(rainfall: Any) -> str:
    rainfall = safe_number(rainfall)
    if rainfall is None:
        return "Unknown"
    if rainfall < 0:
        return "Invalid"
    for limit, label in RAINFALL_CLASSES:
        if rainfall < limit:
            return label
    return "Very High"


def classify_availability(score: Optional[float]) -> str:
    if score is None:
        return "Unknown"
    if score < 20:
        return "Very Scarce"
    if score < 40:
        return "Scarce"
    if score < 60:
        return "Moderate"
    if score < 80:
        return "Adequate"
    return "Abundant"


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Clean, deduplicate and derive fields. One row per species/section/survey."""
    if df.empty:
        return df

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    df["population"] = pd.to_numeric(df["population"], errors="coerce")
    df = df.dropna(subset=["species", "population", "date"])
    df = df[df["population"] >= 0]

    if df.empty:
        return df

    df["survey_day"] = df["date"].dt.tz_convert(None).dt.normalize()

    df["co_survey_key"] = np.where( df["environmental_observation_id"].notna(),  "env:" + df["environmental_observation_id"].astype(str),
        "day:" + df["survey_day"].dt.strftime("%Y-%m-%d"),)

    df = df.sort_values(["date", "observation_id"])

    if SAME_SURVEY_POLICY == "sum":
        df["survey_key"] = df["co_survey_key"]
        keys = ["canonical_species", "section_key", "survey_key"]
        aggregation = {c: "first" for c in df.columns
                       if c not in keys + ["population"]}
        aggregation["population"] = "sum"
        df = df.groupby(keys, as_index=False).agg(aggregation)

    elif SAME_SURVEY_POLICY == "latest":
        df["survey_key"] = df["co_survey_key"]
        df = df.drop_duplicates(
            subset=["canonical_species", "section_key", "survey_key"],
            keep="last")

    else:  # "separate"
        df["survey_key"] = "obs:" + df["observation_id"].astype(str)

    df["role"] = df["species"].apply(get_role)
    df["area_hectares"] = pd.to_numeric(df["area_hectares"], errors="coerce")
    df["population_density"] = np.where(
        df["area_hectares"].notna() & (df["area_hectares"] > 0),
        df["population"] / df["area_hectares"],
        np.nan,
    )
    df["rainfall_class"] = df["rainfall"].apply(classify_rainfall)

    df["location_mismatch"] = (
        df["recorded_location_text"].notna()
        & (df["recorded_location_text"].apply(normalize_name)
           != df["section_name"].apply(normalize_name))
    )

    return df.sort_values("date").reset_index(drop=True)


# ============================================================
# BIODIVERSITY
# ============================================================

def compute_biodiversity_index(df: pd.DataFrame) -> pd.DataFrame:
    """Shannon index per section per survey."""
    if df.empty:
        return pd.DataFrame(columns=["section_key", "co_survey_key",
                                     "biodiversity_index", "species_richness"])

    results = []

    for (section_key, survey_key), group in df.groupby(["section_key",
                                                        "co_survey_key"]):
        total = group["population"].sum()

        if total <= 0:
            index = 0.0
        else:
            proportions = group["population"] / total
            proportions = proportions[proportions > 0]
            index = float(-(proportions * np.log(proportions)).sum())

        results.append({ "section_key": section_key, "co_survey_key": survey_key, "biodiversity_index": round(index, 4),
            "species_richness": int(group["canonical_species"].nunique()),
        })

    return pd.DataFrame(results)


# ============================================================
# POPULATION CHANGE  (the part that was not working)
# ============================================================

def _linear_slope(days: np.ndarray, values: np.ndarray) -> Optional[float]:
    """Least-squares slope in individuals per year. None if undefined."""
    if len(days) < 3 or np.ptp(days) == 0:
        return None
    years = days / DAYS_PER_YEAR
    slope, _intercept = np.polyfit(years, values, 1)
    return float(slope)


def summarise_series(group: pd.DataFrame) -> Dict[str, Any]:
    """
    Population change for one species in one section.
    The baseline is the PREVIOUS survey. The old model used the mean of
    every survey including the latest one, which dragged the percentage
    toward zero and reported real change as "Stable".
    """
    sort_columns = ["date"]
    if "observation_id" in group.columns:
        sort_columns.append("observation_id")
    group = group.sort_values(sort_columns)

    populations = group["population"].astype(float).tolist()
    days = (group["survey_day"] - group["survey_day"].min()).dt.days.to_numpy()

    current = populations[-1]
    previous = populations[-2] if len(populations) >= 2 else None

    slope = _linear_slope(days, np.array(populations, dtype=float))

    result = {
        "current": current,
        "previous": previous,
        "first": populations[0],
        "n_surveys": len(populations),
        "slope_per_year": None if slope is None else round(slope, 3),
        "change": None,
        "percentage_change": None,
        "trend": "No Previous Record",
        "small_sample": False,
        "outlier": False,
    }

    # A count far above everything recorded before is more likely a typo
    # than an ecological event. Flag it so it does not silently drive the
    # impact score.
    if len(populations) >= 2:
        history = [p for p in populations[:-1] if p > 0]
        if history and current > OUTLIER_RATIO * float(np.median(history)):
            result["outlier"] = True

    if previous is None:
        return result

    result["change"] = current - previous

    if previous == 0:
        result["trend"] = "New / Reappeared" if current > 0 else "Stable"
        result["percentage_change"] = None if current > 0 else 0.0
        return result

    percentage = (current - previous) / previous * 100.0
    result["percentage_change"] = percentage

    if percentage > TREND_BAND_PCT:
        result["trend"] = "Increasing"
    elif percentage < -TREND_BAND_PCT:
        result["trend"] = "Declining"
    else:
        result["trend"] = "Stable"

    if max(current, previous) < LOW_COUNT_THRESHOLD:
        result["small_sample"] = True

    return result


def build_states(df: pd.DataFrame) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """
    One state per (species, section), plus a park-wide state per species
    used as a fallback when a section has only one survey.
    """
    states: Dict[Tuple[str, str], Dict[str, Any]] = {}

    # Park-wide series per species (sum across sections per survey day).
    park_series = (
        df.groupby(["canonical_species", "survey_day"], as_index=False)
          .agg(population=("population", "sum"),
               date=("date", "first"))
    )

    park_states = {}
    for canon, group in park_series.groupby("canonical_species"):
        park_states[canon] = summarise_series(group)

    for (canon, section_key), group in df.groupby(["canonical_species", "section_key"]):
        group = group.sort_values("date")
        change = summarise_series(group)

        basis = "section"

        # ----------------------------------------------------------
        # FIX 3: fall back to the park-wide series.
        #
        # A species seen once in a section used to return
        # "No Previous Record" forever, which then suppressed the
        # impact score too.
        # ----------------------------------------------------------
        if change["previous"] is None:
            park = park_states.get(canon)
            if park and park["previous"] is not None:
                change = dict(park)
                basis = "park"

        states[(canon, section_key)] = {
            "current_row": group.iloc[-1],
            "previous_row": group.iloc[-2] if len(group) >= 2 else None,
            "change": change,
            "change_basis": basis,
            "observation_count": len(group),
        }

    return states


def change_direction(change: Dict[str, Any]) -> Optional[str]:
    """increase / decline / stable / None."""
    percentage = change["percentage_change"]

    if percentage is None:
        if change["trend"] == "New / Reappeared":
            return "increase"
        slope = change["slope_per_year"]
        if slope is None:
            return None
        if slope > 0:
            return "increase"
        if slope < 0:
            return "decline"
        return "stable"

    if abs(percentage) <= TREND_BAND_PCT:
        return "stable"
    return "increase" if percentage > 0 else "decline"

# ENVIRONMENT

def classify_factor(factor: str, value: float, ranges) -> str:
    if factor in PERCENT_FACTORS and not (0 <= value <= 100):
        return "Invalid"
    s_low, s_high, m_low, m_high = ranges
    if s_low <= value <= s_high:
        return "Suitable"
    if m_low <= value <= m_high:
        return "Marginal"
    return "Unsuitable"


def assess_environment(row: pd.Series, role: str) -> Dict[str, Any]:
    """Suitability by group, a stress ratio, and a one-line reading."""
    points = {"Suitable": 100, "Marginal": 50, "Unsuitable": 0}
    stress_points = {"Suitable": 0, "Marginal": 1, "Unsuitable": 2}
    weights = GROUP_WEIGHTS.get(role, GROUP_WEIGHTS["default"])

    factors: Dict[str, Dict[str, Any]] = {}
    for members in FACTOR_GROUPS.values():
        for factor in members:
            value = safe_number(row.get(factor))
            if value is None:
                continue
            factors[factor] = {
                "value": value,
                "status": classify_factor(factor, value, SUITABILITY_RANGES[factor]),
            }

    groups: Dict[str, str] = {}
    weighted = 0.0
    total_weight = 0.0

    for group, members in FACTOR_GROUPS.items():
        statuses = [factors[f]["status"] for f in members
                    if f in factors and factors[f]["status"] in points]
        if not statuses:
            groups[group] = "Unknown"
            continue
        if "Unsuitable" in statuses:
            groups[group] = "Unsuitable"
        elif "Marginal" in statuses:
            groups[group] = "Marginal"
        else:
            groups[group] = "Suitable"
        weighted += weights[group] * float(np.mean([points[s] for s in statuses]))
        total_weight += weights[group]

    score = weighted / total_weight if total_weight else None

    if score is None:
        overall = "Unknown"
    elif score >= 75:
        overall = "Suitable"
    elif score >= 50:
        overall = "Marginal"
    else:
        overall = "Unsuitable"

    issues = []
    for factor, info in factors.items():
        if info["status"] in ("Marginal", "Unsuitable"):
            issues.append(f"{FACTOR_LABELS[factor]} {info['value']:g} is "
                          f"{info['status'].lower()}")
        elif info["status"] == "Invalid":
            issues.append(f"{FACTOR_LABELS[factor]} {info['value']:g} is outside 0-100")

    stress = sum(stress_points[i["status"]] for i in factors.values()
                 if i["status"] in stress_points)
    possible = 2 * sum(1 for i in factors.values() if i["status"] in stress_points)

    rainfall_class = classify_rainfall(row.get("rainfall"))
    rainfall_points = {"Very Low": 2, "Low": 1, "Moderate": 0,
                       "High": 0, "Very High": 1}
    if rainfall_class in rainfall_points:
        possible += 2
        stress += rainfall_points[rainfall_class]
        if rainfall_points[rainfall_class]:
            issues.append(f"{rainfall_class.lower()} rainfall")

    if possible == 0:
        stress_ratio = None
        interpretation = "No environmental measurements were recorded."
    else:
        stress_ratio = stress / possible
        if stress_ratio >= 0.5:
            interpretation = "Conditions show high environmental stress."
        elif stress_ratio >= 0.3:
            interpretation = "Conditions show moderate environmental stress."
        elif stress_ratio >= 0.1:
            interpretation = "Some environmental stress indicators are present."
        else:
            interpretation = "No major environmental stress was detected."

    return {
        "factors": factors,
        "groups": groups,
        "score": None if score is None else round(score, 1),
        "overall": overall,
        "issues": issues,
        "stress_ratio": stress_ratio,
        "interpretation": interpretation,
        "rainfall_class": rainfall_class,
        "available": possible > 0,
    }


def assess_water_availability(row: pd.Series) -> Dict[str, Any]:
    explicit = row.get("water_source_available")
    if explicit is None or (not isinstance(explicit, (bool, np.bool_))
                            and pd.isna(explicit)):
        explicit = None
    else:
        explicit = bool(explicit)

    body = safe_number(row.get("water_body_hectares"))
    measured = any(safe_number(row.get(c)) is not None
                   for c in ("water_quality", "water_turbidity", "water_ph"))
    moisture = safe_number(row.get("soil_moisture"))
    rainfall_class = row.get("rainfall_class", "Unknown")

    dry = []
    if rainfall_class in ("Very Low", "Low"):
        dry.append("low rainfall")
    if moisture is not None and moisture < 20:
        dry.append("low soil moisture")

    if explicit is False:
        level, basis = "Unavailable", "no water source recorded for this section"
    elif explicit is True or measured or (body is not None and body > 0):
        level = "Limited" if len(dry) >= 2 else "Available"
        if explicit is True:
            basis = "water source recorded"
        elif body is not None and body > 0:
            basis = "water body recorded"
        else:
            basis = "water quality measured"
        if dry:
            basis += "; " + " and ".join(dry)
    elif dry:
        level = "Limited"
        basis = "no direct water source recorded and " + " and ".join(dry)
    elif moisture is None and rainfall_class == "Unknown":
        level, basis = "Unknown", "no water, soil-moisture or rainfall data"
    else:
        level, basis = "Not recorded", "no water source or measurement recorded"

    return {"level": level, "basis": basis,
            "water_body_hectares": body, "dry_indicators": dry}


# ============================================================
# RESOURCE PRESSURE
# ============================================================

def section_forage_index(row: pd.Series) -> Optional[float]:
    values = [safe_number(row.get(c)) for c in
              ("grass_availability", "vegetation_cover", "vegetation_density")]
    values = [v for v in values if v is not None]
    if not values:
        return None
    return float(min(max(np.mean(values), 0), 100))


def calculate_wildlife_units(survey_rows: pd.DataFrame) -> float:
    total = 0.0
    for _, row in survey_rows.iterrows():
        canon = row["canonical_species"]
        if canon not in LARGE_HERBIVORES:
            continue
        population = safe_number(row["population"])
        if population is None:
            continue
        total += population * WILDLIFE_UNIT_FACTORS.get(canon, 1.0)
    return total


def assess_resource_pressure(current: pd.Series,
                             survey_rows: pd.DataFrame,
                             water: Dict[str, Any]) -> Dict[str, Any]:
    forage_index = section_forage_index(current)
    wildlife_units = calculate_wildlife_units(survey_rows)
    area = safe_number(current.get("area_hectares"))

    units_per_ha = None if (area is None or area <= 0) else wildlife_units / area

    if units_per_ha is None:
        pressure = "Unknown"
        basis = "section area is missing, so grazing density cannot be calculated"
    else:
        if units_per_ha < RESOURCE_PRESSURE_THRESHOLDS["low"]:
            pressure = "Low"
        elif units_per_ha < RESOURCE_PRESSURE_THRESHOLDS["moderate"]:
            pressure = "Moderate"
        elif units_per_ha < RESOURCE_PRESSURE_THRESHOLDS["high"]:
            pressure = "High"
        else:
            pressure = "Very High"
        basis = (f"{wildlife_units:.2f} reference grazing units across "
                 f"{area:.2f} ha")

    if forage_index is not None and pressure != "Unknown":
        if forage_index < 20 and pressure in ("High", "Very High"):
            pressure = "Very High"
            basis += f"; forage very low ({forage_index:.0f}/100)"
        elif forage_index < 40 and pressure == "Moderate":
            pressure = "High"
            basis += f"; forage scarce ({forage_index:.0f}/100)"

    if water["level"] == "Unavailable" and pressure != "Unknown":
        pressure = "Very High"
        basis += "; water is unavailable"
    elif water["level"] == "Limited":
        if pressure == "Low":
            pressure = "Moderate"
        elif pressure == "Moderate":
            pressure = "High"
        basis += "; water is limited"

    return {
        "pressure": pressure,
        "wildlife_units": wildlife_units,
        "units_per_hectare": units_per_ha,
        "forage_index": forage_index,
        "area_hectares": area,
        "basis": basis,
    }


# ============================================================
# FOOD WEB
# ============================================================

def build_producer_layer(row: pd.Series) -> Dict[str, Dict[str, Any]]:
    layer = {}
    for producer, fields in PRODUCER_SOURCES.items():
        values = [safe_number(row.get(f)) for f in fields]
        values = [v for v in values if v is not None]
        if not values:
            continue
        score = float(min(max(np.mean(values), 0), 100))
        layer[producer] = {
            "availability": round(score, 1),
            "status": classify_availability(score),
            "source": " + ".join(fields),
        }
    return layer


def resolve_diet(canon: str, layer: Dict[str, Dict[str, Any]],
                 observed_species: set) -> Dict[str, Any]:
    producers, observed, missing = [], [], []

    for item in SPECIES_DIET.get(canon, []):
        if item in layer:
            producers.append({"name": item, **layer[item]})
        elif item in observed_species and item != canon:
            observed.append(item)
        else:
            missing.append(item)

    forage_score = (float(np.mean([p["availability"] for p in producers]))
                    if producers else None)

    return {
        "producers": producers,
        "observed_resources": sorted(set(observed)),
        "missing": sorted(set(missing)),
        "forage_score": forage_score,
        "forage_status": classify_availability(forage_score),
    }


def get_consumers(canon: str, observed_species: set) -> List[str]:
    return sorted(
        species for species, diet in SPECIES_DIET.items()
        if species != canon and species in observed_species and canon in diet
    )


def generate_food_web_effect(ctx: Dict[str, Any]) -> str:
    species = pretty_name(ctx["species"])
    direction = ctx["direction"]

    food = [pretty_name(p["name"]) for p in ctx["diet"]["producers"]]
    prey = [pretty_name(r) for r in ctx["diet"]["observed_resources"]]
    consumers = [pretty_name(c) for c in ctx["consumers"]]

    if not (food or prey or consumers):
        return "No food-web link is recorded for this section."

    if direction is None:
        return ("Food-web links are mapped, but there is not enough population "
                "history to say how they are changing.")

    if direction == "stable":
        return f"{species} is stable, so no food-web change is expected."

    effects = []
    if direction == "increase":
        if food:
            effects.append("more demand on " + ", ".join(food))
        if prey:
            effects.append("more pressure on " + ", ".join(prey))
        if consumers:
            effects.append("more food for " + ", ".join(consumers))
    else:
        if food:
            effects.append("less demand on " + ", ".join(food))
        if prey:
            effects.append("less pressure on " + ", ".join(prey))
        if consumers:
            effects.append("less food for " + ", ".join(consumers))

    if not effects:
        return f"No directional food-web effect was inferred for {species}."

    return f"The {direction} in {species} means " + " and ".join(effects) + "."


# ============================================================
# ENVIRONMENTAL CONTEXT
#
# A population change is not automatically a "threat". A decline in a
# section with low rainfall, scarce forage, limited water or heavy
# grazing pressure is plausibly a resource-driven adjustment; a decline
# with none of those present is unexplained by the recorded data and
# deserves more attention, not less. This function is the single place
# that reads rainfall, forage, water and grazing pressure together, so
# park-effect reasoning and impact scoring can't disagree with each
# other about what "stressed" means.
# ============================================================

def assess_environmental_context(ctx: Dict[str, Any]) -> Dict[str, Any]:
    factors = []

    rainfall_class = ctx["environment"]["rainfall_class"]
    if rainfall_class in ("Very Low", "Low"):
        factors.append(f"{rainfall_class.lower()} rainfall")

    forage = ctx["diet"]["forage_status"]
    if forage in ("Scarce", "Very Scarce"):
        factors.append(f"{forage.lower()} food availability")

    water = ctx["water"]["level"]
    if water in ("Limited", "Unavailable"):
        factors.append(f"{water.lower()} water")

    pressure = ctx["resource"]["pressure"]
    if pressure in ("High", "Very High"):
        factors.append(f"{pressure.lower()} grazing/resource pressure")

    count = len(factors)

    if count >= 2:
        level = "High"
    elif count == 1:
        level = "Some"
    else:
        level = "None"

    return {"factors": factors, "count": count, "level": level}


# ============================================================
# PARK EFFECT
# ============================================================

def evaluate_increase(ctx: Dict[str, Any]) -> Tuple[str, List[str]]:
    role = ctx["role"]
    canon = ctx["canon"]
    species = pretty_name(ctx["species"])
    consumers = ctx["consumers"]
    context = ctx["environmental_context"]

    if role == "Producer":
        reasons = ["More vegetation means more food and shelter."]
        if consumers:
            reasons.append(f"It supports {names_text(consumers)}.")
        return "Beneficial", reasons

    if canon in LARGE_HERBIVORES or role in ("Herbivore", "Omnivore"):
        if context["level"] == "High":
            return "Concern", [
                f"The section already shows limited resources, so more "
                f"{species} would add to an already strained base."]
        if context["level"] == "Some":
            return "Concern", [
                f"{species} is increasing while resources here are already "
                f"somewhat limited. Worth watching."]
        if role == "Herbivore" and consumers:
            return "Beneficial", [f"{species} is food for {names_text(consumers)}."]
        return "Monitor", [f"Resources currently look adequate for {species}."]

    if role in PREDATOR_ROLES:
        prey = ctx["diet"]["observed_resources"]
        if not prey:
            return "Uncertain", ["No prey species were recorded in this section."]
        declining = [p for p, t in ctx["prey_trends"].items() if t == "Declining"]
        if declining:
            return "Concern", [f"Prey are already declining: {names_text(declining)}."]
        return "Monitor", [f"Prey recorded here: {names_text(prey)}."]

    if role == "Decomposer":
        return "Beneficial", ["Faster nutrient recycling supports soil health."]

    return "Uncertain", ["The ecological role is not defined in the model."]


def evaluate_decline(ctx: Dict[str, Any]) -> Tuple[str, List[str]]:
    role = ctx["role"]
    species = pretty_name(ctx["species"])
    consumers = ctx["consumers"]
    context = ctx["environmental_context"]

    if role == "Producer":
        if consumers:
            return "Concern", [f"Less food and shelter for "
                               f"{names_text(consumers)}."]
        return "Monitor", ["No dependent species were recorded here."]

    if role in ("Herbivore", "Omnivore") or ctx["canon"] in LARGE_HERBIVORES:
        if context["level"] == "High":
            return "Potentially Beneficial", [
                "Likely a natural correction as conditions ease; confirm "
                "with a follow-up count rather than treating it as urgent."]
        if context["level"] == "Some":
            return "Concern", [
                "Worth watching whether the decline continues once "
                "conditions improve."]
        return "Concern", [f"Nothing in the current environmental data points "
                           f"to a cause for {species}'s decline."]

    if role in PREDATOR_ROLES:
        prey = ctx["diet"]["observed_resources"]
        if prey:
            return "Concern", [f"Weaker natural control of {names_text(prey)}."]
        return "Monitor", ["No prey species were recorded here."]

    if role == "Decomposer":
        return "Concern", ["Slower nutrient recycling may reduce soil fertility."]

    if consumers:
        return "Concern", [f"A decline could affect {names_text(consumers)}."]

    return "Monitor", ["No major dependency was recorded."]


def evaluate_park_effect(ctx: Dict[str, Any]) -> Tuple[str, List[str]]:
    direction = ctx["direction"]
    if direction == "increase":
        return evaluate_increase(ctx)
    if direction == "decline":
        return evaluate_decline(ctx)
    if direction == "stable":
        return "Neutral", ["The population is holding steady."]
    return "Insufficient Evidence", ["There is no previous survey to compare with."]


# ============================================================
# EVIDENCE (confidence only - it no longer blocks the result)
# ============================================================

def calculate_evidence(observation_count: int, has_previous: bool,
                       environmental_data_count: int,
                       food_web_links: int) -> Dict[str, Any]:
    score = 0
    reasons = []

    if observation_count >= 5:
        score += 3
    elif observation_count >= 3:
        score += 2
    elif observation_count >= 2:
        score += 1
    else:
        reasons.append("only one approved observation")

    if has_previous:
        score += 2
    else:
        reasons.append("no previous survey in this section")

    if environmental_data_count >= 8:
        score += 2
    elif environmental_data_count >= 4:
        score += 1
    else:
        reasons.append("limited environmental measurements")

    if food_web_links >= 2:
        score += 2
    elif food_web_links == 1:
        score += 1
    else:
        reasons.append("few recorded food-web links")

    if score >= 7:
        level = "High"
    elif score >= 4:
        level = "Moderate"
    else:
        level = "Limited"

    return {"score": score, "level": level, "reasons": reasons}

def calculate_impact(ctx: Dict[str, Any], evidence: Dict[str, Any]) -> Dict[str, Any]:
 
    # ============================================================
    # GET POPULATION INFORMATION
    # ============================================================

    population = ctx.get("population", {})

    current_population = (
        population.get("current")
        if population.get("current") is not None
        else population.get("population")
    )

    previous_population = population.get("previous")

    percentage_change = population.get(
        "percentage_change"
    )

    direction = (
        population.get("direction")
        or ctx.get("direction")
    )

    # ============================================================
    # CONVERT NUMERIC VALUES SAFELY
    # ============================================================

    try:
        current_population = (
            float(current_population)
            if current_population is not None
            else None
        )
    except (TypeError, ValueError):
        current_population = None

    try:
        previous_population = (
            float(previous_population)
            if previous_population is not None
            else None
        )
    except (TypeError, ValueError):
        previous_population = None

    try:
        percentage_change = (
            float(percentage_change)
            if percentage_change is not None
            else None
        )
    except (TypeError, ValueError):
        percentage_change = None

    
    if (
        current_population is not None
        and previous_population is not None
    ):

        if current_population > previous_population:
            direction = "increase"

        elif current_population < previous_population:
            direction = "decline"

        else:
            direction = "stable"

    # ============================================================
    # RESOURCE INFORMATION
    # ============================================================

    resource = ctx.get(
        "resource",
        {}
    )

    resource_pressure = resource.get(
        "pressure"
    )

    # ============================================================
    # LAND AREA
    #
    # ALL LAND VALUES ARE ASSUMED TO BE HECTARES.
    # ============================================================

    area_hectares = resource.get(
        "area_hectares"
    )

    if area_hectares is None:
        area_hectares = resource.get(
            "area"
        )

    if area_hectares is None:
        area_hectares = ctx.get(
            "area_hectares"
        )

    try:
        area_hectares = (
            float(area_hectares)
            if area_hectares is not None
            else None
        )
    except (TypeError, ValueError):
        area_hectares = None

    # ============================================================
    # ENVIRONMENT
    # ============================================================

    park_effect = ctx.get(
        "park_effect"
    )

    environmental_context = ctx.get(
        "environmental_context",
        {}
    )

    stress_factors = environmental_context.get(
        "factors",
        []
    )

    # RESOURCE RISK

    resource_risk = {
        "Very High": 3,
        "High": 2,
        "Moderate": 1,
        "Low": 0,
        "Unknown": None
    }.get(
        resource_pressure
    )

    # ============================================================
    # PARK EFFECT RISK
    # ============================================================

    park_effect_risk = {
        "Harmful": 3,
        "Concern": 2,
        "Monitor": 1,
        "Potentially Beneficial": 0,
        "Beneficial": 0,
        "Neutral": 0,
        "Uncertain": None,
        "Insufficient Evidence": None
    }.get(
        park_effect
    )

    # ============================================================
    # OUTLIER PROTECTION
    # ============================================================

    if population.get("outlier"):

        return {
            "score": None,
            "level": None,
            "label": "Check Data",

            "confidence": evidence.get(
                "level",
                "Unknown"
            ),

            "population": current_population,
            "previous_population": previous_population,

            "population_category": "Unknown",

            "population_risk": None,
            "base_population_risk": None,

            "percentage_change": percentage_change,
            "direction": direction,

            "area_hectares": area_hectares,
            "population_density": None,
            "density_category": "Unknown",

            "resource_pressure": resource_pressure,
            "resource_risk": resource_risk,

            "park_effect_risk": park_effect_risk,

            "context_note": (
                "The population count differs substantially from "
                "previous observations and should be checked before "
                "ecological impact is interpreted."
            ),

            "stress_factors": stress_factors
        }

    # ============================================================
    # POPULATION INFORMATION CHECK
    

    if current_population is None:

        return {
            "score": None,
            "level": None,
            "label": "Insufficient Evidence",

            "confidence": evidence.get(
                "level",
                "Unknown"
            ),

            "population": None,
            "previous_population": previous_population,

            "population_category": "Unknown",

            "population_risk": None,
            "base_population_risk": None,

            "percentage_change": percentage_change,
            "direction": direction,

            "area_hectares": area_hectares,
            "population_density": None,
            "density_category": "Unknown",

            "resource_pressure": resource_pressure,
            "resource_risk": resource_risk,

            "park_effect_risk": park_effect_risk,

            "context_note": (
                "A valid current population count is required "
                "to determine ecological impact."
            ),

            "stress_factors": stress_factors
        }

    # POPULATION CATEGORY

    if current_population < 30:
        population_category = "Low"

    elif current_population < 50:
        population_category = "Moderate"

    elif current_population < 100:
        population_category = "High"

    else:
        population_category = "Very High"


    # LAND AREA CHECK
    

    if (
        area_hectares is None
        or area_hectares <= 0
    ):

        return {
            "score": None,
            "level": None,
            "label": "Insufficient Evidence",

            "confidence": evidence.get(  "level", "Unknown"),

            "population": current_population,
            "previous_population": previous_population,

            "population_category": population_category,

            "population_risk": None,
            "base_population_risk": None,

            "percentage_change": percentage_change,
            "direction": direction,

            "area_hectares": None,
            "population_density": None,
            "density_category": "Unknown",

            "resource_pressure": resource_pressure,
            "resource_risk": resource_risk,

            "park_effect_risk": park_effect_risk,

            "context_note": (
                "The land area is missing or invalid. Land area "
                "must be recorded in hectares before population "
                "pressure can be evaluated relative to available "
                "space."
            ),

            "stress_factors": stress_factors
        }


    # RESOURCE INFORMATION CHECK

    if resource_risk is None:

        return {
            "score": None,
            "level": None,
            "label": "Insufficient Evidence",

            "confidence": evidence.get("level","Unknown"),

            "population": current_population,
            "previous_population": previous_population,
            "population_category": population_category,

            "population_risk": None,
            "base_population_risk": None,

            "percentage_change": percentage_change,
            "direction": direction,

            "area_hectares": area_hectares,
            "population_density": None,
            "density_category": "Unknown",

            "resource_pressure": None,
            "resource_risk": None,

            "park_effect_risk": park_effect_risk,

            "context_note": (
                "Resource pressure information is unavailable. "
                "Population trend can be identified, but ecological "
                "pressure cannot be classified reliably without "
                "resource information."
            ),

            "stress_factors": stress_factors
        }

    population_density = (current_population /area_hectares)
    if population_density < 1:
        density_category = "Low"

    elif population_density < 5:
        density_category = "Moderate"

    elif population_density < 10:
        density_category = "High"

    else:
        density_category = "Very High"

    if direction == "increase":
        population_risk = 1

    else:
        population_risk = 0

    base_population_risk = population_risk
    
    # DECREASING POPULATION
    if direction == "decline":

        ecological_score = 0

        context_note = (
            f"The population has decreased to "
            f"{current_population:g} from "
            f"{previous_population:g}."
            if previous_population is not None
            else
            f"The current population is "
            f"{current_population:g} and is showing a decline."
        )

        context_note += (
            f" The available area is "
            f"{area_hectares:g} hectares, giving a population "
            f"density of {population_density:.2f} per hectare. "
            "The declining population is classified as Low "
            "ecological impact because population demand is "
            "decreasing rather than increasing."
        )

    # ============================================================
    # STABLE POPULATION
    # ============================================================

    elif direction == "stable":

        # Stable + very high density + very high resources
        # can still represent substantial pressure.

        if (
            density_category == "Very High"
            and resource_pressure == "Very High"
        ):

            ecological_score = 2

            context_note = (
                f"The population is stable at "
                f"{current_population:g}, but population density "
                f"is very high at {population_density:.2f} per "
                f"hectare across {area_hectares:g} hectares. "
                "Resources are also very highly constrained. "
                "Although the population is not increasing, the "
                "current population is placing substantial pressure "
                "on the available land and resources."
            )

        elif (
            density_category in (
                "High",
                "Very High"
            )
            and resource_pressure in (
                "High",
                "Very High"
            )
        ):

            ecological_score = 2

            context_note = (
                f"The population is stable at "
                f"{current_population:g}, with a population density "
                f"of {population_density:.2f} per hectare. "
                f"Resources are under {resource_pressure.lower()} "
                "pressure. Current conditions indicate substantial "
                "ecological pressure even though the population is "
                "not currently increasing."
            )

        else:

            ecological_score = 1

            context_note = (
                f"The population is stable at "
                f"{current_population:g}, with "
                f"{area_hectares:g} hectares available and a "
                f"population density of {population_density:.2f} "
                "per hectare. Current population conditions are "
                "generally stable."
            )

    # ============================================================
    # INCREASING POPULATION
    # ============================================================

    elif direction == "increase":

        if (
            density_category == "Very High"
            and resource_pressure in (
                "High",
                "Very High"
            )
        ):

            ecological_score = 3

            context_note = (
                f"The population is increasing and is currently "
                f"{current_population:g}. The population density "
                f"is very high at {population_density:.2f} per "
                f"hectare across {area_hectares:g} hectares. "
                f"Resources are under {resource_pressure.lower()} "
                "pressure. The combination of increasing population, "
                "very high land pressure and constrained resources "
                "indicates critical ecological pressure."
            )
        # HIGH
        # Increasing population should normally be High.


        else:

            ecological_score = 2

            context_note = (
                f"The population is increasing and is currently "
                f"{current_population:g}. The available area is "
                f"{area_hectares:g} hectares, giving a population "
                f"density of {population_density:.2f} per hectare. "
                f"Resources are under {resource_pressure.lower()} "
                "pressure. The increase in population represents "
                "higher ecological demand and is therefore "
                "classified as High impact."
            )

    # ============================================================
    # UNKNOWN DIRECTION
    # ============================================================

    else:

        ecological_score = 1

        context_note = (
            f"The current population is "
            f"{current_population:g}, with a density of "
            f"{population_density:.2f} per hectare across "
            f"{area_hectares:g} hectares. The population trend "
            "could not be determined, so the impact is classified "
            "as Stable pending additional observations."
        )

    # ============================================================
    # POPULATION TREND INFORMATION
    # ============================================================

    if (
        percentage_change is not None
        and direction == "increase"
    ):

        context_note += (
            f" The population increased by approximately "
            f"{percentage_change:.1f}% compared with the previous "
            "observation."
        )

    elif (
        percentage_change is not None
        and direction == "decline"
    ):

        context_note += (
            f" The population decreased by approximately "
            f"{abs(percentage_change):.1f}% compared with the "
            "previous observation."
        )

    # ============================================================
    # ENVIRONMENTAL STRESS
    # ============================================================

    if stress_factors:

        context_note += (
            " Environmental stress factors were also recorded "
            "and should be considered when interpreting future "
            "population changes."
        )

    # ============================================================
    # PARK EFFECT


    if park_effect == "Harmful":

        context_note += (
            " A harmful park effect was recorded and may add "
            "additional ecological pressure."
        )

    elif park_effect == "Concern":

        context_note += (
            " A park-related concern was recorded and should "
            "continue to be monitored."
        )

    # ============================================================
    # SAFETY LIMIT
    # ============================================================

    ecological_score = int(
        max(
            0,
            min(
                3,
                ecological_score
            )
        )
    )

    level = ecological_score

    # ============================================================
    # FINAL RESULT
    # ============================================================

    return {
        "score": float(level),

        "level": level,

        "label": IMPACT_LABELS[level],

        "confidence": evidence.get(
            "level",
            "Unknown"
        ),

        # --------------------------------------------------------
        # POPULATION
        # --------------------------------------------------------

        "population": current_population,

        "previous_population": previous_population,

        "population_category": population_category,

        "population_risk": population_risk,

        "base_population_risk": base_population_risk,

        "percentage_change": percentage_change,

        "direction": direction,

        # --------------------------------------------------------
        # LAND
        # --------------------------------------------------------

        "area_hectares": area_hectares,

        "population_density": population_density,

        "density_category": density_category,

        # --------------------------------------------------------
        # RESOURCES
        # --------------------------------------------------------

        "resource_pressure": resource_pressure,

        "resource_risk": resource_risk,

        # --------------------------------------------------------
        # OTHER
        # --------------------------------------------------------

        "park_effect_risk": park_effect_risk,

        "context_note": context_note,

        "stress_factors": stress_factors
    }
# ============================================================
# INTERPRETATION
# Simple, clear and direct
# ============================================================

def build_summary(
    ctx: Dict[str, Any],
    impact: Dict[str, Any]
) -> str:
    """
    Build the short interpretation shown on the main
    biodiversity analysis page.

    Population trend:
        Declining -> resource demand is decreasing
        Stable   -> resource demand is relatively stable
        Increasing -> resource demand is increasing

    The final Impact label comes from calculate_impact().
    """

    # ========================================================
    # BASIC INFORMATION
    # ========================================================

    species = pretty_name(
        ctx.get(
            "species",
            "Unknown species"
        )
    )

    section = ctx.get(
        "section_name",
        "Monitoring Site"
    )

    population = ctx.get(
        "population",
        {}
    )

    # ========================================================
    # POPULATION VALUES
    # ========================================================

    current = population.get(
        "current"
    )

    previous = population.get(
        "previous"
    )

    percentage = population.get(
        "percentage_change"
    )

    # ========================================================
    # DIRECTION
    # ========================================================

    direction = (
        population.get("direction")
        or ctx.get("direction")
    )

    # ========================================================
    # IMPACT
    # ========================================================

    impact_label = impact.get(
        "label",
        "Insufficient Evidence"
    )

    # ========================================================
    # FIRST SURVEY
    # ========================================================

    if previous is None:

        if current is None:

            return (
                f"{species} in {section}: "
                "No current population was recorded. "
                "There is not enough information to determine "
                "population pressure."
            )

        current = int(current)

        return (
            f"{species} in {section}: "
            f"{current} recorded. "
            "This is the first survey, so population change "
            "cannot yet be determined."
        )

    # ========================================================
    # SAFE CONVERSION
    # ========================================================

    try:
        current = int(current)
    except (TypeError, ValueError):
        current = 0

    try:
        previous = int(previous)
    except (TypeError, ValueError):
        previous = 0

    # ========================================================
    # POPULATION CHANGE TEXT
    # ========================================================

    if percentage is not None:

        try:
            percentage = float(
                percentage
            )

            population_text = (
                f"{species} in {section} changed "
                f"from {previous} to {current} "
                f"({percentage:+.0f}%)."
            )

        except (TypeError, ValueError):

            population_text = (
                f"{species} in {section} changed "
                f"from {previous} to {current}."
            )

    else:

        population_text = (
            f"{species} in {section} changed "
            f"from {previous} to {current}."
        )

    if direction == "decline":

        trend_text = (
            "The population is decreasing, so its demand "
            "for food, water, vegetation and habitat is "
            "also decreasing."
        )

    elif direction == "increase":

        trend_text = (
            "The population is increasing, so its demand "
            "for food, water, vegetation and habitat is "
            "also increasing."
        )

    elif direction == "stable":

        trend_text = (
            "The population is stable, so its demand "
            "for food, water, vegetation and habitat "
            "is relatively stable."
        )

    else:

        trend_text = (
            "The population trend could not be determined."
        )

    # ========================================================
    # IMPACT INTERPRETATION
    #
    # This uses the result from calculate_impact().
    # It does NOT recalculate impact here.
    # ========================================================

    if impact_label == "Low":

        impact_text = (
            "Overall ecological pressure is low."
        )

    elif impact_label == "Stable":

        impact_text = (
            "Overall ecological pressure is stable "
            "and currently manageable."
        )

    elif impact_label == "High":

        impact_text = (
            "Overall ecological pressure is high and "
            "should be monitored."
        )

    elif impact_label == "Critical":

        impact_text = (
            "Overall ecological pressure is critical "
            "and requires close monitoring."
        )


    else:

        impact_text = ( "" )

    # ========================================================
    # OUTLIER
    # ========================================================

    if population.get(
        "outlier"
    ):

        impact_text = (
            "The recorded population is unusually different "
            "from previous observations and should be checked."
        )

    # ========================================================
    # SMALL SAMPLE
    # ========================================================

    small_sample_text = ""

    if population.get(
        "small_sample"
    ):

        small_sample_text = (
            " The population is small, so the percentage "
            "change should be interpreted carefully."
        )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    return (
        f"{population_text} "
        f"{trend_text} "
        f"Impact: {impact_label}. "
        f"{impact_text}"
        f"{small_sample_text}"
    )
# ============================================================
# SUPPORTING DETAILS
# ============================================================

def build_details(
    ctx: Dict[str, Any]
) -> List[str]:
    """
    Build the technical information shown under
    'More Details'.
    """

    details = []

    # ========================================================
    # DATA CHECK
    # ========================================================

    if ctx.get("location_mismatch"):

        recorded_location = ctx.get(
            "recorded_location_text",
            "Unknown"
        )

        section = ctx.get(
            "section_name",
            "Unknown"
        )

        details.append(
            f"Data check: recorded location "
            f"'{recorded_location}' does not match "
            f"monitoring site '{section}'."
        )

    # ========================================================
    # SPECIES ROLE
    # ========================================================

    role = ctx.get("role")

    if role:

        role_text = ROLE_TEXT.get(
            role,
            ""
        )

        if role_text:

            details.append(
                f"Ecological role: {role}. "
                f"{role_text}"
            )

        else:

            details.append(
                f"Ecological role: {role}."
            )

    # ========================================================
    # LONG-TERM TREND
    # ========================================================

    population = ctx.get(
        "population",
        {}
    )

    slope = population.get(
        "slope_per_year"
    )

    if slope is not None:

        details.append(
            f"Long-term population trend: "
            f"{slope:+.1f} individuals per year "
            f"across {population.get('n_surveys', 0)} surveys."
        )

    
    # ENVIRONMENT
    environment = ctx.get(
        "environment",
        {}
    )

    interpretation = environment.get(
        "interpretation"
    )

    if interpretation:

        details.append(
            f"Environment: {interpretation}"
        )

    # ========================================================
    # ENVIRONMENTAL CONDITIONS
    # ========================================================

    groups = environment.get(
        "groups",
        {}
    )

    if groups:

        temperature = groups.get(
            "temperature",
            "Unknown"
        )

        water_condition = groups.get(
            "water",
            "Unknown"
        )

        soil = groups.get(
            "soil",
            "Unknown"
        )

        details.append(
            f"Conditions: temperature "
            f"{temperature.lower()}, "
            f"water {water_condition.lower()}, "
            f"soil {soil.lower()}."
        )

    # POPULATION DENSITY
    resource = ctx.get( "resource",{})

    area = resource.get("area_hectares")

    density = ctx.get(  "current_density")
    if area is not None and density is not None:

        details.append(
            f"Population density: "
            f"{density:.2f} individuals/ha "
            f"over {area:.2f} ha."
        )

    # ========================================================
    # WATER
    # ========================================================

    water = ctx.get(
        "water",
        {}
    )

    water_level = water.get(
        "level"
    )

    water_basis = water.get(
        "basis"
    )

    if water_level:

        water_text = (
            f"Water availability: "
            f"{water_level.lower()}"
        )

        if water_basis:
            water_text += ( f" ({water_basis})")
        details.append( water_text + ".")

    # ========================================================
    # RESOURCE PRESSURE
    # ========================================================

    units = resource.get(
        "units_per_hectare"
    )

    pressure = resource.get(
        "pressure"
    )

    if units is not None:

        pressure_text = (
            pressure.lower()
            if pressure
            else "unknown"
        )

        details.append( f"Resource pressure: " f"{units:.2f} reference units/ha "  f"({pressure_text}).")

    elif pressure:
        details.append(f"Resource pressure: " f"{pressure.lower()}.")

    # ENVIRONMENTAL STRESS
    environmental_context = ctx.get("environmental_context",{})
    factors = environmental_context.get( "factors", [])

    if factors:
        details.append(   "Recorded environmental stress: "     + "; ".join(factors)  + ".")

    else:
        details.append("No recorded stress affecting ""water, vegetation, food or habitat." )


    # FOOD / VEGETATION RESOURCES
    diet = ctx.get("diet",{})

    producers = diet.get("producers", [] )
    if producers:
        food_resources = []

        for producer in producers:
            name = pretty_name(producer.get("name","Unknown"))
            availability = producer.get("availability")
            status = producer.get("status", "Unknown")

            if availability is not None:
                food_resources.append(   f"{name} "  f"{availability:.0f}/100 "  f"({status.lower()})")

            else:
                food_resources.append(f"{name} "  f"({status.lower()})")

        details.append( "Vegetation/food resources: " + "; ".join(food_resources) + ".")

    # FOOD WEB
    food_web_effect = ctx.get( "food_web_effect" )

    if food_web_effect:
        details.append(f"Food-web effect: " f"{food_web_effect}")


    # CONFIDENCE
    evidence = ctx.get( "evidence",{})

    confidence_level = evidence.get("level","Unknown")

    confidence_reasons = evidence.get("reasons",  [])

    confidence_text = ( f"Confidence: " f"{confidence_level.lower()}" )

    if confidence_reasons:
        confidence_text += ( " - " + "; ".join(confidence_reasons) )

    details.append(confidence_text + ".")
    details.append("This interpretation is based on recorded monitoring " "data and does not by itself prove the cause of " "population change.")
    return details

def build_details(ctx: Dict[str, Any]) -> List[str]:
    """Supporting lines. Show these behind a 'more detail' toggle."""
    details = []

    if ctx["location_mismatch"]:
        details.append(
            f"Data check: the recorded location text "
            f"('{ctx['recorded_location_text']}') does not match the "
            f"monitoring site name ('{ctx['section_name']}'). This does not "
            f"affect the analysis, but the two fields have drifted apart "
            f"and are worth reconciling.")

    details.append(f"Role: {ctx['role']}. {ROLE_TEXT.get(ctx['role'], '')}")

    if ctx["population"]["slope_per_year"] is not None:
        details.append(f"Long-term trend: "
                       f"{ctx['population']['slope_per_year']:+.1f} "
                       f"individuals per year across "
                       f"{ctx['population']['n_surveys']} surveys.")

    details.append(f"Environment: {ctx['environment']['interpretation']}")

    groups = ctx["environment"]["groups"]
    details.append(f"Suitability - temperature {groups['temperature'].lower()}, "
                   f"water {groups['water'].lower()}, soil {groups['soil'].lower()}.")

    area = ctx["resource"]["area_hectares"]
    if area is not None and ctx["current_density"] is not None:
        details.append(f"Density: {ctx['current_density']:.2f} per ha "
                       f"over {area:.2f} ha.")

    details.append(f"Water: {ctx['water']['level'].lower()} "
                   f"({ctx['water']['basis']}).")

    if ctx["resource"]["units_per_hectare"] is not None:
        details.append(f"Grazing pressure: "
                       f"{ctx['resource']['units_per_hectare']:.2f} reference "
                       f"units/ha - {ctx['resource']['pressure'].lower()}.")

    context = ctx["environmental_context"]
    if context["factors"]:
        details.append("Resource stress recorded: " +
                       "; ".join(context["factors"]) + ".")
    else:
        details.append("No resource stress (rainfall, food, water, grazing "
                       "pressure) was recorded for this survey.")

    if ctx["diet"]["producers"]:
        details.append("Food resources: " + "; ".join(
            f"{pretty_name(p['name'])} {p['availability']:.0f}/100 "
            f"({p['status'].lower()})" for p in ctx["diet"]["producers"]))

    details.append(ctx["food_web_effect"])

    confidence = f"Confidence: {ctx['evidence']['level'].lower()}"
    if ctx["evidence"]["reasons"]:
        confidence += " - " + "; ".join(ctx["evidence"]["reasons"])
    details.append(confidence + ".")

    details.append("Screening interpretation from recorded data, not proof "
                   "of cause.")

    return details


# ============================================================
# MAIN REPORT
# ============================================================

def generate_species_effect_report() -> pd.DataFrame:
    raw = get_observation_data()
    if raw.empty:
        return pd.DataFrame()

    df = prepare_features(raw)
    if df.empty:
        return pd.DataFrame()

    biodiversity = compute_biodiversity_index(df)
    biodiversity_lookup = {
        (r["section_key"], r["co_survey_key"]): r
        for _, r in biodiversity.iterrows()
    }

    states = build_states(df)
    results = []

    for (canon, section_key), state in sorted(
            states.items(),
            key=lambda i: (str(i[1]["current_row"]["section_name"]),
                           str(i[1]["current_row"]["species"]))):

        current = state["current_row"]
        change = state["change"]
        species = current["species"]
        role = current["role"]

        direction = change_direction(change)

        section_rows = df[df["section_key"] == section_key]
        observed_species = set(section_rows["canonical_species"])
        survey_rows = section_rows[
            section_rows["co_survey_key"] == current["co_survey_key"]]

        environment = assess_environment(current, role)
        water = assess_water_availability(current)
        resource = assess_resource_pressure(current, survey_rows, water)

        producer_layer = build_producer_layer(current)
        diet = resolve_diet(canon, producer_layer, observed_species)
        consumers = get_consumers(canon, observed_species)

        prey_trends = {}
        for prey in diet["observed_resources"]:
            key = (prey, section_key)
            if key in states:
                prey_trends[prey] = states[key]["change"]["trend"]

        environmental_count = sum(
            safe_number(current.get(c)) is not None for c in ENVIRONMENT_COLUMNS)
        food_link_count = (len(diet["producers"])
                           + len(diet["observed_resources"]) + len(consumers))

        evidence = calculate_evidence(
            observation_count=state["observation_count"],
            has_previous=state["previous_row"] is not None,
            environmental_data_count=environmental_count,
            food_web_links=food_link_count,
        )

        current_density = safe_number(current.get("population_density"))
        previous_density = (safe_number(state["previous_row"].get("population_density"))
                            if state["previous_row"] is not None else None)

        ctx = {
            "species": species,
            "canon": canon,
            "role": role,
            "section_name": current["section_name"],
            "location_mismatch": bool(current.get("location_mismatch", False)),
            "recorded_location_text": current.get("recorded_location_text"),
            "population": change,
            "change_basis": state["change_basis"],
            "direction": direction,
            "current_density": current_density,
            "environment": environment,
            "water": water,
            "resource": resource,
            "diet": diet,
            "consumers": consumers,
            "prey_trends": prey_trends,
            "evidence": evidence,
        }

        ctx["environmental_context"] = assess_environmental_context(ctx)

        park_effect, park_reasons = evaluate_park_effect(ctx)
        ctx["park_effect"] = park_effect
        ctx["park_reasons"] = park_reasons
        ctx["food_web_effect"] = generate_food_web_effect(ctx)

        impact = calculate_impact(ctx, evidence)

        summary = build_summary(ctx, impact)
        details = build_details(ctx)

        biodiversity_row = biodiversity_lookup.get(
            (section_key, current["co_survey_key"]), {})

        food_resources = sorted({pretty_name(p["name"]) for p in diet["producers"]})
        observed_food_web = sorted({pretty_name(i) for i in diet["observed_resources"]})
        consumers_pretty = sorted({pretty_name(i) for i in consumers})

        results.append({
            # identity
            "species": species,
            "species_id": int(current["species_id"]),
            "scientific_name": current["scientific_name"],
            "habitat": current["habitat"],
            "section": current["section_name"],
            "location": current["section_name"],
            "role": role,

            # population
            "current_population": int(change["current"]),
            "previous_population": (None if change["previous"] is None
                                    else int(change["previous"])),
            "population_change": change["change"],
            "percentage_change": change["percentage_change"],
            "percent_change": change["percentage_change"],
            "trend": change["trend"],
            "trend_slope_per_year": change["slope_per_year"],
            "change_basis": state["change_basis"],
            "small_sample": change["small_sample"],
            "data_quality_flag": ("Possible data-entry error"
                                  if change.get("outlier") else None),
            "observation_count": state["observation_count"],
            "current_density": current_density,
            "previous_density": previous_density,

            # biodiversity
            "biodiversity_index": biodiversity_row.get("biodiversity_index"),
            "species_richness": biodiversity_row.get("species_richness"),

            # area and pressure
            "area_hectares": resource["area_hectares"],
            "wildlife_grazing_units": resource["wildlife_units"],
            "grazing_units_per_hectare": resource["units_per_hectare"],
            "resource_pressure": resource["pressure"],
            "resource_pressure_basis": resource["basis"],
            "forage_index": resource["forage_index"],

            # water
            "water_availability": water["level"],
            "water_basis": water["basis"],
            "water_body_hectares": water["water_body_hectares"],

            # environment
            "environment_interpretation": environment["interpretation"],
            "environment_stress_ratio": environment["stress_ratio"],
            "rainfall_class": environment["rainfall_class"],
            "temperature": current.get("temperature"),
            "rainfall": current.get("rainfall"),
            "temperature_suitability": environment["groups"]["temperature"],
            "water_suitability": environment["groups"]["water"],
            "soil_suitability": environment["groups"]["soil"],
            "overall_suitability": environment["overall"],
            "suitability_score": environment["score"],

            # food web
            "food_resources": food_resources,
            "observed_food_web_species": observed_food_web,
            "consumers": consumers_pretty,
            "affected_species": sorted({*food_resources, *observed_food_web,
                                        *consumers_pretty}),
            "food_availability": diet["forage_status"],
            "missing_food_resources": [pretty_name(i) for i in diet["missing"]],

            # evidence and impact
            "evidence_score": evidence["score"],
            "evidence_level": evidence["level"],
            "evidence_reasons": evidence["reasons"],
            "impact_score": impact["score"],
            "impact_level": impact["level"],
            "impact_label": impact["label"],
            "confidence": impact["confidence"],
            "population_risk": impact["population_risk"],
            "base_population_risk": impact["base_population_risk"],
            "resource_risk": impact["resource_risk"],
            "park_effect_risk": impact["park_effect_risk"],
            "context_note": impact["context_note"],
            "stress_factors": impact["stress_factors"],

            # effects
            "park_effect": park_effect,
            "park_effect_reason": " ".join(park_reasons),

            # text - summary is the headline, details is the drill-down
            "summary": summary,
            "reason": summary,
            "interpretation": summary,
            "interpretation_details": details,
            "full_interpretation": summary + " " + " ".join(details),
            "downstream_effect": ctx["food_web_effect"],
            "location_mismatch": ctx["location_mismatch"],

            # references
            "environmental_observation_id": current["environmental_observation_id"],
            "observation_date": current["date"],
        })

    if not results:
        return pd.DataFrame()

    report = pd.DataFrame(results)
    report["_sort"] = pd.to_numeric(report["impact_score"],
                                    errors="coerce").fillna(-1)
    return (report
            .sort_values(by=["_sort", "species", "section"],
                         ascending=[False, True, True])
            .drop(columns=["_sort"])
            .reset_index(drop=True))


# ============================================================
# TREND MODEL (replaces the old biodiversity regression)
# ============================================================

def train_population_trend_model() -> pd.Series:
    """
    Fitted slope (individuals per year) for every species, park-wide.
    Negative means declining. Sorted worst first.

    This replaces train_species_impact_model, which regressed the Shannon
    index on the same counts used to compute it - an identity, not a result.
    """
    df = prepare_features(get_observation_data())
    if df.empty:
        return pd.Series(dtype=float)

    series = (df.groupby(["canonical_species", "survey_day"], as_index=False)
                .agg(population=("population", "sum")))

    slopes = {}
    for canon, group in series.groupby("canonical_species"):
        group = group.sort_values("survey_day")
        days = (group["survey_day"] - group["survey_day"].min()).dt.days.to_numpy()
        slope = _linear_slope(days, group["population"].to_numpy(dtype=float))
        if slope is not None:
            slopes[pretty_name(canon)] = round(slope, 3)

    return pd.Series(slopes, dtype=float).sort_values()


# ============================================================
# FORECAST
# ============================================================

def forecast_species_population(species_name: str, years_ahead: int = 3,
                                section: Optional[str] = None) -> Dict[str, Any]:
    try:
        years_ahead = int(years_ahead)
    except (TypeError, ValueError):
        return {"success": False, "error": "years_ahead must be a whole number."}

    if years_ahead <= 0:
        return {"success": False, "error": "years_ahead must be greater than zero."}
    if years_ahead > MAX_FORECAST_YEARS:
        return {"success": False,
                "error": f"years_ahead cannot exceed {MAX_FORECAST_YEARS}."}

    df = prepare_features(get_observation_data())
    if df.empty:
        return {"success": False, "error": "No approved observations available."}

    canon = canonical_species_name(species_name)
    df = df[df["canonical_species"] == canon].copy()
    if df.empty:
        return {"success": False,
                "error": f"No approved observations found for {species_name}."}

    sections = sorted(df["section_name"].unique())
    warnings = []

    if section is None:
        selected = "ALL" if len(sections) > 1 else sections[0]
    else:
        selected = section

    if str(selected).lower() == "all":
        selected = "ALL"
        if len(sections) > 1:
            warnings.append("Counts are summed across sections; the trend can be "
                            "distorted if surveys do not cover the same sections "
                            "each time.")
    else:
        df = df[df["section_name"].str.lower() == str(selected).lower()]
        if df.empty:
            return {"success": False,
                    "error": f"{species_name} has no observations in {selected}."}

    grouped = (df.groupby("survey_day", as_index=False)["population"]
                 .sum().sort_values("survey_day"))

    if len(grouped) < MIN_FORECAST_POINTS:
        return {"success": False,
                "error": f"At least {MIN_FORECAST_POINTS} survey dates are needed "
                         f"to forecast. Only {len(grouped)} are available."}

    first_day = grouped["survey_day"].min()
    last_day = grouped["survey_day"].max()
    days = (grouped["survey_day"] - first_day).dt.days.to_numpy()

    if np.ptp(days) == 0:
        return {"success": False,
                "error": "All observations fall on the same day, so no trend "
                         "can be fitted."}

    years = days / DAYS_PER_YEAR
    values = grouped["population"].to_numpy(dtype=float)
    slope, intercept = np.polyfit(years, values, 1)

    fitted = slope * years + intercept
    mae = float(np.mean(np.abs(values - fitted)))
    ss_res = float(np.sum((values - fitted) ** 2))
    ss_tot = float(np.sum((values - values.mean()) ** 2))
    r2 = None if ss_tot == 0 else round(1 - ss_res / ss_tot, 3)

    observed_years = float(years.max())
    if years_ahead > max(observed_years, 1):
        warnings.append(f"The {years_ahead}-year horizon is longer than the "
                        f"{observed_years:.1f} years of data. Treat it as "
                        f"illustrative only.")

    warnings.append("The forecast uses the time trend only. It does not predict "
                    "rainfall, vegetation, water or grazing pressure.")

    predictions = []
    for year_number in range(1, years_ahead + 1):
        future_day = last_day + pd.DateOffset(years=year_number)
        future_year = (future_day - first_day).days / DAYS_PER_YEAR
        predicted = max(0.0, float(slope * future_year + intercept))
        predictions.append({
            "year_ahead": year_number,
            "date": future_day.strftime("%Y-%m-%d"),
            "predicted_population": round(predicted, 2),
        })

    if slope > 0:
        direction = "increasing"
    elif slope < 0:
        direction = "declining"
    else:
        direction = "flat"

    return {
        "success": True,
        "species": species_name,
        "section": selected,
        "model": "Time-trend linear regression",
        "slope_per_year": round(float(slope), 3),
        "direction": direction,
        "historical_observations": len(grouped),
        "historical_start": first_day.strftime("%Y-%m-%d"),
        "historical_end": last_day.strftime("%Y-%m-%d"),
        "mae": round(mae, 3),
        "mae_in_sample": round(mae, 3),
        "r2": r2,
        "predictions": predictions,
        "warning": " ".join(warnings),
    }


# ============================================================
# COMPATIBILITY HELPERS
# ============================================================

def get_available_species() -> List[str]:
    df = get_observation_data()
    if df.empty:
        return []
    return sorted(df["species"].dropna().unique().tolist())


def get_available_sections() -> List[str]:
    df = get_observation_data()
    if df.empty:
        return []
    return sorted(df["section_name"].dropna().unique().tolist())


def get_risk_level_from_impact(impact_label: Optional[str]) -> str:
    return {
        "Low Concern": "Low",
        "Watch": "Moderate",
        "Moderate Concern": "Moderate",
        "High Concern": "High",
        "Critical Concern": "Critical",
        "Insufficient Evidence": "Insufficient Evidence",
    }.get(impact_label, "Unknown")


def build_analysis_records() -> List[Dict[str, Any]]:
    report = generate_species_effect_report()
    if report.empty:
        return []

    records = []
    for _, row in report.iterrows():
        previous = row.get("previous_population")
        if previous is None or pd.isna(previous):
            continue

        percentage = row.get("percentage_change")

        records.append({
            "species_id": int(row["species_id"]),
            "current_population": int(row["current_population"]),
            "previous_population": int(previous),
            "population_change": int(row["population_change"]),
            "percentage_change": (0.0 if pd.isna(percentage) else float(percentage)),
            "trend": row["trend"],
            "risk_level": get_risk_level_from_impact(row["impact_label"]),
            "evidence_level": row["evidence_level"],
            "section": row["section"],
        })

    return records


# ============================================================
# DEBUG OUTPUT
# ============================================================

def print_analysis_report(show_details: bool = False) -> None:
    report = generate_species_effect_report()

    if report.empty:
        print("No approved observations available for analysis.")
        return

    print()
    print("=" * 78)
    print("CBU NATURE PARK - ECOLOGICAL ANALYSIS")
    print("=" * 78)

    for _, row in report.iterrows():
        print()
        print(row["summary"])
        print(f"  trend={row['trend']}  impact={row['impact_label']}  "
              f"confidence={row['confidence']}  "
              f"pressure={row['resource_pressure']}")
        if show_details:
            for line in row["interpretation_details"]:
                print(f"    - {line}")

    print()
    print("-" * 78)
    print("PARK-WIDE POPULATION TRENDS (individuals per year)")
    print("-" * 78)
    trends = train_population_trend_model()
    if trends.empty:
        print("Not enough repeat surveys to fit trends.")
    else:
        for species, slope in trends.items():
            marker = "DOWN" if slope < 0 else ("UP" if slope > 0 else "FLAT")
            print(f"  {species:<35} {slope:+8.2f}  {marker}")


if __name__ == "__main__":
    print_analysis_report(show_details=True)
