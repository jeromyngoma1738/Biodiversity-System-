"""
AI Ecological Analysis Model
CBU Nature Park Biodiversity Monitoring System

Purpose
-------
Provides:
    1. Environment-aware ecological analysis
    2. Population-density analysis
    3. Species-specific food-web analysis
    4. Ecological interpretation
    5. Transparent ecological-risk classification
    6. Species population forecasting

Important
---------
This module does not claim ecological causation unless supported by
the available data. Environmental relationships are interpreted as
associations unless a validated causal model exists.
"""

from __future__ import annotations

from typing import Optional, Dict, List, Any

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

from models import Observation, EnvironmentalObservation


# ============================================================================
# CONFIGURATION
# ============================================================================

ENVIRONMENT_MATCH_DAYS = 7

MIN_FORECAST_OBSERVATIONS = 5
MIN_FORECAST_TIME_POINTS = 4

FORECAST_DAYS_PER_YEAR = 365.25

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


# ============================================================================
# TROPHIC ROLES
# ============================================================================

TROPHIC_ROLE = {

    # ------------------------------------------------------------------------
    # PRODUCERS
    # ------------------------------------------------------------------------
    "Tree": "producer",
    "Grass": "producer",
    "Flowering Plant": "producer",
    "Acacia": "producer",
    "Miombo Tree": "producer",
    "Mopane": "producer",
    "Grass Species": "producer",
    "Plant": "producer",
    "Brachystegia": "producer",
    "Julbernardia": "producer",
    "Isoberlinia": "producer",
    "Faidherbia Albida": "producer",
    "Khasi Pine": "producer",
    "Apple-ring Acacia": "producer",
    "Ana Tree": "producer",
    "Wild Loquate": "producer",
    "Wild Custard Apple": "producer",
    "Msasa": "producer",

    # ------------------------------------------------------------------------
    # HERBIVORES
    # ------------------------------------------------------------------------
    "Grasshopper": "herbivore",
    "Caterpillar": "herbivore",
    "Snail": "herbivore",
    "Rabbit": "herbivore",

    "Impala": "herbivore",
    "Waterbuck": "herbivore",
    "Puku": "herbivore",
    "Kudu": "herbivore",
    "Sable Antelope": "herbivore",
    "Bushbuck": "herbivore",
    "Duiker": "herbivore",
    "Warthog": "herbivore",
    "Zebra": "herbivore",
    "Buffalo": "herbivore",

    # ------------------------------------------------------------------------
    # OMNIVORES
    # ------------------------------------------------------------------------
    "Mouse": "omnivore",
    "Mice": "omnivore",
    "Rat": "omnivore",
    "Rats": "omnivore",
    "Rodent": "omnivore",

    "Crow": "omnivore",
    "African Pied Crow": "omnivore",
    "Dove": "omnivore",
    "Cape Turtle Dove": "omnivore",
    "Laughing Dove": "omnivore",
    "Pigeon": "omnivore",

    "African Grey Hornbill": "omnivore",
    "Southern Yellow-billed Hornbill": "omnivore",
    "Crested Barbet": "omnivore",
    "Civet": "omnivore",
    "Hadada Ibis": "omnivore",

    # ------------------------------------------------------------------------
    # SMALL PREDATORS / INSECTIVORES
    # ------------------------------------------------------------------------
    "Spider": "small_predator",
    "Praying Mantis": "small_predator",
    "Frog": "small_predator",
    "Frogs": "small_predator",
    "Lizard": "small_predator",

    "Insectivorous Bird": "small_predator",
    "African Hoopoe": "small_predator",
    "Fork-tailed Drongo": "small_predator",
    "Common Fiscal": "small_predator",
    "African Paradise Flycatcher": "small_predator",

    # ------------------------------------------------------------------------
    # PREDATORS
    # ------------------------------------------------------------------------
    "Snake": "predator",
    "Snakes": "predator",
    "Mongoose": "predator",
    "Cat": "predator",

    # ------------------------------------------------------------------------
    # LARGE PREDATORS
    #
    # These are classifications only.
    # The model will NOT assume that these species occur at CBU Nature Park.
    # They only participate if they have approved observations.
    # ------------------------------------------------------------------------
    "Monitor Lizard": "large_predator",
    "Eagle": "large_predator",
    "Crocodile": "large_predator",

    # ------------------------------------------------------------------------
    # DECOMPOSERS
    # ------------------------------------------------------------------------
    "Fungi": "decomposer",
    "Bacteria": "decomposer",
    "Termite": "decomposer",
    "Earthworm": "decomposer",
}


# ============================================================================
# SPECIES DIET
# ============================================================================
#
# Format:
#
#     consumer -> food resources / prey
#
# The model uses this dictionary for direct species-level food-web analysis.
# It does NOT create food-web relationships simply from trophic roles.
# ============================================================================

SPECIES_DIET = {

    # ------------------------------------------------------------------------
    # SMALL PREDATORS
    # ------------------------------------------------------------------------
    "Frog": [
        "Grasshopper",
        "Caterpillar",
        "Spider",
        "Snail",
    ],

    "Frogs": [
        "Grasshopper",
        "Caterpillar",
        "Spider",
        "Snail",
    ],

    "Lizard": [
        "Grasshopper",
        "Caterpillar",
        "Spider",
    ],

    "Spider": [
        "Grasshopper",
        "Caterpillar",
    ],

    "Praying Mantis": [
        "Grasshopper",
        "Caterpillar",
    ],

    "Insectivorous Bird": [
        "Grasshopper",
        "Caterpillar",
        "Spider",
    ],

    "African Hoopoe": [
        "Grasshopper",
        "Caterpillar",
    ],

    "Fork-tailed Drongo": [
        "Grasshopper",
        "Caterpillar",
        "Spider",
    ],

    "Common Fiscal": [
        "Grasshopper",
        "Caterpillar",
        "Lizard",
    ],

    "African Paradise Flycatcher": [
        "Grasshopper",
        "Caterpillar",
    ],

    # ------------------------------------------------------------------------
    # PREDATORS
    # ------------------------------------------------------------------------
    "Snake": [
        "Frog",
        "Mouse",
        "Rat",
        "Lizard",
        "Helmeted Guineafowl",
    ],

    "Snakes": [
        "Frog",
        "Mouse",
        "Rat",
        "Lizard",
        "Helmeted Guineafowl",
    ],

    "Mongoose": [
        "Snake",
        "Mouse",
        "Rat",
        "Frog",
        "Grasshopper",
    ],

    # ------------------------------------------------------------------------
    # LARGE PREDATORS
    # ------------------------------------------------------------------------
    "Monitor Lizard": [
        "Frog",
        "Snake",
        "Lizard",
        "Mouse",
        "Rat",
        "Helmeted Guineafowl",
        "Grasshopper",
    ],

    "Eagle": [
        "Snake",
        "Mouse",
        "Rat",
        "Lizard",
        "Dove",
    ],

    "Crocodile": [
        "Impala",
        "Waterbuck",
        "Puku",
    ],

    # ------------------------------------------------------------------------
    # HERBIVORES
    # ------------------------------------------------------------------------
    "Impala": [
        "Grass",
        "Grass Species",
        "Miombo Tree",
        "Acacia",
    ],

    "Zebra": [
        "Grass",
        "Grass Species",
    ],

    "Puku": [
        "Grass",
        "Plant",
    ],

    "Kudu": [
        "Acacia",
        "Miombo Tree",
        "Brachystegia",
    ],

    "Waterbuck": [
        "Grass",
        "Grass Species",
    ],

    "Buffalo": [
        "Grass",
        "Grass Species",
    ],

    "Sable Antelope": [
        "Grass",
        "Grass Species",
        "Miombo Tree",
    ],

    "Bushbuck": [
        "Acacia",
        "Miombo Tree",
    ],

    "Duiker": [
        "Grass",
        "Grass Species",
    ],

    "Warthog": [
        "Grass",
        "Grass Species",
        "Plant",
    ],

    "Rabbit": [
        "Grass",
        "Grass Species",
    ],

    "Snail": [
        "Grass",
        "Flowering Plant",
    ],

    "Grasshopper": [
        "Grass",
        "Grass Species",
    ],

    "Caterpillar": [
        "Flowering Plant",
        "Grass",
    ],

    "Helmeted Guineafowl": [
        "Grass Species",
    ],

    "Cape Turtle Dove": [
        "Grass Species",
    ],

    "Laughing Dove": [
        "Grass Species",
    ],

    "Dove": [
        "Grass Species",
    ],

    "Village Weaver": [
        "Grass Species",
    ],

    "Southern Red Bishop": [
        "Grass Species",
    ],

    "Speckled Mousebird": [
        "Flowering Plant",
    ],

    # ------------------------------------------------------------------------
    # OMNIVORES
    # ------------------------------------------------------------------------
    "Mouse": [
        "Grass",
        "Grass Species",
        "Grasshopper",
    ],

    "Mice": [
        "Grass",
        "Grass Species",
        "Grasshopper",
    ],

    "Rat": [
        "Grass",
        "Grass Species",
        "Grasshopper",
    ],

    "Rats": [
        "Grass",
        "Grass Species",
        "Grasshopper",
    ],

    "Rodent": [
        "Grass",
        "Grass Species",
        "Grasshopper",
    ],

    "African Pied Crow": [
        "Grasshopper",
        "Flowering Plant",
    ],

    "Hadada Ibis": [
        "Earthworm",
        "Grasshopper",
    ],

    "African Grey Hornbill": [
        "Grasshopper",
        "Flowering Plant",
    ],

    "Southern Yellow-billed Hornbill": [
        "Grasshopper",
        "Flowering Plant",
    ],

    "Crested Barbet": [
        "Grasshopper",
        "Flowering Plant",
    ],

    "Civet": [
        "Mouse",
        "Rat",
        "Grasshopper",
        "Flowering Plant",
    ],
}


# ============================================================================
# IMPACT LEVELS
# ============================================================================

IMPACT_LABELS = {
    0: "Stable Ecosystem",
    1: "Slight Disturbance",
    2: "Moderate Risk",
    3: "High Risk",
    4: "Critical",
}


# ============================================================================
# GENERAL HELPERS
# ============================================================================

def normalize_name(value: Any) -> str:
    """Return a normalized species/location string."""
    if value is None:
        return ""

    return " ".join(str(value).strip().lower().split())


def safe_number(value: Any, default: Optional[float] = None) -> Optional[float]:
    """Safely convert a value to float."""
    if value is None:
        return default

    try:
        number = float(value)

        if np.isnan(number):
            return default

        return number

    except (TypeError, ValueError):
        return default


def clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    """Clamp a numeric value to a specified range."""
    return float(np.clip(value, minimum, maximum))


def get_role(species_name: str) -> str:
    """Return the trophic role of a species."""
    normalized = normalize_name(species_name)

    for species, role in TROPHIC_ROLE.items():

        if normalize_name(species) == normalized:
            return role

    return "unknown"


def canonical_species_name(species_name: str) -> Optional[str]:
    """
    Find the canonical name used in the diet dictionary.

    Matching is case-insensitive.
    """
    normalized = normalize_name(species_name)

    for species in SPECIES_DIET:

        if normalize_name(species) == normalized:
            return species

    return None


# ============================================================================
# RAINFALL
# ============================================================================

def classify_rainfall(rainfall: Any) -> str:
    """Classify rainfall using the project's existing thresholds."""
    value = safe_number(rainfall)

    if value is None:
        return "Unknown"

    if value < 0:
        return "Invalid"

    if value < 10:
        return "Very Low"

    if value < 50:
        return "Low"

    if value < 100:
        return "Moderate"

    if value < 200:
        return "High"

    return "Very High"


# ============================================================================
# ENVIRONMENTAL DATA MATCHING
# ============================================================================

def get_environmental_observation(
    location: str,
    observation_date,
    max_days: int = ENVIRONMENT_MATCH_DAYS,
):
    """
    Find the nearest environmental observation for a wildlife observation.

    Only records within max_days are accepted.

    This prevents the model from associating a wildlife observation with
    environmental conditions recorded months away from the observation date.
    """

    if not location or observation_date is None:
        return None

    records = (
        EnvironmentalObservation.query
        .filter_by(location=location)
        .all()
    )

    if not records:
        return None

    valid_records = []

    for record in records:

        if record.observation_date is None:
            continue

        try:
            difference = abs(
                (record.observation_date - observation_date).total_seconds()
            )

        except (TypeError, AttributeError):
            continue

        valid_records.append((difference, record))

    if not valid_records:
        return None

    difference, closest = min(
        valid_records,
        key=lambda item: item[0],
    )

    max_seconds = max_days * 24 * 60 * 60

    if difference > max_seconds:
        return None

    return closest


# ============================================================================
# DATA EXTRACTION
# ============================================================================

def get_observation_data() -> pd.DataFrame:
    """
    Load approved biodiversity observations and attach environmental data.
    """

    observations = (
        Observation.query
        .filter_by(status="Approved")
        .all()
    )

    rows = []

    for observation in observations:

        if not observation.species:
            continue

        species = observation.species

        environment = get_environmental_observation(
            location=species.location,
            observation_date=observation.observation_date,
        )

        row = {
            "species": species.specie_Common_Name,
            "habitat": species.specie_Habitat,
            "location": species.location,
            "population": observation.population_count,
            "date": observation.observation_date,
        }

        for column in ENVIRONMENT_COLUMNS:
            row[column] = None

        if environment:

            for column in ENVIRONMENT_COLUMNS:
                row[column] = getattr(
                    environment,
                    column,
                    None,
                )

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================================
# FEATURE PREPARATION
# ============================================================================

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Clean observations and calculate derived variables."""

    if df is None or df.empty:
        return pd.DataFrame()

    result = df.copy()

    result["date"] = pd.to_datetime(
        result["date"],
        errors="coerce",
    )

    result["population"] = pd.to_numeric(
        result["population"],
        errors="coerce",
    )

    result = result.dropna(
        subset=[
            "species",
            "habitat",
            "location",
            "population",
            "date",
        ]
    )

    result["role"] = result["species"].apply(get_role)

    result["area_hectares"] = pd.to_numeric(
        result["area_hectares"],
        errors="coerce",
    )

    result["population_density"] = np.where(
        result["area_hectares"] > 0,
        result["population"] / result["area_hectares"],
        np.nan,
    )

    result["rainfall_class"] = result["rainfall"].apply(
        classify_rainfall
    )

    return result


# ============================================================================
# POPULATION DENSITY
# ============================================================================

def population_density(
    population: Any,
    area_hectares: Any,
) -> Optional[float]:
    """
    Calculate population density.

    Unit:
        individuals per hectare
    """

    population = safe_number(population)
    area = safe_number(area_hectares)

    if population is None or area is None:
        return None

    if area <= 0:
        return None

    return population / area


def calculate_density_change(
    current_density: Optional[float],
    baseline_density: Optional[float],
) -> Optional[float]:
    """Calculate percentage change in population density."""

    if (
        current_density is None
        or baseline_density is None
        or baseline_density <= 0
    ):
        return None

    return (
        (current_density - baseline_density)
        / baseline_density
    ) * 100


# ============================================================================
# ENVIRONMENTAL ASSESSMENT
# ============================================================================

def assess_environment(row: pd.Series) -> Dict[str, Any]:
    """
    Assess environmental conditions.

    Important:
    This function identifies environmental stress indicators using the
    project's recorded thresholds. It does not claim that these conditions
    caused a population change.
    """

    factors: List[str] = []
    score = 0

    rainfall = safe_number(row.get("rainfall"))
    temperature = safe_number(row.get("temperature"))

    soil_quality = safe_number(row.get("soil_quality"))
    soil_moisture = safe_number(row.get("soil_moisture"))

    water_quality = safe_number(row.get("water_quality"))
    water_turbidity = safe_number(row.get("water_turbidity"))

    vegetation_cover = safe_number(row.get("vegetation_cover"))
    vegetation_density = safe_number(row.get("vegetation_density"))

    grass = safe_number(row.get("grass_availability"))
    tree_density = safe_number(row.get("tree_density"))

    area = safe_number(row.get("area_hectares"))

    rainfall_class = classify_rainfall(rainfall)

    # ------------------------------------------------------------------------
    # RAINFALL
    # ------------------------------------------------------------------------

    if rainfall_class == "Very Low":
        score += 2
        factors.append("very low rainfall")

    elif rainfall_class == "Low":
        score += 1
        factors.append("low rainfall")

    elif rainfall_class == "Very High":
        score += 1
        factors.append("very high rainfall")

    # ------------------------------------------------------------------------
    # TEMPERATURE
    # ------------------------------------------------------------------------

    if temperature is not None:

        if temperature >= 35:
            score += 2
            factors.append("high recorded temperature")

        elif temperature >= 30:
            score += 1
            factors.append("elevated recorded temperature")

    # ------------------------------------------------------------------------
    # SOIL
    # ------------------------------------------------------------------------

    if soil_quality is not None:

        if soil_quality < 30:
            score += 2
            factors.append("poor soil quality")

        elif soil_quality < 60:
            score += 1
            factors.append("moderate soil quality")

    if soil_moisture is not None:

        if soil_moisture < 20:
            score += 1
            factors.append("low soil moisture")

    # ------------------------------------------------------------------------
    # WATER
    # ------------------------------------------------------------------------

    if water_quality is not None:

        if water_quality < 30:
            score += 2
            factors.append("poor water quality")

        elif water_quality < 60:
            score += 1
            factors.append("moderate water quality")

    if water_turbidity is not None:

        if water_turbidity > 70:
            score += 2
            factors.append("high water turbidity")

        elif water_turbidity > 40:
            score += 1
            factors.append("elevated water turbidity")

    # ------------------------------------------------------------------------
    # VEGETATION
    # ------------------------------------------------------------------------

    if vegetation_cover is not None:

        if vegetation_cover < 20:
            score += 2
            factors.append("low vegetation cover")

        elif vegetation_cover < 40:
            score += 1
            factors.append("moderate vegetation cover")

    if vegetation_density is not None:

        if vegetation_density < 20:
            score += 1
            factors.append("low vegetation density")

    # ------------------------------------------------------------------------
    # GRASS
    # ------------------------------------------------------------------------

    if grass is not None:

        if grass < 20:
            score += 2
            factors.append("low grass availability")

        elif grass < 40:
            score += 1
            factors.append("moderate grass availability")

    # ------------------------------------------------------------------------
    # TREE DENSITY
    # ------------------------------------------------------------------------

    if tree_density is not None:

        if tree_density < 20:
            score += 1
            factors.append("low tree density")

    # ------------------------------------------------------------------------
    # AREA
    # ------------------------------------------------------------------------

    if area is not None:

        if area <= 0:
            factors.append("invalid monitoring area")

        else:
            factors.append(
                f"{area:.2f} hectares monitored"
            )

    # ------------------------------------------------------------------------
    # CONDITION
    # ------------------------------------------------------------------------

    if score >= 7:
        condition = "Highly stressed environmental conditions"

    elif score >= 4:
        condition = "Moderately stressed environmental conditions"

    elif score >= 1:
        condition = "Some environmental stress indicators detected"

    else:
        condition = "No major environmental stress indicators detected"

    return {
        "score": score,
        "condition": condition,
        "rainfall_class": rainfall_class,
        "factors": factors,
        "has_environmental_data": any(
            safe_number(row.get(column)) is not None
            for column in ENVIRONMENT_COLUMNS
        ),
    }


# ============================================================================
# SPECIES-SPECIFIC RESOURCE ANALYSIS
# ============================================================================

def get_recorded_food_resources(
    species_name: str,
    observed_species: set,
) -> List[str]:
    """
    Return food resources/prey actually observed in the dataset.
    """

    canonical = canonical_species_name(species_name)

    if canonical is None:
        return []

    normalized_observed = {
        normalize_name(species)
        for species in observed_species
    }

    resources = []

    for food in SPECIES_DIET.get(canonical, []):

        if normalize_name(food) in normalized_observed:
            resources.append(food)

    return sorted(set(resources))


def get_consumers(
    species_name: str,
    observed_species: set,
) -> List[str]:
    """
    Find observed species that directly consume the selected species.
    """

    normalized_target = normalize_name(species_name)

    normalized_observed = {
        normalize_name(species)
        for species in observed_species
    }

    consumers = []

    for consumer, food_list in SPECIES_DIET.items():

        if normalize_name(consumer) not in normalized_observed:
            continue

        if any(
            normalize_name(food) == normalized_target
            for food in food_list
        ):
            consumers.append(consumer)

    return sorted(set(consumers))


def get_food_web_relationships(
    species_name: str,
    df: pd.DataFrame,
) -> Dict[str, List[str]]:
    """
    Return direct observed food-web relationships.

    Returns:
        resources:
            Species/resources eaten by the target species.

        consumers:
            Observed species that consume the target species.
    """

    if df is None or df.empty:
        return {
            "resources": [],
            "consumers": [],
        }

    observed_species = set(
        df["species"]
        .dropna()
        .astype(str)
        .unique()
    )

    return {
        "resources": get_recorded_food_resources(
            species_name,
            observed_species,
        ),
        "consumers": get_consumers(
            species_name,
            observed_species,
        ),
    }


def get_affected_species(
    species_name: str,
    df: pd.DataFrame,
) -> List[str]:
    """
    Backwards-compatible helper.

    Returns all direct observed food-web connections, without introducing
    generic trophic-role relationships.
    """

    relationships = get_food_web_relationships(
        species_name,
        df,
    )

    return sorted(
        set(
            relationships["resources"]
            + relationships["consumers"]
        )
    )


# ============================================================================
# RESOURCE PRESSURE
# ============================================================================

def calculate_resource_pressure(
    species: str,
    role: str,
    df: pd.DataFrame,
) -> str:
    """
    Estimate resource pressure using resources relevant to the species.

    The function deliberately avoids treating every environmental variable
    as equally relevant to every species.
    """

    if df is None or df.empty:
        return "Unknown"

    species_df = df[
        df["species"].apply(normalize_name)
        == normalize_name(species)
    ]

    if species_df.empty:
        return "Unknown"

    row = species_df.sort_values("date").iloc[-1]

    resources = []

    if role == "herbivore":

        food = SPECIES_DIET.get(
            canonical_species_name(species) or species,
            [],
        )

        normalized_food = {
            normalize_name(item)
            for item in food
        }

        if (
            "grass" in normalized_food
            or "grass species" in normalized_food
        ):
            value = safe_number(
                row.get("grass_availability")
            )

            if value is not None:
                resources.append(value)

        if "miombo tree" in normalized_food:
            value = safe_number(
                row.get("tree_density")
            )

            if value is not None:
                resources.append(value)

        if "acacia" in normalized_food:
            value = safe_number(
                row.get("tree_density")
            )

            if value is not None:
                resources.append(value)

        if "plant" in normalized_food:
            value = safe_number(
                row.get("vegetation_cover")
            )

            if value is not None:
                resources.append(value)

    elif role == "producer":

        for column in [
            "vegetation_cover",
            "vegetation_density",
            "soil_quality",
            "soil_moisture",
        ]:
            value = safe_number(row.get(column))

            if value is not None:
                resources.append(value)

    else:

        for column in [
            "vegetation_cover",
            "vegetation_density",
            "grass_availability",
            "tree_density",
            "water_quality",
        ]:
            value = safe_number(row.get(column))

            if value is not None:
                resources.append(value)

    if not resources:
        return "Unknown"

    resource_score = float(np.mean(resources))

    if resource_score < 20:
        return "Very High"

    if resource_score < 40:
        return "High"

    if resource_score < 60:
        return "Moderate"

    return "Low"


# ============================================================================
# ECOLOGICAL INTERPRETATION
# ============================================================================

def ecological_interpretation(
    species: str,
    role: str,
    pct_change: float,
    environment: Dict[str, Any],
    resource_pressure: str,
    food_web: Optional[Dict[str, List[str]]] = None,
) -> str:
    """
    Generate a cautious species-specific ecological interpretation.
    """

    condition = environment["condition"]
    factors = environment["factors"]

    if factors:
        factor_text = (
            " Recorded environmental indicators include "
            + ", ".join(factors)
            + "."
        )
    else:
        factor_text = (
            " No environmental measurements were available "
            "for this observation."
        )

    resources = (
        food_web["resources"]
        if food_web
        else []
    )

    consumers = (
        food_web["consumers"]
        if food_web
        else []
    )

    # ------------------------------------------------------------------------
    # PRODUCERS
    # ------------------------------------------------------------------------

    if role == "producer":

        if pct_change <= -50:

            return (
                f"{species} shows a severe population decline. "
                f"{condition}.{factor_text} "
                f"The decline may reduce plant resources or habitat "
                f"available to observed dependent species."
            )

        if pct_change < -20:

            return (
                f"{species} shows a noticeable population decline. "
                f"{condition}.{factor_text} "
                f"Reduced abundance may affect species that directly "
                f"depend on this plant resource."
            )

        if pct_change > 50:

            return (
                f"{species} shows substantial population growth. "
                f"{condition}.{factor_text} "
                f"The increase may increase the availability of plant "
                f"resources or habitat for dependent species."
            )

        return (
            f"{species} is relatively stable. "
            f"{condition}.{factor_text}"
        )

    # ------------------------------------------------------------------------
    # HERBIVORES
    # ------------------------------------------------------------------------

    if role == "herbivore":

        if pct_change <= -50:

            resource_text = ""

            if resource_pressure in {"High", "Very High"}:
                resource_text = (
                    f" The recorded resource conditions indicate "
                    f"{resource_pressure.lower()} resource pressure."
                )

            return (
                f"{species} shows a severe population decline. "
                f"{condition}.{resource_text}{factor_text} "
                f"The available data may indicate an association between "
                f"population change and resource or habitat conditions."
            )

        if pct_change >= 100:

            return (
                f"{species} shows substantial population growth. "
                f"{condition}.{factor_text} "
                f"Continued growth should be monitored because increasing "
                f"population density may increase demand for its recorded "
                f"food resources."
            )

        if abs(pct_change) < 20:

            return (
                f"{species} is relatively stable. "
                f"{condition}.{factor_text}"
            )

        return (
            f"{species} shows a moderate population change. "
            f"{condition}.{factor_text}"
        )

    # ------------------------------------------------------------------------
    # SMALL PREDATORS
    # ------------------------------------------------------------------------

    if role == "small_predator":

        if pct_change < -30:

            if resources:
                return (
                    f"{species} is declining. "
                    f"{condition}.{factor_text} "
                    f"Its decline may reduce feeding pressure on its "
                    f"observed food resources: {', '.join(resources)}."
                )

            return (
                f"{species} is declining. "
                f"{condition}.{factor_text}"
            )

        if pct_change > 50:

            if resources:
                return (
                    f"{species} is increasing. "
                    f"{condition}.{factor_text} "
                    f"If sustained, this may increase feeding pressure on "
                    f"its observed food resources: {', '.join(resources)}."
                )

            return (
                f"{species} is increasing. "
                f"{condition}.{factor_text}"
            )

        return (
            f"{species} shows a moderate population change. "
            f"{condition}.{factor_text}"
        )

    # ------------------------------------------------------------------------
    # PREDATORS
    # ------------------------------------------------------------------------

    if role in {
        "predator",
        "large_predator",
    }:

        if pct_change < -30:

            if resources:
                return (
                    f"{species} shows a population decline. "
                    f"{condition}.{factor_text} "
                    f"This may reduce feeding pressure on its observed "
                    f"prey: {', '.join(resources)}."
                )

            return (
                f"{species} shows a population decline. "
                f"{condition}.{factor_text}"
            )

        if pct_change > 50:

            if resources:
                return (
                    f"{species} shows a population increase. "
                    f"{condition}.{factor_text} "
                    f"If sustained, this may increase feeding pressure on "
                    f"its observed prey: {', '.join(resources)}."
                )

            return (
                f"{species} shows a population increase. "
                f"{condition}.{factor_text}"
            )

        return (
            f"{species} shows a moderate population change. "
            f"{condition}.{factor_text}"
        )

    # ------------------------------------------------------------------------
    # OMNIVORES
    # ------------------------------------------------------------------------

    if role == "omnivore":

        if resources:

            return (
                f"{species} shows a {pct_change:.1f}% population change. "
                f"{condition}.{factor_text} "
                f"The species has recorded food resources including "
                f"{', '.join(resources)}. Changes in these resources may "
                f"be relevant to future population trends."
            )

        return (
            f"{species} shows a {pct_change:.1f}% population change. "
            f"{condition}.{factor_text}"
        )

    # ------------------------------------------------------------------------
    # DECOMPOSERS
    # ------------------------------------------------------------------------

    if role == "decomposer":

        return (
            f"{species} shows a {pct_change:.1f}% population change. "
            f"{condition}.{factor_text} "
            f"Changes in decomposer abundance may be relevant to nutrient "
            f"cycling and soil processes."
        )

    return (
        f"{species} shows a {pct_change:.1f}% population change. "
        f"{condition}.{factor_text}"
    )


# ============================================================================
# IMPACT CLASSIFICATION
# ============================================================================

def classify_impact(
    pct_change: float,
    environment_score: float,
    resource_pressure: str,
    density_change: Optional[float] = None,
    food_web_connections: int = 0,
) -> int:
    """
    Calculate a transparent project-defined ecological risk index.

    This is NOT an externally validated conservation status.

    Components:
        50% population trend
        25% environmental stress
        15% resource pressure
        10% food-web connectivity
    """

    # ------------------------------------------------------------------------
    # Population risk
    # ------------------------------------------------------------------------

    if pct_change <= -80:
        population_risk = 4

    elif pct_change <= -50:
        population_risk = 3

    elif pct_change <= -25:
        population_risk = 2

    elif pct_change < -20:
        population_risk = 1

    elif pct_change >= 150:
        population_risk = 3

    elif pct_change >= 100:
        population_risk = 2

    elif abs(pct_change) < 20:
        population_risk = 0

    else:
        population_risk = 1

    # ------------------------------------------------------------------------
    # Environmental risk
    # ------------------------------------------------------------------------

    if environment_score >= 7:
        environment_risk = 4

    elif environment_score >= 4:
        environment_risk = 3

    elif environment_score >= 1:
        environment_risk = 1

    else:
        environment_risk = 0

    # ------------------------------------------------------------------------
    # Resource risk
    # ------------------------------------------------------------------------

    resource_risk = {
        "Very High": 4,
        "High": 3,
        "Moderate": 2,
        "Low": 0,
        "Unknown": 0,
    }.get(
        resource_pressure,
        0,
    )

    # ------------------------------------------------------------------------
    # Density component
    # ------------------------------------------------------------------------

    density_risk = 0

    if density_change is not None:

        if density_change <= -50:
            density_risk = 3

        elif density_change <= -25:
            density_risk = 2

        elif density_change < -10:
            density_risk = 1

    # ------------------------------------------------------------------------
    # Food-web connectivity
    # ------------------------------------------------------------------------

    food_web_risk = 0

    if food_web_connections >= 5:
        food_web_risk = 2

    elif food_web_connections >= 2:
        food_web_risk = 1

    # ------------------------------------------------------------------------
    # Final weighted score
    # ------------------------------------------------------------------------

    score = (
        0.50 * population_risk
        + 0.25 * environment_risk
        + 0.15 * resource_risk
        + 0.05 * density_risk
        + 0.05 * food_web_risk
    )

    if score >= 3.5:
        return 4

    if score >= 2.5:
        return 3

    if score >= 1.5:
        return 2

    if score >= 0.5:
        return 1

    return 0


# ============================================================================
# FOOD-WEB EFFECT
# ============================================================================

def generate_food_web_effect(
    species: str,
    pct_change: float,
    df: pd.DataFrame,
    environment: Dict[str, Any],
    resource_pressure: str,
) -> str:
    """
    Generate food-web interpretation using only direct observed relationships.
    """

    relationships = get_food_web_relationships(
        species,
        df,
    )

    resources = relationships["resources"]
    consumers = relationships["consumers"]

    condition = environment["condition"]

    parts = []

    # ------------------------------------------------------------------------
    # RESOURCE EFFECTS
    # ------------------------------------------------------------------------

    if resources:

        resource_names = ", ".join(resources)

        if pct_change < 0:

            parts.append(
                f"A decline in {species} may reduce feeding pressure "
                f"on its recorded food resources: {resource_names}."
            )

        elif pct_change > 0:

            parts.append(
                f"An increase in {species} may increase feeding pressure "
                f"on its recorded food resources: {resource_names}."
            )

    # ------------------------------------------------------------------------
    # CONSUMER EFFECTS
    # ------------------------------------------------------------------------

    if consumers:

        consumer_names = ", ".join(consumers)

        if pct_change < 0:

            parts.append(
                f"The decline may also reduce food availability for "
                f"observed consumers of {species}: {consumer_names}."
            )

        elif pct_change > 0:

            parts.append(
                f"The increase may provide greater food availability for "
                f"observed consumers of {species}: {consumer_names}."
            )

    # ------------------------------------------------------------------------
    # NO DIRECT RELATIONSHIP
    # ------------------------------------------------------------------------

    if not parts:

        return (
            f"No direct food-web relationship with another observed species "
            f"was identified for {species}. {condition}"
        )

    return (
        " ".join(parts)
        + f" {condition}."
    )


# ============================================================================
# FORECASTING
# ============================================================================

def forecast_species_population(
    species_name: str,
    years_ahead: int = 5,
    future_environment: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Forecast the population of a selected species for any positive number
    of future years.

    Parameters
    ----------
    species_name:
        Species to forecast.

    years_ahead:
        Number of years into the future.

    future_environment:
        Optional future environmental data.

        If supplied, it can be used by an environment-aware model in a
        future extension. Without future environmental measurements, this
        function deliberately uses historical time trend only.

    Returns
    -------
    Dictionary containing:
        success
        species
        model_type
        historical_observations
        historical_time_points
        mae
        r2
        forecast
        warning
    """

    # ------------------------------------------------------------------------
    # Validate years
    # ------------------------------------------------------------------------

    if not isinstance(years_ahead, int):

        raise ValueError(
            "years_ahead must be an integer."
        )

    if years_ahead <= 0:

        raise ValueError(
            "years_ahead must be greater than zero."
        )

    # ------------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------------

    df = prepare_features(
        get_observation_data()
    )

    if df.empty:

        return {
            "success": False,
            "message": (
                "No approved biodiversity observations are available."
            ),
        }

    # ------------------------------------------------------------------------
    # Find species
    # ------------------------------------------------------------------------

    species_df = df[
        df["species"].apply(normalize_name)
        == normalize_name(species_name)
    ].copy()

    if species_df.empty:

        return {
            "success": False,
            "message": (
                f"No approved observations were found for "
                f"{species_name}."
            ),
        }

    # ------------------------------------------------------------------------
    # Aggregate duplicate observations on the same date
    # ------------------------------------------------------------------------

    historical = (
        species_df
        .groupby("date", as_index=False)["population"]
        .sum()
        .sort_values("date")
    )

    observation_count = len(historical)

    if observation_count < MIN_FORECAST_OBSERVATIONS:

        return {
            "success": False,
            "message": (
                f"At least {MIN_FORECAST_OBSERVATIONS} distinct historical "
                f"observation dates are recommended for forecasting "
                f"{species_name}. Only {observation_count} are available."
            ),
        }

    # ------------------------------------------------------------------------
    # Require multiple time points
    # ------------------------------------------------------------------------

    historical["year"] = historical["date"].dt.year

    unique_years = historical["year"].nunique()

    if unique_years < MIN_FORECAST_TIME_POINTS:

        return {
            "success": False,
            "message": (
                f"At least {MIN_FORECAST_TIME_POINTS} different years of "
                f"observations are recommended. Only {unique_years} years "
                f"are available for {species_name}."
            ),
        }

    # ------------------------------------------------------------------------
    # Time feature
    # ------------------------------------------------------------------------

    first_date = historical["date"].min()

    historical["years_since_start"] = (
        (
            historical["date"] - first_date
        ).dt.total_seconds()
        / (
            FORECAST_DAYS_PER_YEAR
            * 24
            * 60
            * 60
        )
    )

    X = historical[
        ["years_since_start"]
    ]

    y = historical["population"]

    # ------------------------------------------------------------------------
    # Train model
    # ------------------------------------------------------------------------

    model = LinearRegression()

    model.fit(
        X,
        y,
    )

    # ------------------------------------------------------------------------
    # Historical model performance
    # ------------------------------------------------------------------------

    historical_predictions = model.predict(X)

    mae = mean_absolute_error(
        y,
        historical_predictions,
    )

    r2 = r2_score(
        y,
        historical_predictions,
    )

    # ------------------------------------------------------------------------
    # Future dates
    # ------------------------------------------------------------------------

    last_date = historical["date"].max()

    future_rows = []

    for year_offset in range(
        1,
        years_ahead + 1,
    ):

        future_date = (
            last_date
            + pd.DateOffset(
                years=year_offset
            )
        )

        elapsed_years = (
            (
                future_date - first_date
            ).total_seconds()
            / (
                FORECAST_DAYS_PER_YEAR
                * 24
                * 60
                * 60
            )
        )

        predicted = model.predict(
            pd.DataFrame(
                {
                    "years_since_start": [
                        elapsed_years
                    ]
                }
            )
        )[0]

        # Population cannot be negative.
        predicted = max(
            0.0,
            float(predicted),
        )

        future_rows.append(
            {
                "species": species_name,
                "forecast_year": future_date.year,
                "forecast_date": future_date,
                "predicted_population": round(
                    predicted,
                    2,
                ),
            }
        )

    # ------------------------------------------------------------------------
    # Forecast warning
    # ------------------------------------------------------------------------

    if future_environment is None:

        warning = (
            "Forecast uses historical population trend only. "
            "Future environmental conditions were not supplied."
        )

        model_type = "Time-trend Linear Regression"

    else:

        warning = (
            "Future environmental data were supplied, but this version "
            "uses the time-trend model. Environmental forecasting should "
            "only be enabled after sufficient historical training data "
            "are available."
        )

        model_type = "Time-trend Linear Regression"

    return {
        "success": True,
        "species": species_name,
        "model_type": model_type,
        "historical_observations": observation_count,
        "historical_time_points": unique_years,
        "mae": round(
            float(mae),
            3,
        ),
        "r2": round(
            float(r2),
            3,
        ),
        "forecast": future_rows,
        "warning": warning,
    }


# ============================================================================
# SPECIES EFFECT REPORT
# ============================================================================

def generate_species_effect_report() -> pd.DataFrame:
    """
    Generate the complete ecological analysis report.
    """

    df = prepare_features(
        get_observation_data()
    )

    if df.empty:
        return pd.DataFrame()

    report = []

    # ------------------------------------------------------------------------
    # Baseline population
    # ------------------------------------------------------------------------

    baselines = (
        df.groupby("species")["population"]
        .mean()
    )

    # ------------------------------------------------------------------------
    # Latest observation for every species
    # ------------------------------------------------------------------------

    latest_rows = (
        df.sort_values("date")
        .groupby("species")
        .tail(1)
    )

    latest = latest_rows.set_index(
        "species"
    )

    # ------------------------------------------------------------------------
    # Analyze every observed species
    # ------------------------------------------------------------------------

    for species in baselines.index:

        if species not in latest.index:
            continue

        current_row = latest.loc[
            species
        ]

        current = safe_number(
            current_row["population"],
            0,
        )

        baseline = safe_number(
            baselines[species],
            0,
        )

        # ------------------------------------------------------------
        # Population change
        # ------------------------------------------------------------

        if baseline > 0:

            pct_change = (
                (current - baseline)
                / baseline
            ) * 100

        else:

            pct_change = 0.0

        # ------------------------------------------------------------
        # Density
        # ------------------------------------------------------------

        current_density = population_density(
            current,
            current_row.get("area_hectares"),
        )

        baseline_density_values = df[
            df["species"]
            .apply(normalize_name)
            == normalize_name(species)
        ]["population_density"].dropna()

        baseline_density = (
            float(
                baseline_density_values.mean()
            )
            if not baseline_density_values.empty
            else None
        )

        density_change = calculate_density_change(
            current_density,
            baseline_density,
        )

        # ------------------------------------------------------------
        # Role
        # ------------------------------------------------------------

        role = get_role(species)

        # ------------------------------------------------------------
        # Environment
        # ------------------------------------------------------------

        environment = assess_environment(
            current_row
        )

        # ------------------------------------------------------------
        # Food web
        # ------------------------------------------------------------

        food_web = get_food_web_relationships(
            species,
            df,
        )

        food_web_connections = (
            len(food_web["resources"])
            + len(food_web["consumers"])
        )

        # ------------------------------------------------------------
        # Resource pressure
        # ------------------------------------------------------------

        resource_pressure = calculate_resource_pressure(
            species=species,
            role=role,
            df=df,
        )

        # ------------------------------------------------------------
        # Ecological impact
        # ------------------------------------------------------------

        impact_level = classify_impact(
            pct_change=pct_change,
            environment_score=environment["score"],
            resource_pressure=resource_pressure,
            density_change=density_change,
            food_web_connections=food_web_connections,
        )

        # ------------------------------------------------------------
        # Interpretation
        # ------------------------------------------------------------

        reason = ecological_interpretation(
            species=species,
            role=role,
            pct_change=pct_change,
            environment=environment,
            resource_pressure=resource_pressure,
            food_web=food_web,
        )

        # ------------------------------------------------------------
        # Food-web effect
        # ------------------------------------------------------------

        downstream = generate_food_web_effect(
            species=species,
            pct_change=pct_change,
            df=df,
            environment=environment,
            resource_pressure=resource_pressure,
        )

        # ------------------------------------------------------------
        # Area
        # ------------------------------------------------------------

        area = safe_number(
            current_row.get("area_hectares")
        )

        # ------------------------------------------------------------
        # Result
        # ------------------------------------------------------------

        report.append(
            {
                "species": species,
                "role": role,

                "current_population": current,

                "baseline_population": round(
                    baseline,
                    1,
                ),

                "percent_change": round(
                    pct_change,
                    1,
                ),

                "area_hectares": (
                    round(area, 2)
                    if area is not None
                    else None
                ),

                "population_density": (
                    round(
                        current_density,
                        4,
                    )
                    if current_density is not None
                    else None
                ),

                "baseline_density": (
                    round(
                        baseline_density,
                        4,
                    )
                    if baseline_density is not None
                    else None
                ),

                "density_change_percent": (
                    round(
                        density_change,
                        1,
                    )
                    if density_change is not None
                    else None
                ),

                # Environment
                "temperature": safe_number(
                    current_row.get("temperature")
                ),

                "rainfall": safe_number(
                    current_row.get("rainfall")
                ),

                "rainfall_class": environment[
                    "rainfall_class"
                ],

                "soil_ph": safe_number(
                    current_row.get("soil_ph")
                ),

                "soil_moisture": safe_number(
                    current_row.get("soil_moisture")
                ),

                "soil_quality": safe_number(
                    current_row.get("soil_quality")
                ),

                "water_ph": safe_number(
                    current_row.get("water_ph")
                ),

                "water_turbidity": safe_number(
                    current_row.get("water_turbidity")
                ),

                "water_quality": safe_number(
                    current_row.get("water_quality")
                ),

                "vegetation_cover": safe_number(
                    current_row.get("vegetation_cover")
                ),

                "vegetation_density": safe_number(
                    current_row.get("vegetation_density")
                ),

                "grass_availability": safe_number(
                    current_row.get("grass_availability")
                ),

                "tree_density": safe_number(
                    current_row.get("tree_density")
                ),

                "environmental_condition": (
                    environment["condition"]
                ),

                "environmental_factors": (
                    environment["factors"]
                ),

                "resource_pressure": resource_pressure,

                # Food web
                "food_resources": (
                    food_web["resources"]
                ),

                "consumers": (
                    food_web["consumers"]
                ),

                "affected_species": sorted(
                    set(
                        food_web["resources"]
                        + food_web["consumers"]
                    )
                ),

                # Impact
                "impact_level": impact_level,

                "impact_label": IMPACT_LABELS[
                    impact_level
                ],

                "reason": reason,

                "downstream_effect": downstream,
            }
        )

    if not report:
        return pd.DataFrame()

    return (
        pd.DataFrame(report)
        .sort_values(
            "impact_level",
            ascending=False,
        )
        .reset_index(drop=True)
    )


# ============================================================================
# AVAILABLE SPECIES
# ============================================================================

def get_available_species() -> List[str]:
    """
    Return species that currently have approved observations.
    """

    df = prepare_features(
        get_observation_data()
    )

    if df.empty:
        return []

    return sorted(
        df["species"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )


# ============================================================================
# TEST / COMMAND LINE
# ============================================================================

if __name__ == "__main__":

    report = generate_species_effect_report()

    if report.empty:
        print("No approved observations available for analysis.")

    else:
        print(report.to_string( index=False))
        print("\nAvailable species:")
        for species in get_available_species():
            print(
                f" - {species}"
            )