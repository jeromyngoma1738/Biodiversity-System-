"""
CBU NATURE PARK
SECTION-CENTRED ECOSYSTEM ANALYSIS ENGINE

The system analyses:

    Monitoring Site
        |
        +-- CURRENT Environmental Survey
        |       +-- Water
        |       +-- Temperature
        |       +-- Rainfall
        |       +-- Vegetation
        |       +-- Soil
        |
        +-- PREVIOUS Environmental Survey
        |       +-- Used for comparison only
        |
        +-- Biodiversity
        |       +-- Species richness
        |       +-- Population
        |       +-- Population density
        |       +-- Shannon diversity
        |
        +-- Resource pressure
        |
        +-- Land-section ecological condition


IMPORTANT DESIGN:

The CURRENT environmental survey is the survey selected
by the user.

The PREVIOUS environmental survey is NOT displayed as
another current survey.

Species observations are historical records. A new survey does
not overwrite old species observations. If a species is not
recorded in a new survey, its last known record remains available
as historical/carry-forward data, while current biodiversity
statistics use only species actually observed in the current survey.

It is used internally to calculate changes such as:

    Current temperature - Previous temperature
    Current rainfall - Previous rainfall
    Current vegetation - Previous vegetation
"""


import math

import numpy as np
import pandas as pd

from models import (
    Observation,
    EnvironmentalObservation,
    MonitoringSite,
    Species
)


# ============================================================
# GENERAL HELPERS
# ============================================================

def safe_number(value):
    """
    Safely convert a value to a normal Python float.

    Returns None when the value is missing or invalid.
    """

    if value is None:
        return None

    try:

        value = float(value)

        if not math.isfinite(value):
            return None

        return value

    except (TypeError, ValueError):

        return None


def section_key(site_id):
    """
    Create a unique identifier for a monitoring section.
    """

    return f"site:{site_id}"


def clean_value(value):
    """
    Convert NumPy values into normal Python values
    so Flask jsonify can serialize them.
    """

    if value is None:
        return None

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, np.bool_):
        return bool(value)

    return value


def safe_iso_date(value):
    """
    Convert a date/datetime value into an ISO string.
    """

    if value is None:
        return None

    try:
        return value.isoformat()

    except AttributeError:
        return str(value)


# ============================================================
# WATER ANALYSIS
# ============================================================

def classify_water_for_wildlife(row):

    quality = safe_number(
        row.get("water_quality")
    )

    ph = safe_number(
        row.get("water_ph")
    )

    turbidity = safe_number(
        row.get("water_turbidity")
    )

    problems = []

    score = 100

    # --------------------------------------------------------
    # WATER QUALITY
    # --------------------------------------------------------

    if quality is not None:

        if quality < 30:

            score -= 60

            problems.append(
                "Very poor water quality."
            )

        elif quality < 60:

            score -= 30

            problems.append(
                "Reduced water quality."
            )

    # --------------------------------------------------------
    # PH
    # --------------------------------------------------------

    if ph is not None:

        if ph < 6.0 or ph > 9.0:

            score -= 40

            problems.append(
                "Water pH is outside the broad acceptable screening range."
            )

        elif ph < 6.5 or ph > 8.5:

            score -= 15

            problems.append(
                "Water pH is marginal."
            )

    # --------------------------------------------------------
    # TURBIDITY
    # --------------------------------------------------------

    if turbidity is not None:

        if turbidity > 70:

            score -= 40

            problems.append(
                "Very high water turbidity."
            )

        elif turbidity > 40:

            score -= 15

            problems.append(
                "Elevated water turbidity."
            )

    # --------------------------------------------------------
    # NO DATA
    # --------------------------------------------------------

    if (
        quality is None
        and ph is None
        and turbidity is None
    ):

        return {

            "status": "Unknown",

            "score": None,

            "problems": [
                "No water-quality measurements recorded."
            ]
        }

    score = max(
        0,
        min(100, score)
    )

    if score >= 75:

        status = "Suitable"

    elif score >= 50:

        status = "Marginal"

    else:

        status = "Unsuitable"

    return {

        "status": status,

        "score": score,

        "problems": problems
    }


def classify_water_for_vegetation(row):

    quality = safe_number(
        row.get("water_quality")
    )

    ph = safe_number(
        row.get("water_ph")
    )

    turbidity = safe_number(
        row.get("water_turbidity")
    )

    problems = []

    score = 100

    # --------------------------------------------------------
    # WATER QUALITY
    # --------------------------------------------------------

    if quality is not None:

        if quality < 30:

            score -= 50

            problems.append(
                "Poor water quality."
            )

        elif quality < 60:

            score -= 25

            problems.append(
                "Reduced water quality."
            )

    # --------------------------------------------------------
    # PH
    # --------------------------------------------------------

    if ph is not None:

        if ph < 5.5 or ph > 8.5:

            score -= 35

            problems.append(
                "Water pH may stress vegetation."
            )

        elif ph < 6.0 or ph > 8.0:

            score -= 15

            problems.append(
                "Water pH is marginal for vegetation."
            )

    # --------------------------------------------------------
    # TURBIDITY
    # --------------------------------------------------------

    if turbidity is not None:

        if turbidity > 70:

            score -= 30

            problems.append(
                "High water turbidity."
            )

        elif turbidity > 40:

            score -= 10

            problems.append(
                "Elevated water turbidity."
            )

    # --------------------------------------------------------
    # NO DATA
    # --------------------------------------------------------

    if (
        quality is None
        and ph is None
        and turbidity is None
    ):

        return {

            "status": "Unknown",

            "score": None,

            "problems": [
                "No water-quality measurements recorded."
            ]
        }

    score = max(
        0,
        min(100, score)
    )

    if score >= 75:

        status = "Suitable"

    elif score >= 50:

        status = "Marginal"

    else:

        status = "Unsuitable"

    return {

        "status": status,

        "score": score,

        "problems": problems
    }


def assess_water_suitability(row):

    wildlife = classify_water_for_wildlife(
        row
    )

    vegetation = classify_water_for_vegetation(
        row
    )

    problems = []

    problems.extend(
        wildlife["problems"]
    )

    problems.extend(
        vegetation["problems"]
    )

    # Remove duplicates
    problems = list(
        dict.fromkeys(problems)
    )

    return {

        "wildlife": wildlife,

        "vegetation": vegetation,

        "problems": problems
    }


# ============================================================
# ENVIRONMENTAL COMPARISON
# ============================================================

def compare_environmental_value( current, previous, field_name):
    """
    Compare the CURRENT environmental value with the
    PREVIOUS survey.

    The current value always comes from the selected
    environmental observation.
    """

    current_value = safe_number(current.get(field_name))

    # There is no previous survey
    if previous is None:

        return {

            "current": current_value,

            "previous": None,

            "change": None,

            "percentage_change": None,

            "direction": "No Previous Record"
        }

    previous_value = safe_number( previous.get(field_name))

    if (current_value is None or previous_value is None ):

        return {

            "current": current_value,

            "previous": previous_value,

            "change": None,

            "percentage_change": None,

            "direction": "Insufficient Data"
        }

    change = ( current_value - previous_value)

    if previous_value != 0:

        percentage = (  change /  abs(previous_value) ) * 100

    else:
        percentage = None

    if change > 0:
        direction = "Increasing"

    elif change < 0:
        direction = "Decreasing"

    else:
        direction = "Stable"

    return {
        "current": round( current_value, 4),

        "previous": round( previous_value, 4 ),
        "change": round(change, 4),

        "percentage_change": (
            round(
                percentage,
                2
            )
            if percentage is not None
            else None
        ),

        "direction": direction
    }


# ============================================================
# VEGETATION
# ============================================================

def assess_vegetation_change(
    current,
    previous
):

    fields = [

        "vegetation_cover",

        "vegetation_density",

        "grass_availability",

        "tree_density"
    ]

    results = {}

    for field in fields:

        results[field] = (
            compare_environmental_value(
                current,
                previous,
                field
            )
        )

    declining = []

    increasing = []

    for field, result in results.items():

        if result["direction"] == "Decreasing":

            declining.append(field)

        elif result["direction"] == "Increasing":

            increasing.append(field)

    if len(declining) >= 2:

        overall = "Declining"

    elif len(increasing) >= 2:

        overall = "Increasing"

    elif declining:

        overall = "Some Decline"

    elif increasing:

        overall = "Some Increase"

    else:

        overall = "Stable / Unknown"

    return {

        "overall": overall,

        "measurements": results,

        "declining": declining,

        "increasing": increasing
    }


# ============================================================
# ENVIRONMENTAL HISTORY
# ============================================================

def get_environmental_history( site_id):

    records = ( EnvironmentalObservation.query .filter_by(monitoring_site_id=site_id)
        .order_by( EnvironmentalObservation.observation_date.asc(),
            EnvironmentalObservation.id.asc()
        )
        .all()
    )

    rows = []

    for env in records:
        site = env.monitoring_site
        if site is None:
            continue

        # Support either area_hectares or area
        env_area = getattr(
            env,
            "area_hectares",
            None
        )

        if env_area is None:

            env_area = getattr(
                env,
                "area",
                None
            )

        area = safe_number(
            env_area
        )

        if area is None:

            area = safe_number(
                site.area_hectares
            )

        rows.append({

            "environmental_observation_id":
                env.id,

            "section_key":
                section_key(site.id),

            "section_name":
                site.name,

            "area_hectares":
                area,

            "date":
                env.observation_date,

            "temperature":
                safe_number(
                    env.temperature
                ),

            "rainfall":
                safe_number(
                    env.rainfall
                ),

            "water_quality":
                safe_number(
                    env.water_quality
                ),

            "water_ph":
                safe_number(
                    env.water_ph
                ),

            "water_turbidity":
                safe_number(
                    env.water_turbidity
                ),

            "vegetation_cover":
                safe_number(
                    env.vegetation_cover
                ),

            "vegetation_density":
                safe_number(
                    env.vegetation_density
                ),

            "grass_availability":
                safe_number(
                    env.grass_availability
                ),

            "tree_density":
                safe_number(
                    env.tree_density
                ),

            "soil_ph":
                safe_number(
                    env.soil_ph
                ),

            "soil_moisture":
                safe_number(
                    env.soil_moisture
                ),

            "soil_quality":
                safe_number(
                    env.soil_quality
                )
        })

    if not rows:

        return pd.DataFrame()

    return (
        pd.DataFrame(rows)
        .sort_values(
            [
                "date",
                "environmental_observation_id"
            ]
        )
        .reset_index(drop=True)
    )


# ============================================================
# GET PREVIOUS SURVEY
# ============================================================

def get_previous_environmental_survey(site_id, current_environmental_observation_id):
    """
    Get the previous environmental survey for the SAME
    monitoring section.

    IMPORTANT:

    The current survey is identified by ID.

    The previous survey must have an earlier observation
    date than the current survey.

    Same-day records are not treated as the previous
    comparison record.
    """

    current = (
        EnvironmentalObservation.query
        .filter_by(
            id=current_environmental_observation_id,
            monitoring_site_id=site_id
        )
        .first()
    )

    if current is None:

        return None

    if current.observation_date is None:

        return None

    previous = (
        EnvironmentalObservation.query
        .filter(
            EnvironmentalObservation.monitoring_site_id
            == site_id,

            EnvironmentalObservation.id
            != current.id,

            EnvironmentalObservation.observation_date
            < current.observation_date
        )
        .order_by(
            EnvironmentalObservation.observation_date.desc(),
            EnvironmentalObservation.id.desc()
        )
        .first()
    )

    return previous


# ============================================================
# CONVERT ENVIRONMENTAL MODEL TO DICTIONARY
# ============================================================

def environmental_record_to_dict(
    env
):
    """
    Convert one EnvironmentalObservation database record
    into the structure used by the analysis engine.
    """

    site = env.monitoring_site

    if site is None:

        return None

    env_area = getattr(
        env,
        "area_hectares",
        None
    )

    if env_area is None:

        env_area = getattr(
            env,
            "area",
            None
        )

    area = safe_number(
        env_area
    )

    if area is None:

        area = safe_number(
            site.area_hectares
        )

    return {

        "environmental_observation_id":
            env.id,

        "section_id":
            site.id,

        "section_key":
            section_key(site.id),

        "section_name":
            site.name,

        "area_hectares":
            area,

        "date":
            env.observation_date,

        "temperature":
            safe_number(
                env.temperature
            ),

        "rainfall":
            safe_number(
                env.rainfall
            ),

        "water_quality":
            safe_number(
                env.water_quality
            ),

        "water_ph":
            safe_number(
                env.water_ph
            ),

        "water_turbidity":
            safe_number(
                env.water_turbidity
            ),

        "vegetation_cover":
            safe_number(
                env.vegetation_cover
            ),

        "vegetation_density":
            safe_number(
                env.vegetation_density
            ),

        "grass_availability":
            safe_number(
                env.grass_availability
            ),

        "tree_density":
            safe_number(
                env.tree_density
            ),

        "soil_ph":
            safe_number(
                env.soil_ph
            ),

        "soil_moisture":
            safe_number(
                env.soil_moisture
            ),

        "soil_quality":
            safe_number(
                env.soil_quality
            )
    }


SPECIES_CHANGE_CONFIG = {
    # Population change is considered notable at 20% or more.
    "population_pct": 20.0,

    # A minimum absolute difference helps avoid calling
    # 1 -> 2 individuals a large percentage change.
    "minimum_population_difference": 2,

    # A species must be missing from repeated surveys before
    # the system treats the absence as a stronger signal.
    "absence_surveys_for_attention": 2,

    # Minimum current population for percentage-based comparison.
    "minimum_population_for_pct": 5,
}


def get_species_history(site_id):
    """
    Retrieve ALL approved species observations for one monitoring
    section, ordered chronologically.

    Historical records are never deleted or replaced here.
    """
    observations = (
        Observation.query
        .join(
            EnvironmentalObservation,
            Observation.environmental_observation_id
            == EnvironmentalObservation.id
        )
        .filter(
            EnvironmentalObservation.monitoring_site_id == site_id,
            Observation.status == "Approved"
        )
        .order_by(
            EnvironmentalObservation.observation_date.asc(),
            EnvironmentalObservation.id.asc(),
            Observation.id.asc()
        )
        .all()
    )

    rows = []

    for obs in observations:
        if obs.species is None:
            continue

        env = getattr(obs, "environmental_observation", None)

        # The relationship may not be defined on the model, so use
        # a safe fallback query when necessary.
        if env is None:
            env = (
                EnvironmentalObservation.query
                .filter_by(id=obs.environmental_observation_id)
                .first()
            )

        if env is None:
            continue

        rows.append({
            "observation_id": obs.id,
            "species_id": obs.species_id,
            "species": obs.species.specie_Common_Name,
            "population": max(
                0,
                int(obs.population_count or 0)
            ),
            "environmental_observation_id": env.id,
            "date": env.observation_date,
        })

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


def get_current_species_records(environmental_observation_id):
    """
    Get species actually recorded in the selected survey.

    Multiple approved records for the same species in one survey
    are aggregated by species_id rather than overwriting one another.
    """
    rows = get_section_species_observations(
        environmental_observation_id
    )

    if rows.empty:
        return rows

    rows["population"] = pd.to_numeric(
        rows["population"],
        errors="coerce"
    ).fillna(0).clip(lower=0)

    grouped = (
        rows.groupby(
            ["species_id", "species"],
            as_index=False
        )
        .agg(
            population=("population", "sum"),
            observation_count=("observation_id", "count")
        )
    )

    return grouped


def get_previous_species_population( history, species_id, current_environmental_observation_id):
    """Get the most recent population before the current survey."""
    if history.empty:
        return None

    previous = history[
        (history["species_id"] == species_id)
        & (
            history["environmental_observation_id"]
            != current_environmental_observation_id
        )
    ]

    if previous.empty:
        return None

    return previous.sort_values(
        ["date", "observation_id"]
    ).iloc[-1]


def _species_population_change(
    current_population,
    previous_population
):
    """
    Calculate species population change without producing a misleading
    percentage when the baseline is zero or extremely small.
    """
    current_population = safe_number(current_population)
    previous_population = safe_number(previous_population)

    if current_population is None or previous_population is None:
        return {
            "change": None,
            "percentage_change": None,
            "direction": "Insufficient Data",
            "meaningful_change": False,
        }

    change = current_population - previous_population

    if change > 0:
        direction = "Increasing"
    elif change < 0:
        direction = "Decreasing"
    else:
        direction = "Stable"

    if previous_population != 0:
        pct = (
            change / abs(previous_population)
        ) * 100
    else:
        pct = None

    # Absolute difference matters when populations are small.
    abs_change_flag = (
        abs(change)
        >= SPECIES_CHANGE_CONFIG[
            "minimum_population_difference"
        ]
    )

    pct_change_flag = (
        pct is not None
        and abs(pct)
        >= SPECIES_CHANGE_CONFIG[
            "population_pct"
        ]
        and max(
            abs(current_population),
            abs(previous_population)
        )
        >= SPECIES_CHANGE_CONFIG[
            "minimum_population_for_pct"
        ]
    )

    meaningful = (
        abs_change_flag
        or pct_change_flag
    )

    return {
        "change": int(change),
        "percentage_change": (
            round(float(pct), 2)
            if pct is not None
            else None
        ),
        "direction": direction,
        "meaningful_change": bool(meaningful),
    }


def build_persistent_species_state( site_id, current_environmental_observation_id):
    """
    Build a persistent species state for a monitoring section.

    This is the key function for the user's requirement:

        Old species data stays available until a newer
        observation for that species exists.

    A missing species is NOT treated as extinct.

    Each species gets:
        - current_population: actual current-survey value
        - last_known_population: most recent approved value
        - last_observed_date
        - data_status
        - change from previous observation
        - number of surveys since last observation
    """
    history = get_species_history(site_id)

    current = get_current_species_records(
        current_environmental_observation_id
    )

    current_map = {}

    if not current.empty:
        for _, row in current.iterrows():
            current_map[int(row["species_id"])] = {
                "species_id": int(row["species_id"]),
                "species": str(row["species"]),
                "population": int(row["population"]),
                "observation_count": int(
                    row["observation_count"]
                ),
            }

    # No species history at all.
    if history.empty:
        return {
            "records": [],
            "current_species_count": 0,
            "historical_species_count": 0,
            "carried_forward_count": 0,
            "new_species_count": 0,
            "not_re_recorded_count": 0,
        }

    current_env = (
        EnvironmentalObservation.query
        .filter_by(id=current_environmental_observation_id)
        .first()
    )

    current_date = (
        current_env.observation_date
        if current_env is not None
        else None
    )

    # Distinct survey dates/IDs containing approved observations.
    survey_history = (
        history[
            [
                "environmental_observation_id",
                "date"
            ]
        ]
        .drop_duplicates()
        .sort_values(
            ["date", "environmental_observation_id"]
        )
    )

    records = []

    for species_id in sorted(
        history["species_id"].dropna().unique().tolist()
    ):
        species_id = int(species_id)

        species_history = history[
            history["species_id"] == species_id
        ].sort_values(
            ["date", "observation_id"]
        )

        if species_history.empty:
            continue

        latest = species_history.iloc[-1]

        current_record = current_map.get(
            species_id
        )

        previous_record = get_previous_species_population(
            history,
            species_id,
            current_environmental_observation_id
        )

        previous_population = (
            previous_record["population"]
            if previous_record is not None
            else None
        )

        current_population = (
            current_record["population"]
            if current_record is not None
            else None
        )

        change_info = _species_population_change(
            current_population,
            previous_population
        )

        observed_currently = (
            current_record is not None
        )

        first_observed_date = (
            species_history["date"].min()
        )

        last_observed_date = (
            latest["date"]
        )

        # Count completed surveys after the species' last observation.
        if current_date is not None:
            later_surveys = survey_history[
                survey_history["date"] > last_observed_date
            ]
            surveys_since = int(
                len(later_surveys)
            )
        else:
            surveys_since = 0

        is_new_species = (
            observed_currently
            and len(species_history) == 1
        )

        if observed_currently:
            data_status = (
                "Newly Observed"
                if is_new_species
                else "Observed Current Survey"
            )
            display_population = current_population
        else:
            data_status = "Carried Forward"
            display_population = int(
                latest["population"]
            )

        records.append({
            "species_id": species_id,
            "species": str(latest["species"]),
            "current_population": current_population,
            "last_known_population": int(
                latest["population"]
            ),
            "display_population": display_population,
            "first_observed_date": safe_iso_date(
                first_observed_date
            ),
            "last_observed_date": safe_iso_date(
                last_observed_date
            ),
            "last_observation_id": int(
                latest["observation_id"]
            ),
            "data_status": data_status,
            "observed_currently": observed_currently,
            "carried_forward": not observed_currently,
            "new_species": is_new_species,
            "surveys_since_last_observation": surveys_since,
            "previous_population": (
                int(previous_population)
                if previous_population is not None
                else None
            ),
            "change": change_info["change"],
            "percentage_change": change_info[
                "percentage_change"
            ],
            "direction": change_info["direction"],
            "meaningful_change": change_info[
                "meaningful_change"
            ],
            "attention_needed": (
                not observed_currently
                and surveys_since
                >= SPECIES_CHANGE_CONFIG[
                    "absence_surveys_for_attention"
                ]
            ),
        })

    # Add species that are in the current survey but have not
    # appeared in the historical query (normally impossible, but
    # retained as a safety net).
    known_ids = {
        r["species_id"]
        for r in records
    }

    for species_id, current_record in current_map.items():
        if species_id in known_ids:
            continue

        records.append({
            "species_id": species_id,
            "species": current_record["species"],
            "current_population": current_record["population"],
            "last_known_population": current_record["population"],
            "display_population": current_record["population"],
            "first_observed_date": safe_iso_date(
                current_date
            ),
            "last_observed_date": safe_iso_date(
                current_date
            ),
            "last_observation_id": None,
            "data_status": "Newly Observed",
            "observed_currently": True,
            "carried_forward": False,
            "new_species": True,
            "surveys_since_last_observation": 0,
            "previous_population": None,
            "change": None,
            "percentage_change": None,
            "direction": "New",
            "meaningful_change": False,
            "attention_needed": False,
        })

    records.sort(
        key=lambda r: r["species"].lower()
    )

    return {
        "records": records,
        "current_species_count": sum(
            1 for r in records
            if r["observed_currently"]
        ),
        "historical_species_count": len(records),
        "carried_forward_count": sum(
            1 for r in records
            if r["carried_forward"]
        ),
        "new_species_count": sum(
            1 for r in records
            if r["new_species"]
        ),
        "not_re_recorded_count": sum(
            1 for r in records
            if r["carried_forward"]
        ),
    }


def detect_species_changes(
    site_id,
    current_environmental_observation_id,
    previous_environmental_observation_id
):
    """
    Detect species-level changes.

    The system distinguishes:
      - new species
      - population increase
      - population decrease
      - unchanged population
      - species not recorded in the current survey
      - species missing from repeated surveys

    It NEVER labels one missed survey as extinction.
    """
    current = get_current_species_records(
        current_environmental_observation_id
    )

    previous = (
        get_current_species_records(
            previous_environmental_observation_id
        )
        if previous_environmental_observation_id is not None
        else pd.DataFrame()
    )

    current_map = {}
    previous_map = {}

    if not current.empty:
        current_map = {
            int(row["species_id"]): row
            for _, row in current.iterrows()
        }

    if not previous.empty:
        previous_map = {
            int(row["species_id"]): row
            for _, row in previous.iterrows()
        }

    all_ids = sorted(
        set(current_map) | set(previous_map)
    )

    changes = []

    for species_id in all_ids:
        cur = current_map.get(species_id)
        prev = previous_map.get(species_id)

        species_name = (
            str(cur["species"])
            if cur is not None
            else str(prev["species"])
        )

        if cur is not None and prev is None:
            changes.append({
                "species_id": species_id,
                "species": species_name,
                "status": "Newly Observed",
                "current_population": int(
                    cur["population"]
                ),
                "previous_population": None,
                "change": None,
                "percentage_change": None,
                "direction": "New",
                "meaningful_change": True,
            })
            continue

        if cur is None and prev is not None:
            changes.append({
                "species_id": species_id,
                "species": species_name,
                "status": "Not Recorded in Current Survey",
                "current_population": None,
                "previous_population": int(
                    prev["population"]
                ),
                "change": None,
                "percentage_change": None,
                "direction": "Not Observed",
                "meaningful_change": False,
            })
            continue

        change_info = _species_population_change(
            cur["population"],
            prev["population"]
        )

        if change_info["meaningful_change"]:
            status = (
                "Population Increased"
                if change_info["direction"] == "Increasing"
                else "Population Decreased"
            )
        else:
            status = "No Meaningful Population Change"

        changes.append({
            "species_id": species_id,
            "species": species_name,
            "status": status,
            "current_population": int(
                cur["population"]
            ),
            "previous_population": int(
                prev["population"]
            ),
            **change_info,
        })

    meaningful = [
        c for c in changes
        if c["meaningful_change"]
    ]

    new_species = [
        c for c in changes
        if c["status"] == "Newly Observed"
    ]

    decreased = [
        c for c in changes
        if c["status"] == "Population Decreased"
    ]

    increased = [
        c for c in changes
        if c["status"] == "Population Increased"
    ]

    not_recorded = [
        c for c in changes
        if c["status"] == "Not Recorded in Current Survey"
    ]

    return {
        "available": previous_environmental_observation_id is not None,
        "changes": changes,
        "meaningful_changes": meaningful,
        "new_species": new_species,
        "increased_species": increased,
        "decreased_species": decreased,
        "not_recorded_current_survey": not_recorded,
        "summary": {
            "total_species_compared": len(changes),
            "new_species_count": len(new_species),
            "increased_count": len(increased),
            "decreased_count": len(decreased),
            "not_recorded_count": len(not_recorded),
        },
        "interpretation_note": (
            "A species not recorded in one survey is not treated as "
            "extinct. Repeated non-detection should trigger monitoring "
            "attention, especially when survey effort is comparable."
        ),
    }


def get_species_history_for_section(site_id):
    """
    Return the complete historical species record for a section.

    Useful for charts, reports and an audit/history page.
    """
    history = get_species_history(site_id)

    if history.empty:
        return []

    results = []

    for _, row in history.iterrows():
        results.append({
            "observation_id": clean_value(
                row["observation_id"]
            ),
            "species_id": clean_value(
                row["species_id"]
            ),
            "species": row["species"],
            "population": clean_value(
                row["population"]
            ),
            "environmental_observation_id": clean_value(
                row["environmental_observation_id"]
            ),
            "date": (
                row["date"].isoformat()
                if pd.notna(row["date"])
                else None
            ),
        })

    return results



# ============================================================
# BIODIVERSITY
# ============================================================

def get_section_species_observations( environmental_observation_id):
    """
    Retrieve ONLY approved biodiversity observations
    belonging to the selected environmental survey.
    """

    observations = (
        Observation.query
        .filter_by(
            environmental_observation_id=
                environmental_observation_id,

            status="Approved"
        )
        .all()
    )

    rows = []

    for obs in observations:

        if obs.species is None:

            continue

        rows.append({

            "observation_id":
                obs.id,

            "species_id":
                obs.species_id,

            "species":
                obs.species.specie_Common_Name,

            "population":
                int(
                    obs.population_count or 0
                ),

            "date":
                obs.observation_date
        })

    if not rows:

        return pd.DataFrame()

    return pd.DataFrame(rows)


def assess_section_biodiversity(
    survey_rows,
    area_hectares
):

    if survey_rows.empty:

        return {

            "species_count": 0,

            "total_individuals": 0,

            "density": None,

            "species": [],

            "shannon_index": 0
        }

    species_count = (
        survey_rows["species"]
        .nunique()
    )

    total_individuals = int(
        survey_rows["population"].sum()
    )

    area = safe_number(
        area_hectares
    )

    if (
        area is not None
        and area > 0
    ):

        density = (
            total_individuals /
            area
        )

    else:

        density = None

    species_totals = (
        survey_rows
        .groupby("species")[
            "population"
        ]
        .sum()
    )

    total = species_totals.sum()

    if total > 0:

        proportions = (
            species_totals /
            total
        )

        shannon = -(
            proportions *
            np.log(proportions)
        ).sum()

    else:

        shannon = 0

    species_list = sorted(
        survey_rows["species"]
        .dropna()
        .unique()
        .tolist()
    )

    return {

        "species_count":
            int(species_count),

        "total_individuals":
            total_individuals,

        "density": (
            round(
                float(density),
                2
            )
            if density is not None
            else None
        ),

        "species":
            species_list,

        "shannon_index":
            round(
                float(shannon),
                4
            )
    }


# ============================================================
# RESOURCE PRESSURE


def calculate_resource_pressure(
    biodiversity,
    vegetation
):
    """
    Calculate current resource pressure using:

        1. Vegetation condition
        2. Current population density
        3. Species population increases/decreases
        4. Newly observed species

    IMPORTANT:

    Population increase:
        -> increases potential pressure on resources.

    Population decrease:
        -> reduces potential pressure from that species.

    A species that is not recorded in the current survey is
    NOT automatically treated as a population decrease.
    """

    species_count = biodiversity.get(
        "species_count",
        0
    )

    density = biodiversity.get(
        "density"
    )

    species_changes = biodiversity.get(
        "species_changes",
        {}
    )

    changes = species_changes.get(
        "changes",
        []
    )

    pressure_points = 0

    reasons = []

    increased_species = []
    decreased_species = []
    new_species = []

    # ========================================================
    # 1. VEGETATION
    # ========================================================

    if vegetation["overall"] == "Declining":

        pressure_points += 2

        reasons.append(
            "Vegetation resources are declining."
        )

    elif vegetation["overall"] == "Some Decline":

        pressure_points += 1

        reasons.append(
            "Some vegetation indicators are declining."
        )

    elif vegetation["overall"] == "Increasing":

        reasons.append(
            "Vegetation resources are increasing."
        )

    elif vegetation["overall"] == "Some Increase":

        reasons.append(
            "Some vegetation indicators are increasing."
        )

    # ========================================================
    # 2. SPECIES POPULATION CHANGES
    # ========================================================

    for change in changes:

        status = change.get(
            "status"
        )

        species_name = change.get(
            "species",
            "Unknown species"
        )

        current_population = change.get(
            "current_population"
        )

        previous_population = change.get(
            "previous_population"
        )

        # ----------------------------------------------------
        # POPULATION INCREASE
        # ----------------------------------------------------

        if status == "Population Increased":

            increased_species.append({
                "species": species_name,
                "current_population": current_population,
                "previous_population": previous_population,
                "change": change.get("change"),
                "percentage_change": change.get(
                    "percentage_change"
                )
            })

        # ----------------------------------------------------
        # POPULATION DECREASE
        # ----------------------------------------------------

        elif status == "Population Decreased":

            decreased_species.append({
                "species": species_name,
                "current_population": current_population,
                "previous_population": previous_population,
                "change": change.get("change"),
                "percentage_change": change.get(
                    "percentage_change"
                )
            })

        # ----------------------------------------------------
        # NEW SPECIES
        # ----------------------------------------------------

        elif status == "Newly Observed":

            new_species.append({
                "species": species_name,
                "current_population": current_population
            })

    # ========================================================
    # 3. INCREASED SPECIES = INCREASED RESOURCE PRESSURE
    # ========================================================

    if increased_species:

        # Each meaningful population increase contributes
        # additional resource pressure.
        increase_pressure = min(
            len(increased_species),
            2
        )

        pressure_points += increase_pressure

        names = [
            item["species"]
            for item in increased_species
        ]

        if len(names) == 1:

            reasons.append(
                f"The population of {names[0]} increased, "
                "which may increase pressure on available "
                "food, vegetation, water and habitat."
            )

        else:

            reasons.append(
                f"Populations of {len(names)} species increased, "
                "which may increase pressure on available "
                "food, vegetation, water and habitat."
            )

    # ========================================================
    # 4. DECREASED SPECIES = REDUCED RESOURCE PRESSURE
    # ========================================================

    if decreased_species:

        # Decreases reduce resource demand.
        #
        # We do NOT allow the reduction to create a negative
        # pressure score. It can only offset population-based
        # pressure.
        decrease_relief = min(
            len(decreased_species),
            2
        )

        pressure_points -= decrease_relief

        names = [
            item["species"]
            for item in decreased_species
        ]

        if len(names) == 1:

            reasons.append(
                f"The population of {names[0]} decreased, "
                "which reduces its pressure on available "
                "food, vegetation, water and habitat."
            )

        else:

            reasons.append(
                f"Populations of {len(names)} species decreased, "
                "which reduces their pressure on available "
                "food, vegetation, water and habitat."
            )

    # ========================================================
    # 5. NEW SPECIES
    # ========================================================

    if new_species:

        # A newly recorded species adds potential resource
        # demand, but it is treated more cautiously than a
        # confirmed population increase.
        pressure_points += 1

        names = [
            item["species"]
            for item in new_species
        ]

        if len(names) == 1:

            reasons.append(
                f"{names[0]} was newly recorded in the current "
                "survey and may add additional demand on "
                "available resources."
            )

        else:

            reasons.append(
                f"{len(names)} species were newly recorded in "
                "the current survey and may add additional "
                "demand on available resources."
            )

    # ========================================================
    # 6. CURRENT POPULATION DENSITY
    # ========================================================

    if density is not None:

        if density > 100:

            pressure_points += 2

            reasons.append(
                "High recorded population density relative "
                "to section area."
            )

        elif density > 50:

            pressure_points += 1

            reasons.append(
                "Moderately high recorded population density "
                "relative to section area."
            )

    # ========================================================
    # 7. NO SPECIES
    # ========================================================

    if species_count == 0:

        reasons.append(
            "No approved species observations were recorded "
            "for this survey."
        )

    # ========================================================
    # PREVENT NEGATIVE PRESSURE
    # ========================================================

    pressure_points = max(
        0,
        pressure_points
    )

    # ========================================================
    # PRESSURE LEVEL
    # ========================================================

    if pressure_points >= 4:

        level = "Very High"

    elif pressure_points >= 2:

        level = "High"

    elif pressure_points >= 1:

        level = "Moderate"

    else:

        level = "Low"

    # ========================================================
    # RETURN
    # ========================================================

    return {

        "pressure": level,

        "score": pressure_points,

        "reasons": reasons,

        # -----------------------------------------------
        # SPECIES CHANGE INFORMATION
        # -----------------------------------------------

        "increased_species": increased_species,

        "decreased_species": decreased_species,

        "new_species": new_species,

        "increased_species_count":
            len(increased_species),

        "decreased_species_count":
            len(decreased_species),

        "new_species_count":
            len(new_species),

        # -----------------------------------------------
        # EXPLICIT RESOURCE EFFECT
        # -----------------------------------------------

        "population_resource_effect": (

            "Increasing populations are adding pressure "
            "on available resources."
            if increased_species
            else

            "Decreasing populations are reducing pressure "
            "on available resources."
            if decreased_species
            else

            "No meaningful population change affecting "
            "resource pressure was recorded."
        )
    }

# ============================================================
# TEMPERATURE EFFECT
# ============================================================

def assess_temperature_effect(
    temperature_change
):

    change = temperature_change[
        "change"
    ]

    if change is None:

        return {

            "effect": "Unknown",

            "reasons": [
                "No previous temperature measurement."
            ]
        }

    if change >= 5:

        return {

            "effect":
                "Potential environmental stress",

            "reasons": [
                "Temperature increased substantially."
            ]
        }

    if change >= 2:

        return {

            "effect":
                "Potential warming pressure",

            "reasons": [
                "Temperature increased."
            ]
        }

    if change <= -5:

        return {

            "effect":
                "Potential cooling effect",

            "reasons": [
                "Temperature decreased substantially."
            ]
        }

    if change <= -2:

        return {

            "effect":
                "Cooling observed",

            "reasons": [
                "Temperature decreased."
            ]
        }

    return {

        "effect":
            "Limited recorded change",

        "reasons": [
            "Temperature changed only slightly."
        ]
    }


# ============================================================
# RAINFALL EFFECT
# ============================================================

def assess_rainfall_effect(
    rainfall_change
):

    change = rainfall_change[
        "change"
    ]

    percentage = rainfall_change[
        "percentage_change"
    ]

    if change is None:

        return {

            "effect": "Unknown",

            "reasons": [
                "No previous rainfall measurement."
            ]
        }

    if percentage is not None:

        if percentage <= -30:

            return {

                "effect":
                    "Potential reduced water/resource availability",

                "reasons": [
                    "Rainfall decreased substantially."
                ]
            }

        if percentage >= 30:

            return {

                "effect":
                    "Potential increased water availability",

                "reasons": [
                    "Rainfall increased substantially."
                ]
            }

    if change < 0:

        return {

            "effect":
                "Rainfall decreased",

            "reasons": [
                "Rainfall decreased compared with "
                "the previous survey."
            ]
        }

    if change > 0:

        return {

            "effect":
                "Rainfall increased",

            "reasons": [
                "Rainfall increased compared with "
                "the previous survey."
            ]
        }

    return {

        "effect":
            "Stable rainfall",

        "reasons": []
    }


# ============================================================
# LAND SECTION IMPACT
# ============================================================

def assess_section_land_impact(
    biodiversity,
    vegetation,
    water,
    rainfall,
    temperature,
    resource
):

    score = 0

    factors = []

    # --------------------------------------------------------
    # VEGETATION
    # --------------------------------------------------------

    if vegetation["overall"] == "Declining":

        score += 2

        factors.append(
            "Vegetation resources are declining."
        )

    elif vegetation["overall"] == "Some Decline":

        score += 1

        factors.append(
            "Some vegetation indicators are declining."
        )

    # --------------------------------------------------------
    # WATER
    # --------------------------------------------------------

    wildlife_status = (
        water["wildlife"]["status"]
    )

    vegetation_status = (
        water["vegetation"]["status"]
    )

    if wildlife_status == "Unsuitable":

        score += 2

        factors.append(
            "Water is unsuitable under the wildlife "
            "screening criteria."
        )

    elif wildlife_status == "Marginal":

        score += 1

        factors.append(
            "Wildlife water suitability is marginal."
        )

    if vegetation_status == "Unsuitable":

        score += 2

        factors.append(
            "Water conditions are unsuitable under "
            "the vegetation screening criteria."
        )

    elif vegetation_status == "Marginal":

        score += 1

        factors.append(
            "Water suitability for vegetation is marginal."
        )

    # --------------------------------------------------------
    # RESOURCE PRESSURE
    # --------------------------------------------------------

    if resource["pressure"] == "Very High":

        score += 2

        factors.append(
            "Resource pressure is very high."
        )

    elif resource["pressure"] == "High":

        score += 1

        factors.append(
            "Resource pressure is high."
        )

    # --------------------------------------------------------
    # TEMPERATURE
    # --------------------------------------------------------

    temp_change = temperature[
        "change"
    ]

    if temp_change is not None:

        if temp_change >= 5:

            score += 2

            factors.append(
                "Temperature increased substantially."
            )

        elif temp_change >= 2:

            score += 1

            factors.append(
                "Temperature increased."
            )

    # --------------------------------------------------------
    # RAINFALL
    # --------------------------------------------------------

    rainfall_change = rainfall[
        "change"
    ]

    rainfall_pct = rainfall[
        "percentage_change"
    ]

    if (
        rainfall_change is not None
        and rainfall_change < 0
    ):

        if (
            rainfall_pct is not None
            and abs(rainfall_pct) >= 30
        ):

            score += 2

            factors.append(
                "Rainfall decreased substantially."
            )

        else:

            score += 1

            factors.append(
                "Rainfall decreased."
            )

    # --------------------------------------------------------
    # CLASSIFICATION
    # --------------------------------------------------------

    if score >= 7:

        condition = (
            "High Environmental Pressure"
        )

    elif score >= 4:

        condition = (
            "Moderate Environmental Pressure"
        )

    elif score >= 1:

        condition = (
            "Some Environmental Pressure"
        )

    else:

        condition = (
            "Low Recorded Environmental Pressure"
        )

    return {

        "score": score,

        "condition": condition,

        "factors": factors,

        "species_count":
            biodiversity[
                "species_count"
            ],

        "total_individuals":
            biodiversity[
                "total_individuals"
            ],

        "density":
            biodiversity[
                "density"
            ]
    }


# ============================================================
# MAIN SECTION ANALYSIS
# ============================================================

def analyse_land_section(
    environmental_observation_id
):
    """
    Analyse ONE CURRENT environmental survey.

    The selected environmental_observation_id is ALWAYS
    treated as the CURRENT survey.

    A previous survey is retrieved only for comparison.
    """

    # ========================================================
    # FIND CURRENT SURVEY
    # ========================================================

    current = (
        EnvironmentalObservation.query
        .filter_by(
            id=environmental_observation_id
        )
        .first()
    )

    if current is None:

        raise ValueError(
            "Environmental observation not found."
        )

    # ========================================================
    # FIND MONITORING SITE
    # ========================================================

    site = current.monitoring_site

    if site is None:

        raise ValueError(
            "Environmental observation has no "
            "monitoring section."
        )

    # ========================================================
    # CURRENT DATA
    # ========================================================

    current_data = (
        environmental_record_to_dict(
            current
        )
    )

    if current_data is None:

        raise ValueError(
            "Unable to read the current environmental survey."
        )

    # ========================================================
    # PREVIOUS SURVEY
    # ========================================================
    #
    # IMPORTANT:
    #
    # This record is used for comparison only.
    #
    # It is NOT returned as the current survey.
    # ========================================================

    previous_record = (
        get_previous_environmental_survey(
            site.id,
            current.id
        )
    )

    # Convert previous model to dictionary

    if previous_record is not None:

        previous_data = (
            environmental_record_to_dict(
                previous_record
            )
        )

    else:

        previous_data = None

    # ========================================================
    # TEMPERATURE
    # ========================================================

    temperature_change = (
        compare_environmental_value(
            current_data,
            previous_data,
            "temperature"
        )
    )

    temperature_effect = (
        assess_temperature_effect(
            temperature_change
        )
    )

    # ========================================================
    # RAINFALL
    # ========================================================

    rainfall_change = (
        compare_environmental_value(
            current_data,
            previous_data,
            "rainfall"
        )
    )

    rainfall_effect = (
        assess_rainfall_effect(
            rainfall_change
        )
    )

    # ========================================================
    # VEGETATION
    # ========================================================

    vegetation_change = (
        assess_vegetation_change(
            current_data,
            previous_data
        )
    )

    # ========================================================
    # WATER
    # ========================================================

    water = assess_water_suitability(
        current_data
    )

    # ========================================================
    # BIODIVERSITY
    # ========================================================
    #
    # ONLY observations connected to the CURRENT survey.
    # ========================================================

    species_rows = (
        get_section_species_observations(
            current.id
        )
    )

    biodiversity = (
        assess_section_biodiversity(
            species_rows,
            current_data[
                "area_hectares"
            ]
        )
    )


    # ========================================================
    # SPECIES PERSISTENCE + SPECIES-LEVEL CHANGE
    # ========================================================

    persistent_species = (
        build_persistent_species_state(
            site_id=site.id,
            current_environmental_observation_id=current.id
        )
    )

    species_changes = (
        detect_species_changes(
            site_id=site.id,
            current_environmental_observation_id=current.id,
            previous_environmental_observation_id=(
                previous_record.id
                if previous_record is not None
                else None
            )
        )
    )

    # Keep the current biodiversity metrics based ONLY on species
    # actually observed in the current survey. Carried-forward data
    # is history/display data and must not silently inflate current
    # biodiversity measurements.
    biodiversity["persistent_species"] = persistent_species
    biodiversity["species_changes"] = species_changes

    # ========================================================
    # RESOURCE PRESSURE
    # ========================================================

    resource = (
        calculate_resource_pressure(
            biodiversity,
            vegetation_change
        )
    )

    # ========================================================
    # LAND IMPACT
    # ========================================================

    land_impact = (
        assess_section_land_impact(
            biodiversity=biodiversity,
            vegetation=vegetation_change,
            water=water,
            rainfall=rainfall_change,
            temperature=temperature_change,
            resource=resource
        )
    )

    # ========================================================
    # FINAL RESPONSE
    # ========================================================

    return {

        # ====================================================
        # CURRENT SECTION
        # ====================================================

        "section": {

            "id":
                site.id,

            "key":
                section_key(
                    site.id
                ),

            "name":
                site.name,

            "area_hectares":
                current_data[
                    "area_hectares"
                ]
        },

        # ====================================================
        # CURRENT SURVEY
        # ====================================================

        "survey": {

            "id":
                current.id,

            "date":
                safe_iso_date(
                    current.observation_date
                ),

            "previous_survey_date": (
                safe_iso_date(
                    previous_record.observation_date
                )
                if previous_record is not None
                else None
            ),

            "is_current":
                True
        },

        # ====================================================
        # WATER
        # ====================================================

        "water": {

            "quality":
                current_data[
                    "water_quality"
                ],

            "ph":
                current_data[
                    "water_ph"
                ],

            "turbidity":
                current_data[
                    "water_turbidity"
                ],

            "wildlife":
                water[
                    "wildlife"
                ],

            "vegetation":
                water[
                    "vegetation"
                ],

            "problems":
                water[
                    "problems"
                ]
        },

        # ====================================================
        # TEMPERATURE
        # ====================================================

        "temperature": {

            **temperature_change,

            "effect":
                temperature_effect
        },

        # ====================================================
        # RAINFALL
        # ====================================================

        "rainfall": {

            **rainfall_change,

            "effect":
                rainfall_effect
        },

        # ====================================================
        # VEGETATION
        # ====================================================

        "vegetation":
            vegetation_change,

        # ====================================================
        # BIODIVERSITY
        # ====================================================

        "biodiversity":
            biodiversity,

        # ====================================================
        # SPECIES CHANGE DETECTION
        # ====================================================

        "species_change_detection": {
            "persistent_species":
                persistent_species,

            "changes":
                species_changes
        },

        # ====================================================
        # RESOURCE PRESSURE
        # ====================================================

        "resource_pressure":
            resource,

        # ====================================================
        # LAND IMPACT
        # ====================================================

        "land_impact":
            land_impact,

        # ====================================================
        # CURRENT ENVIRONMENTAL CONDITIONS
        # ====================================================

        "current_conditions": {

            "temperature":
                current_data[
                    "temperature"
                ],

            "rainfall":
                current_data[
                    "rainfall"
                ],

            "water_quality":
                current_data[
                    "water_quality"
                ],

            "water_ph":
                current_data[
                    "water_ph"
                ],

            "water_turbidity":
                current_data[
                    "water_turbidity"
                ],

            "vegetation_cover":
                current_data[
                    "vegetation_cover"
                ],

            "vegetation_density":
                current_data[
                    "vegetation_density"
                ],

            "grass_availability":
                current_data[
                    "grass_availability"
                ],

            "tree_density":
                current_data[
                    "tree_density"
                ],

            "soil_ph":
                current_data[
                    "soil_ph"
                ],

            "soil_moisture":
                current_data[
                    "soil_moisture"
                ],

            "soil_quality":
                current_data[
                    "soil_quality"
                ]
        }
    }


# ============================================================
# SECTION HISTORY
# ============================================================

def analyse_section_history(
    site_id
):
    """
    Return historical environmental records for charts
    and trend analysis.

    This is separate from the CURRENT survey displayed
    on the analysis page.
    """

    history = (
        get_environmental_history(
            site_id
        )
    )

    if history.empty:

        return []

    results = []

    for _, row in history.iterrows():

        results.append({

            "environmental_observation_id":
                clean_value(
                    row[
                        "environmental_observation_id"
                    ]
                ),

            "date":
                (
                    row["date"].isoformat()
                    if pd.notna(
                        row["date"]
                    )
                    else None
                ),

            "temperature":
                clean_value(
                    row["temperature"]
                ),

            "rainfall":
                clean_value(
                    row["rainfall"]
                ),

            "water_quality":
                clean_value(
                    row["water_quality"]
                ),

            "water_ph":
                clean_value(
                    row["water_ph"]
                ),

            "water_turbidity":
                clean_value(
                    row["water_turbidity"]
                ),

            "vegetation_cover":
                clean_value(
                    row["vegetation_cover"]
                ),

            "vegetation_density":
                clean_value(
                    row["vegetation_density"]
                ),

            "grass_availability":
                clean_value(
                    row["grass_availability"]
                ),

            "tree_density":
                clean_value(
                    row["tree_density"]
                ),

            "soil_ph":
                clean_value(
                    row["soil_ph"]
                ),

            "soil_moisture":
                clean_value(
                    row["soil_moisture"]
                ),

            "soil_quality":
                clean_value(
                    row["soil_quality"]
                )
        })

    return results



# ============================================================
# SECTION SPECIES HISTORY
# ============================================================

def analyse_species_history(site_id):
    """
    Return species history separately from the current survey.

    This allows the UI/report to show:
        Current observation
        Previous observations
        Last known population
        Population changes over time
    without replacing old records.
    """
    return get_species_history_for_section(site_id)


# ============================================================
# GET LATEST SURVEY FOR ONE MONITORING SITE
# ============================================================

def get_latest_environmental_survey(
    site_id
):
    """
    Return the CURRENT/latest survey for one monitoring site.
    """

    return (
        EnvironmentalObservation.query
        .filter_by(
            monitoring_site_id=site_id
        )
        .order_by(
            EnvironmentalObservation.observation_date.desc(),
            EnvironmentalObservation.id.desc()
        )
        .first()
    )


# ============================================================
# GET CURRENT SURVEY FOR EVERY SECTION
# ============================================================

def get_current_section_surveys():
    """
    Return ONLY the latest environmental survey for every
    monitoring section.

    Older surveys are not returned here.
    """

    sites = (
        MonitoringSite.query
        .order_by(
            MonitoringSite.name.asc()
        )
        .all()
    )

    results = []

    for site in sites:

        current = (
            get_latest_environmental_survey(
                site.id
            )
        )

        if current is None:

            continue

        results.append(
            current
        )

    results.sort(
        key=lambda survey: (
            survey.observation_date,
            survey.id
        ),
        reverse=True
    )

    return results


# ============================================================
# ALL SECTION SUMMARIES
# ============================================================

def get_all_section_summaries():
    """
    Analyse the CURRENT/latest survey for every monitoring
    section.

    Historical surveys are not treated as current.
    """

    sites = (
        MonitoringSite.query
        .order_by(
            MonitoringSite.name.asc()
        )
        .all()
    )

    results = []

    for site in sites:

        # Get ONLY the latest survey
        latest = (
            get_latest_environmental_survey(
                site.id
            )
        )

        # ----------------------------------------------------
        # SECTION HAS NO SURVEY
        # ----------------------------------------------------

        if latest is None:

            results.append({

                "section_id":
                    site.id,

                "section_name":
                    site.name,

                "area_hectares":
                    safe_number(
                        site.area_hectares
                    ),

                "current_survey_id":
                    None,

                "analysis":
                    None
            })

            continue

        # ----------------------------------------------------
        # ANALYSE CURRENT SURVEY
        # ----------------------------------------------------

        try:

            analysis_result = (
                analyse_land_section(
                    latest.id
                )
            )

        except Exception as exc:

            analysis_result = {

                "error":
                    str(exc)
            }

        results.append({

            "section_id":
                site.id,

            "section_name":
                site.name,

            "area_hectares":
                safe_number(
                    site.area_hectares
                ),

            "current_survey_id":
                latest.id,

            "current_survey_date":
                safe_iso_date(
                    latest.observation_date
                ),

            "analysis":
                analysis_result
        })

    return results