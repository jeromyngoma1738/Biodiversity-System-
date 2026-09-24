
from flask import Flask, flash, redirect, url_for, request, render_template, session, send_file
from flask_sqlalchemy import SQLAlchemy
from extensions import db
from models import Details, Species, Observation, Notification, Analysis,EnvironmentalObservation,MonitoringSite
from datetime import datetime, timezone
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
import os
import math
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
import matplotlib.pyplot as plt
from reportlab.lib.units import inch
from sqlalchemy import or_
from werkzeug.utils import secure_filename
from mail_config import mail, configure_mail 
import secrets
from sqlalchemy import func
from datetime import datetime, timedelta
from datetime import datetime, timezone
import uuid
import numpy as np
from routes.ecosystem import analyse_land_section
from routes.AI_Analysis_Model import ( generate_species_effect_report, forecast_species_population)
import requests
import pandas as pd
import requests
from io import StringIO
from datetime import datetime, timezone
from sklearn.linear_model import LinearRegression


app = Flask(__name__)

configure_mail(app)

app.secret_key = "my_super_secret_key_12345"


app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///Biodiversity.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True

db.init_app(app)


# --- Decorators ---
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def role_required(*roles):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if 'user_id' not in session:
                return redirect(url_for('login'))
            if session.get('role') not in roles:
                return "Access Denied", 403
            return f(*args, **kwargs)
        return decorated_function
    return decorator  


# ============================================================
# SPECIES POPULATION ANALYSIS
# ============================================================

from flask import jsonify

@app.route("/api/section/<int:environmental_observation_id>/analysis")
@login_required
def section_analysis(environmental_observation_id):
    try:
        result = analyse_land_section(environmental_observation_id)

        return jsonify({"success": True,"data": result })

    except ValueError as exc:
        return jsonify({ "success": False, "error": str(exc) }), 404

    except Exception:
        app.logger.exception("Section analysis failed")

        return jsonify({
            "success": False,
            "error": "Unable to analyse this section."
        }), 500

def analysis():

    species_list = Species.query.all()

    results = []

    for species in species_list:

        observations = (
            Observation.query
            .filter_by(
                species_id=species.id,
                status="Approved"
            )
            .order_by(
                Observation.observation_date.desc()
            )
            .all()
        )

        # Need at least two observations
        if len(observations) < 2:
            continue

        # Latest observation
        current = observations[0]

        # Previous observation
        previous = observations[1]

        current_population = current.population_count or 0

        previous_population = previous.population_count or 0

        # Population change
        population_change = (
            current_population - previous_population
        )

        # Percentage change
        if previous_population > 0:

            percentage_change = (
                population_change /
                previous_population
            ) * 100

        else:

            percentage_change = 0

        # Determine trend
        if population_change > 0:

            trend = "Increasing"

        elif population_change < 0:

            trend = "Declining"

        else:

            trend = "Stable"

        # Add result
        results.append({

            "species_id": species.id,

            "species": species.specie_Common_Name,

            "scientific_name": getattr(
                species,
                "scientificName",
                "N/A"
            ),

            "habitat": species.specie_Habitat,

            "location": species.location,

            "current_population": current_population,

            "previous_population": previous_population,

            "population_change": population_change,

            "percentage_change": round(
                percentage_change,
                2
            ),

            "trend": trend,

            "current_date": current.observation_date,

            "previous_date": previous.observation_date

        })

    return results

# ============================================================
# CURRENT POPULATION ESTIMATION
# ============================================================

def estimate_species_population(df):
    """
    Estimate the current population of a species using
    approved historical observations.

    The estimate considers:

    1. Number of approved observations
    2. Recency of observations
    3. Historical population values
    4. Population trend
    5. Historical variability

    IMPORTANT:
    This is an ESTIMATE, not a replacement for an actual
    field observation.
    """

    # ========================================================
    # VALIDATION
    # ========================================================

    if df is None or df.empty:
        return {
            "success": False,
            "error": "No population data is available."
        }

    required_columns = {
        "date",
        "population"
    }

    if not required_columns.issubset(df.columns):
        return {
            "success": False,
            "error": "Population data is incomplete."
        }

    data = df.copy()

    # ========================================================
    # CLEAN DATA
    # ========================================================

    data["date"] = pd.to_datetime(
        data["date"],
        errors="coerce"
    )

    data["population"] = pd.to_numeric(
        data["population"],
        errors="coerce"
    )

    data = data.dropna(
        subset=[
            "date",
            "population"
        ]
    )

    data = data[
        data["population"] >= 0
    ]

    data = (
        data
        .sort_values("date")
        .reset_index(drop=True)
    )

    observation_count = len(data)

    # ========================================================
    # MINIMUM DATA
    # ========================================================

    if observation_count < 3:
        return {
            "success": False,
            "error": (
                "At least 3 approved observations are "
                "required to estimate population."
            ),
            "approved_observations": observation_count
        }

    # ========================================================
    # CURRENT / LATEST OBSERVATION
    # ========================================================

    latest_date = data["date"].iloc[-1]

    latest_population = float(
        data["population"].iloc[-1]
    )

    # ========================================================
    # RECENCY WEIGHT
    # ========================================================
    #
    # More recent observations receive more influence.
    #
    # The decay period is approximately one year.
    #
    # ========================================================

    data["days_old"] = (
        latest_date
        - data["date"]
    ).dt.total_seconds() / 86400

    decay_days = 365.25

    data["recency_weight"] = np.exp(
        -data["days_old"] / decay_days
    )

    # ========================================================
    # RECENCY-WEIGHTED POPULATION
    # ========================================================

    weighted_population = (
        (
            data["population"]
            *
            data["recency_weight"]
        ).sum()
        /
        data["recency_weight"].sum()
    )

    weighted_population = float(
        weighted_population
    )

    # ========================================================
    # HISTORICAL MEAN
    # ========================================================

    historical_mean = float(
        data["population"].mean()
    )

    # ========================================================
    # HISTORICAL MEDIAN
    # ========================================================

    historical_median = float(
        data["population"].median()
    )

    # ========================================================
    # VARIABILITY
    # ========================================================

    standard_deviation = float(
        data["population"].std(
            ddof=1
        )
    ) if observation_count > 1 else 0.0

    if not np.isfinite(
        standard_deviation
    ):
        standard_deviation = 0.0

    # ========================================================
    # TREND MODEL
    # ========================================================

    first_date = data["date"].iloc[0]

    data["years_since_start"] = (
        (
            data["date"]
            -
            first_date
        ).dt.total_seconds()
        /
        (365.25 * 24 * 60 * 60)
    )

    X = data[
        ["years_since_start"]
    ].astype(float)

    y = data[
        "population"
    ].astype(float)

    trend_model = LinearRegression()

    trend_model.fit(
        X,
        y
    )

    slope = float(
        trend_model.coef_[0]
    )

    # ========================================================
    # CURRENT MODEL ESTIMATE
    # ========================================================

    latest_time = float(
        data["years_since_start"].iloc[-1]
    )

    model_current_population = float(
        trend_model.predict(
            pd.DataFrame({
                "years_since_start": [
                    latest_time
                ]
            })
        )[0]
    )

    model_current_population = max(
        0,
        model_current_population
    )

    # ========================================================
    # COMBINE EVIDENCE
    # ========================================================
    #
    # The estimate combines:
    #
    # 50% recent observations
    # 20% historical median
    # 30% trend model
    #
    # This avoids allowing one unusual observation to
    # completely determine the estimate.
    #
    # ========================================================

    estimated_population = (
        (weighted_population * 0.50)
        +
        (historical_median * 0.20)
        +
        (model_current_population * 0.30)
    )

    estimated_population = max(
        0,
        float(estimated_population)
    )

    # ========================================================
    # ESTIMATED RANGE
    # ========================================================

    estimated_lower = max(
        0,
        estimated_population
        -
        standard_deviation
    )

    estimated_upper = (
        estimated_population
        +
        standard_deviation
    )

    # ========================================================
    # TREND
    # ========================================================

    if slope > 0.01:
        trend = "Increasing"

    elif slope < -0.01:
        trend = "Declining"

    else:
        trend = "Stable"

    # ========================================================
    # DATA CONSISTENCY
    # ========================================================

    coefficient_variation = None

    if historical_mean > 0:
        coefficient_variation = (
            standard_deviation
            /
            historical_mean
        )

    # ========================================================
    # CONFIDENCE
    # ========================================================

    if (
        observation_count >= 10
        and
        coefficient_variation is not None
        and
        coefficient_variation <= 0.20
    ):
        confidence = "High"

    elif (
        observation_count >= 5
        and
        coefficient_variation is not None
        and
        coefficient_variation <= 0.40
    ):
        confidence = "Moderate"

    else:
        confidence = "Low"

    # ========================================================
    # PERCENTAGE DIFFERENCE FROM ACTUAL OBSERVATION
    # ========================================================

    if latest_population > 0:

        estimate_difference = (
            (
                estimated_population
                -
                latest_population
            )
            /
            latest_population
        ) * 100

    else:
        estimate_difference = None

    # ========================================================
    # RETURN
    # ========================================================

    return {

        "success": True,

        "latest_observed_population": round(
            latest_population,
            2
        ),

        "estimated_population": round(
            estimated_population,
            2
        ),

        "estimated_lower": round(
            estimated_lower,
            2
        ),

        "estimated_upper": round(
            estimated_upper,
            2
        ),

        "historical_mean": round(
            historical_mean,
            2
        ),

        "historical_median": round(
            historical_median,
            2
        ),

        "standard_deviation": round(
            standard_deviation,
            2
        ),

        "approved_observations": int(
            observation_count
        ),

        "trend": trend,

        "slope_per_year": round(
            slope,
            2
        ),

        "estimate_difference_percent": (
            round(
                estimate_difference,
                2
            )
            if estimate_difference is not None
            else None
        ),

        "confidence": confidence
    }
# ============================================================
# QUARTERLY POPULATION FORECASTING
# ============================================================
# ============================================================
# CBU NATURE PARK
# QUARTERLY POPULATION FORECASTING
# ============================================================

QUARTERS_PER_YEAR = 4

# A forecast should not suddenly grow or decline by an
# unrealistic amount in one 3-month period.
#
# This is a stability safeguard, NOT a biological law.
MAX_QUARTERLY_GROWTH = 0.15
MAX_QUARTERLY_DECLINE = -0.15


# ============================================================
# SAFE NUMBER
# ============================================================

def safe_float(value):
    """
    Convert a value to float safely.
    """

    try:
        number = float(value)

        if not np.isfinite(number):
            return None

        return number

    except (ValueError, TypeError):
        return None


# ============================================================
# RESOURCE VALUE
# ============================================================

def resource_value(value):
    """
    Convert environmental resource values into a 0-1 score.

    Supports:
        0-100 numeric percentages
        0-1 numeric values
        Excellent / Good / Fair / Poor
        High / Medium / Low
        Very High / Very Low
    """

    if value is None:
        return None

    # --------------------------------------------------------
    # Numeric
    # --------------------------------------------------------

    number = safe_float(value)

    if number is not None:

        if 0 <= number <= 1:
            return number

        if 0 <= number <= 100:
            return number / 100

        return None

    # --------------------------------------------------------
    # Text
    # --------------------------------------------------------

    text = str(value).strip().lower()

    mapping = {

        "excellent": 1.00,
        "very good": 0.90,
        "good": 0.80,
        "high": 0.80,

        "fair": 0.60,
        "moderate": 0.60,
        "medium": 0.60,

        "poor": 0.30,
        "low": 0.30,

        "very poor": 0.15,
        "very low": 0.15,

        "bad": 0.20,
        "critical": 0.10
    }

    return mapping.get(text)


# ============================================================
# RESOURCE SCORE
# ============================================================

def calculate_resource_score(environment):
    """
    Estimate the current habitat/resource support level.

    Uses existing EnvironmentalObservation fields.

    Returns:
        0.0 - 1.0

    IMPORTANT:
    This is a management estimate based on recorded
    environmental indicators. It is not a laboratory-derived
    carrying-capacity measurement.
    """

    if environment is None:
        return None

    resource_values = []

    # --------------------------------------------------------
    # Vegetation
    # --------------------------------------------------------

    vegetation_cover = resource_value(
        getattr(
            environment,
            "vegetation_cover",
            None
        )
    )

    vegetation_density = resource_value(
        getattr(
            environment,
            "vegetation_density",
            None
        )
    )

    grass_availability = resource_value(
        getattr(
            environment,
            "grass_availability",
            None
        )
    )

    tree_density = resource_value(
        getattr(
            environment,
            "tree_density",
            None
        )
    )

    # --------------------------------------------------------
    # Water
    # --------------------------------------------------------

    water_quality = resource_value(
        getattr(
            environment,
            "water_quality",
            None
        )
    )

    # --------------------------------------------------------
    # Soil
    # --------------------------------------------------------

    soil_quality = resource_value(
        getattr(
            environment,
            "soil_quality",
            None
        )
    )

    # --------------------------------------------------------
    # Add available measurements
    # --------------------------------------------------------

    indicators = [
        vegetation_cover,
        vegetation_density,
        grass_availability,
        tree_density,
        water_quality,
        soil_quality
    ]

    for value in indicators:

        if value is not None:
            resource_values.append(
                float(value)
            )

    if not resource_values:
        return None

    return float(
        np.mean(resource_values)
    )


# ============================================================
# RESOURCE DESCRIPTION
# ============================================================

def resource_description(score):

    if score is None:
        return "Environmental resource data is not available."

    percentage = score * 100

    if percentage >= 80:
        return (
            "Environmental resource indicators are strong."
        )

    if percentage >= 60:
        return (
            "Environmental resource indicators are moderate "
            "to good."
        )

    if percentage >= 40:
        return (
            "Environmental resource indicators are moderate "
            "and may limit population growth."
        )

    return (
        "Environmental resource indicators are low and may "
        "place pressure on population sustainability."
    )


# ============================================================
# LOAD ENVIRONMENTAL INFORMATION
# ============================================================

def get_environment_for_observation(obs):

    if not obs:
        return None, None

    environmental = None
    site = None

    # --------------------------------------------------------
    # Environmental observation
    # --------------------------------------------------------

    if getattr(
        obs,
        "environmental_observation_id",
        None
    ):

        environmental = (
            EnvironmentalObservation.query
            .filter_by(
                id=obs.environmental_observation_id
            )
            .first()
        )

    # --------------------------------------------------------
    # Monitoring site
    # --------------------------------------------------------

    if environmental:

        site_id = getattr(
            environmental,
            "monitoring_site_id",
            None
        )

        if site_id:

            site = (
                MonitoringSite.query
                .filter_by(id=site_id)
                .first()
            )

    return environmental, site


# ============================================================
# GET AREA
# ============================================================

def get_observation_area(environment, site):

    area = None

    if environment is not None:

        area = safe_float(
            getattr(
                environment,
                "area_hectares",
                None
            )
        )

    if (
        area is None
        and
        site is not None
    ):

        area = safe_float(
            getattr(
                site,
                "area_hectares",
                None
            )
        )

    if area is not None and area > 0:
        return area

    return None


# ============================================================
# CARRYING CAPACITY ESTIMATE
# ============================================================

def calculate_species_capacity(
    historical_records,
    latest_environment,
    latest_site
):
    """
    Estimate habitat/resource-supported population.

    The calculation is based on:

        historical population density
        × current park area
        × current resource condition

    Historical density is calculated from approved
    observations that have area information.

    The 90th percentile is used instead of the absolute
    maximum so one unusual observation does not define
    the entire capacity.
    """

    densities = []

    # --------------------------------------------------------
    # Historical population densities
    # --------------------------------------------------------

    for record in historical_records:

        population = safe_float(
            record.get("population")
        )

        area = safe_float(
            record.get("area_hectares")
        )

        if (
            population is None
            or
            area is None
            or
            area <= 0
        ):
            continue

        density = (
            population / area
        )

        if np.isfinite(density):
            densities.append(
                density
            )

    # --------------------------------------------------------
    # Current area
    # --------------------------------------------------------

    current_area = get_observation_area(
        latest_environment,
        latest_site
    )

    if (
        current_area is None
        or
        current_area <= 0
    ):

        return {
            "available": False,
            "reason": (
                "A valid monitoring area was not available."
            )
        }

    # --------------------------------------------------------
    # Need historical density
    # --------------------------------------------------------

    if not densities:

        return {
            "available": False,
            "reason": (
                "There is not enough area information to "
                "estimate habitat capacity."
            ),
            "area_hectares": round(
                current_area,
                2
            )
        }

    # --------------------------------------------------------
    # Robust historical density
    # --------------------------------------------------------

    baseline_density = float(
        np.percentile(
            densities,
            90
        )
    )

    # --------------------------------------------------------
    # Resource condition
    # --------------------------------------------------------

    resource_score = calculate_resource_score(
        latest_environment
    )

    if resource_score is None:

        return {
            "available": False,
            "reason": (
                "Area is available, but current resource "
                "indicators are not available."
            ),
            "area_hectares": round(
                current_area,
                2
            )
        }

    # --------------------------------------------------------
    # Resource adjustment
    #
    # 0.5 means the habitat is already capable of supporting
    # the historical density at 50% resource availability.
    #
    # 1.0 means full support.
    #
    # This prevents the resource score from reducing the
    # capacity to zero.
    # --------------------------------------------------------

    resource_factor = (
        0.50
        +
        (0.50 * resource_score)
    )

    estimated_capacity = (
        baseline_density
        *
        current_area
        *
        resource_factor
    )

    estimated_capacity = max(
        0,
        estimated_capacity
    )

    return {

        "available": True,

        "area_hectares": round(
            current_area,
            2
        ),

        "historical_density_90th": round(
            baseline_density,
            4
        ),

        "resource_score": round(
            resource_score * 100,
            1
        ),

        "resource_factor": round(
            resource_factor,
            3
        ),

        "estimated_capacity": int(
            round(
                estimated_capacity
            )
        ),

        "resource_description":
            resource_description(
                resource_score
            )
    }


# ============================================================
# CAPACITY STATUS
# ============================================================

def get_capacity_status(
    population,
    capacity
):

    if capacity is None:
        return {
            "status": "Unknown",
            "message": (
                "Carrying capacity cannot be estimated "
                "from the available environmental data."
            )
        }

    population = float(population)
    capacity = float(capacity)

    if capacity <= 0:

        return {
            "status": "Insufficient Support",
            "message": (
                "The estimated habitat support capacity "
                "is currently very low."
            )
        }

    utilisation = (
        population / capacity
    ) * 100

    if utilisation >= 100:

        return {
            "status": "Above Estimated Capacity",
            "utilisation": round(
                utilisation,
                1
            ),
            "message": (
                "The population is at or above the estimated "
                "habitat/resource support capacity."
            )
        }

    if utilisation >= 80:

        return {
            "status": "Near Estimated Capacity",
            "utilisation": round(
                utilisation,
                1
            ),
            "message": (
                "The population is approaching the estimated "
                "habitat/resource support capacity."
            )
        }

    return {
        "status": "Within Estimated Capacity",
        "utilisation": round(
            utilisation,
            1
        ),
        "message": (
            "The population is below the estimated "
            "habitat/resource support capacity."
        )
    }


# ============================================================
# FORECAST REASON
# ============================================================

# ============================================================
# FORECAST CONFIGURATION
# ============================================================

QUARTERS_PER_YEAR = 4

# Maximum change allowed in one 3-month period.
# These limits prevent unrealistic exponential forecasts.
MAX_QUARTERLY_GROWTH = 0.20
MAX_QUARTERLY_DECLINE = -0.20


# ============================================================
# SAFE NUMBER CONVERSION
# ============================================================

def safe_float(value):
    """
    Safely convert a value to float.

    Returns None when conversion is not possible.
    """

    if value is None:
        return None

    try:

        value = float(value)

        if not np.isfinite(value):
            return None

        return value

    except (TypeError, ValueError):

        return None


# ============================================================
# SAFE INTEGER CONVERSION
# ============================================================

def safe_int(value, default=0):
    """
    Convert value to a whole number safely.
    """

    try:

        value = float(value)

        if not np.isfinite(value):
            return default

        return int(round(value))

    except (TypeError, ValueError):

        return default


# ============================================================
# ENVIRONMENT VALUE NORMALIZATION
# ============================================================

def normalize_environment_value(value, minimum=0, maximum=100):
    """
    Convert an environmental numeric value into a 0-100 scale.

    Values already between 0 and 100 are treated as percentages/
    scores.

    Values outside this range are clipped.
    """

    value = safe_float(value)

    if value is None:
        return None

    value = np.clip(
        value,
        minimum,
        maximum
    )

    return float(value)


# ============================================================
# GET LATEST ENVIRONMENTAL OBSERVATION
# ============================================================

def get_latest_environment_for_species_observation(obs):
    """
    Find the environmental observation most relevant to the
    species observation.

    Matching priority:

    1. Same location + closest date
    2. Same location
    3. Closest environmental observation

    This makes the capacity calculation more robust when
    monitoring-site relationships are not directly available.
    """

    try:

        observation_date = pd.to_datetime(
            obs.observation_date,
            errors="coerce"
        )

        if pd.isna(observation_date):
            observation_date = None

    except Exception:

        observation_date = None

    # --------------------------------------------------------
    # Determine species/location
    # --------------------------------------------------------

    location = None

    try:

        if obs.species:

            location = (
                getattr(
                    obs.species,
                    "location",
                    None
                )
            )

    except Exception:

        location = None

    # --------------------------------------------------------
    # Get environmental observations
    # --------------------------------------------------------

    try:

        environmental_records = (
            EnvironmentalObservation.query
            .order_by(
                EnvironmentalObservation.observation_date.desc()
            )
            .all()
        )

    except Exception:

        return None

    if not environmental_records:

        return None

    # --------------------------------------------------------
    # Same location + nearest date
    # --------------------------------------------------------

    if location:

        location_matches = []

        for env in environmental_records:

            env_location = getattr(
                env,
                "location",
                None
            )

            if not env_location:
                continue

            if str(env_location).strip().lower() != str(
                location
            ).strip().lower():

                continue

            env_date = pd.to_datetime(
                getattr(
                    env,
                    "observation_date",
                    None
                ),
                errors="coerce"
            )

            if pd.isna(env_date):

                continue

            if observation_date is not None:

                difference = abs(
                    (
                        env_date
                        -
                        observation_date
                    ).total_seconds()
                )

            else:

                difference = 0

            location_matches.append(
                (
                    difference,
                    env
                )
            )

        if location_matches:

            location_matches.sort(
                key=lambda x: x[0]
            )

            return location_matches[0][1]

    # --------------------------------------------------------
    # Fallback: nearest date
    # --------------------------------------------------------

    date_matches = []

    for env in environmental_records:

        env_date = pd.to_datetime(
            getattr(
                env,
                "observation_date",
                None
            ),
            errors="coerce"
        )

        if pd.isna(env_date):
            continue

        if observation_date is not None:

            difference = abs(
                (
                    env_date
                    -
                    observation_date
                ).total_seconds()
            )

        else:

            difference = 0

        date_matches.append(
            (
                difference,
                env
            )
        )

    if date_matches:

        date_matches.sort(
            key=lambda x: x[0]
        )

        return date_matches[0][1]

    return None


# ============================================================
# GET AREA FROM ENVIRONMENTAL RECORD
# ============================================================

def get_environment_area(environment):
    """
    Obtain monitoring area in hectares.

    EnvironmentalObservation may have area_hectares directly.
    """

    if environment is None:

        return None

    area = safe_float(
        getattr(
            environment,
            "area_hectares",
            None
        )
    )

    if area is not None and area > 0:

        return area

    return None


# ============================================================
# RESOURCE SCORE
# ============================================================

def calculate_resource_score(environment):
    """
    Estimate a transparent 0-100 environmental resource score.

    Available fields are used when present:

        vegetation_cover
        vegetation_density
        grass_availability
        tree_density
        water_quality
        soil_quality
        water_ph
        water_turbidity
        soil_ph

    Missing fields are simply excluded.

    This is a heuristic support score, not a scientifically
    validated carrying-capacity measurement.
    """

    if environment is None:

        return None

    scores = []

    # --------------------------------------------------------
    # Vegetation cover
    # --------------------------------------------------------

    vegetation_cover = safe_float(
        getattr(
            environment,
            "vegetation_cover",
            None
        )
    )

    if vegetation_cover is not None:

        scores.append(
            np.clip(
                vegetation_cover,
                0,
                100
            )
        )

    # --------------------------------------------------------
    # Vegetation density
    # --------------------------------------------------------

    vegetation_density = safe_float(
        getattr(
            environment,
            "vegetation_density",
            None
        )
    )

    if vegetation_density is not None:

        scores.append(
            np.clip(
                vegetation_density,
                0,
                100
            )
        )

    # --------------------------------------------------------
    # Grass availability
    # --------------------------------------------------------

    grass_availability = safe_float(
        getattr(
            environment,
            "grass_availability",
            None
        )
    )

    if grass_availability is not None:

        scores.append(
            np.clip(
                grass_availability,
                0,
                100
            )
        )

    # --------------------------------------------------------
    # Tree density
    # --------------------------------------------------------

    tree_density = safe_float(
        getattr(
            environment,
            "tree_density",
            None
        )
    )

    if tree_density is not None:

        scores.append(
            np.clip(
                tree_density,
                0,
                100
            )
        )

    # --------------------------------------------------------
    # Water quality
    # --------------------------------------------------------

    water_quality = safe_float(
        getattr(
            environment,
            "water_quality",
            None
        )
    )

    if water_quality is not None:

        scores.append(
            np.clip(
                water_quality,
                0,
                100
            )
        )

    # --------------------------------------------------------
    # Soil quality
    # --------------------------------------------------------

    soil_quality = safe_float(
        getattr(
            environment,
            "soil_quality",
            None
        )
    )

    if soil_quality is not None:

        scores.append(
            np.clip(
                soil_quality,
                0,
                100
            )
        )

    # --------------------------------------------------------
    # Water pH
    # --------------------------------------------------------

    water_ph = safe_float(
        getattr(
            environment,
            "water_ph",
            None
        )
    )

    if water_ph is not None:

        # Approximate neutral-water suitability.
        #
        # 7 is treated as the center.
        # The score decreases as pH moves away from 7.

        water_ph_score = (
            100
            -
            abs(
                water_ph
                -
                7.0
            )
            * 25
        )

        scores.append(
            np.clip(
                water_ph_score,
                0,
                100
            )
        )

    # --------------------------------------------------------
    # Soil pH
    # --------------------------------------------------------

    soil_ph = safe_float(
        getattr(
            environment,
            "soil_ph",
            None
        )
    )

    if soil_ph is not None:

        # General soil-pH suitability approximation.
        #
        # Around 6.5 is treated as a reasonable midpoint.

        soil_ph_score = (
            100
            -
            abs(
                soil_ph
                -
                6.5
            )
            * 20
        )

        scores.append(
            np.clip(
                soil_ph_score,
                0,
                100
            )
        )

    # --------------------------------------------------------
    # Water turbidity
    # --------------------------------------------------------

    water_turbidity = safe_float(
        getattr(
            environment,
            "water_turbidity",
            None
        )
    )

    if water_turbidity is not None:

        # Lower turbidity is generally preferable.
        #
        # This converts turbidity into a simple 0-100
        # suitability score.

        turbidity_score = (
            100
            -
            np.clip(
                water_turbidity,
                0,
                100
            )
        )

        scores.append(
            turbidity_score
        )

    if not scores:

        return None

    return round(
        float(
            np.mean(scores)
        ),
        2
    )


# ============================================================
# CALCULATE SPECIES CAPACITY
# ============================================================

def calculate_species_capacity(
    historical_data,
    latest_environment=None,
    latest_site=None
):
    """
    Estimate how many individuals the monitored area may
    currently support.

    This is a DATA-SUPPORTED HEURISTIC.

    It is NOT a scientifically validated ecological carrying
    capacity unless the underlying ecological assumptions have
    been independently validated.

    Method:

        historical population density
        =
        population / area

        reference density
        =
        median valid historical density

        resource factor
        =
        0.50 + resource_score / 200

        estimated capacity
        =
        reference density
        *
        current area
        *
        resource factor
    """

    if not historical_data:

        return {
            "available": False,
            "reason": "No historical population data available.",
            "estimated_capacity": None,
            "resource_score": None,
            "area_hectares": None,
            "utilisation_percent": None,
            "status": "Unavailable",
            "message": (
                "There is not enough population history "
                "to estimate habitat support capacity."
            )
        }

    # ========================================================
    # AREA
    # ========================================================

    current_area = get_environment_area(
        latest_environment
    )

    # Fallback to historical areas.
    if current_area is None:

        historical_areas = []

        for row in historical_data:

            area = safe_float(
                row.get(
                    "area_hectares"
                )
            )

            if (
                area is not None
                and
                area > 0
            ):

                historical_areas.append(
                    area
                )

        if historical_areas:

            current_area = float(
                historical_areas[-1]
            )

    if (
        current_area is None
        or
        current_area <= 0
    ):

        return {
            "available": False,
            "reason": (
                "No valid monitoring area was found."
            ),
            "estimated_capacity": None,
            "resource_score": None,
            "area_hectares": None,
            "utilisation_percent": None,
            "status": "Unavailable",
            "message": (
                "The monitoring area is missing or "
                "is not greater than zero hectares."
            )
        }

    # ========================================================
    # HISTORICAL POPULATION DENSITIES
    # ========================================================

    densities = []

    for row in historical_data:

        population = safe_float(
            row.get(
                "population"
            )
        )

        area = safe_float(
            row.get(
                "area_hectares"
            )
        )

        if (
            population is None
            or
            area is None
            or
            area <= 0
        ):

            continue

        density = (
            population
            /
            area
        )

        if np.isfinite(density):

            densities.append(
                density
            )

    if not densities:

        return {
            "available": False,
            "reason": (
                "Population and area could not be combined "
                "to calculate historical density."
            ),
            "estimated_capacity": None,
            "resource_score": None,
            "area_hectares": round(
                current_area,
                2
            ),
            "utilisation_percent": None,
            "status": "Unavailable",
            "message": (
                "The park area is known, but there is not "
                "enough historical area-linked population "
                "data to estimate support capacity."
            )
        }

    # ========================================================
    # REFERENCE DENSITY
    # ========================================================

    reference_density = float(
        np.median(
            densities
        )
    )

    # ========================================================
    # RESOURCE SCORE
    # ========================================================

    resource_score = calculate_resource_score(
        latest_environment
    )

    # If no environmental measurements exist,
    # use a neutral factor rather than inventing a
    # high-resource environment.

    if resource_score is None:

        resource_factor = 0.75

        resource_basis = (
            "No complete environmental resource score "
            "was available; a neutral 75% resource factor "
            "was applied."
        )

    else:

        resource_factor = (
            0.50
            +
            (
                resource_score
                /
                200.0
            )
        )

        resource_factor = float(
            np.clip(
                resource_factor,
                0.50,
                1.00
            )
        )

        resource_basis = (
            "Resource factor was calculated from the "
            "latest available environmental measurements."
        )

    # ========================================================
    # ESTIMATED CAPACITY
    # ========================================================

    estimated_capacity = (
        reference_density
        *
        current_area
        *
        resource_factor
    )

    estimated_capacity = max(
        0,
        estimated_capacity
    )

    estimated_capacity = safe_int(
        estimated_capacity
    )

    return {
        "available": True,

        "area_hectares": round(
            current_area,
            2
        ),

        "resource_score": (
            round(
                resource_score,
                2
            )
            if resource_score is not None
            else None
        ),

        "resource_factor": round(
            resource_factor,
            3
        ),

        "reference_density": round(
            reference_density,
            4
        ),

        "estimated_capacity": int(
            estimated_capacity
        ),

        "density_observations": int(
            len(densities)
        ),

        "status": "Estimated",

        "basis": (
            "Median historical population density "
            "multiplied by the current monitoring area "
            "and an environmental resource factor."
        ),

        "resource_basis": resource_basis,

        "message": (
            "This is an estimated habitat support capacity "
            "based on historical population density, "
            "monitoring area and available environmental "
            "resource measurements. It should not be treated "
            "as a formally validated ecological carrying "
            "capacity."
        )
    }


# ============================================================
# CAPACITY STATUS
# ============================================================

def get_capacity_status(
    population,
    capacity
):
    """
    Determine population utilisation relative to estimated
    support capacity.
    """

    population = safe_float(
        population
    )

    capacity = safe_float(
        capacity
    )

    if (
        population is None
        or
        capacity is None
        or
        capacity <= 0
    ):

        return {
            "status": "Unavailable",
            "utilisation_percent": None,
            "message": (
                "Support capacity could not be estimated."
            )
        }

    utilisation = (
        population
        /
        capacity
    ) * 100

    utilisation = max(
        0,
        utilisation
    )

    if utilisation <= 80:

        status = (
            "Within estimated support capacity"
        )

        message = (
            "The population is below the estimated "
            "support capacity based on the available "
            "area, historical population density and "
            "environmental resource data."
        )

    elif utilisation <= 100:

        status = (
            "Approaching estimated support capacity"
        )

        message = (
            "The population is using a large proportion "
            "of the estimated available support capacity. "
            "Changes in habitat resources should be "
            "monitored closely."
        )

    else:

        status = (
            "Above estimated support capacity"
        )

        message = (
            "The population is above the estimated "
            "support capacity calculated from the available "
            "historical and environmental data."
        )

    return {
        "status": status,

        "utilisation_percent": round(
            utilisation,
            2
        ),

        "message": message
    }


# ============================================================
# BUILD FORECAST REASONS
# ============================================================

def build_forecast_reasons(
    quarterly_growth,
    current_population,
    capacity,
    resource_score,
    forecast_population
):
    """
    Generate human-readable explanations for the forecast.

    These are model-based indicators/associations.
    They should not be interpreted as proof of causation.
    """

    reasons = []

    quarterly_growth = safe_float(
        quarterly_growth
    )

    current_population = safe_float(
        current_population
    )

    capacity = safe_float(
        capacity
    )

    resource_score = safe_float(
        resource_score
    )

    forecast_population = safe_float(
        forecast_population
    )

    # ========================================================
    # TREND REASON
    # ========================================================

    if quarterly_growth is None:

        reasons.append({
            "type": "trend",
            "title": "Population trend",
            "message": (
                "There is not enough valid quarterly change "
                "information to explain the population trend."
            )
        })

    elif quarterly_growth > 0.02:

        reasons.append({
            "type": "trend",
            "title": "Recent population trend",
            "message": (
                "The most recent approved observations show "
                "a positive median quarterly growth rate of "
                f"{quarterly_growth * 100:.1f}%. "
                "The forecast therefore continues a moderated "
                "increasing trend rather than applying the full "
                "historical growth rate."
            )
        })

    elif quarterly_growth < -0.02:

        reasons.append({
            "type": "trend",
            "title": "Recent population trend",
            "message": (
                "The most recent approved observations show "
                "a negative median quarterly growth rate of "
                f"{abs(quarterly_growth * 100):.1f}%. "
                "The forecast therefore continues a moderated "
                "declining trend."
            )
        })

    else:

        reasons.append({
            "type": "trend",
            "title": "Recent population trend",
            "message": (
                "Recent approved observations show only a "
                "small quarterly change. The model therefore "
                "treats the population as broadly stable."
            )
        })

    # ========================================================
    # RESOURCE REASON
    # ========================================================

    if resource_score is None:

        reasons.append({
            "type": "resource",
            "title": "Environmental resources",
            "message": (
                "A complete environmental resource score was "
                "not available. The model therefore uses a "
                "neutral resource assumption rather than "
                "assuming that habitat conditions are either "
                "excellent or poor."
            )
        })

    elif resource_score >= 70:

        reasons.append({
            "type": "resource",
            "title": "Environmental resources",
            "message": (
                f"The latest available environmental data "
                f"produced an estimated resource score of "
                f"{resource_score:.0f}/100. This indicates "
                "relatively strong available resource "
                "conditions in the data used by the model."
            )
        })

    elif resource_score >= 40:

        reasons.append({
            "type": "resource",
            "title": "Environmental resources",
            "message": (
                f"The latest environmental data produced "
                f"a resource score of {resource_score:.0f}/100. "
                "The available habitat resources are therefore "
                "treated as moderate by the model."
            )
        })

    else:

        reasons.append({
            "type": "resource",
            "title": "Environmental resources",
            "message": (
                f"The latest environmental data produced "
                f"a relatively low resource score of "
                f"{resource_score:.0f}/100. Resource "
                "conditions may therefore limit the projected "
                "population in the model."
            )
        })

    # ========================================================
    # CAPACITY REASON
    # ========================================================

    if (
        capacity is None
        or
        capacity <= 0
    ):

        reasons.append({
            "type": "capacity",
            "title": "Habitat support capacity",
            "message": (
                "An estimated support capacity could not "
                "be calculated because sufficient area-linked "
                "population data was not available."
            )
        })

    else:

        utilisation = (
            current_population
            /
            capacity
        ) * 100

        forecast_utilisation = (
            forecast_population
            /
            capacity
        ) * 100

        if forecast_utilisation <= 80:

            message = (
                f"The projected population is about "
                f"{forecast_utilisation:.1f}% of the estimated "
                f"support capacity. Based on the available "
                "data, the forecast remains below the estimated "
                "support limit."
            )

        elif forecast_utilisation <= 100:

            message = (
                f"The projected population is about "
                f"{forecast_utilisation:.1f}% of the estimated "
                f"support capacity. The population is therefore "
                "approaching the estimated support limit."
            )

        else:

            message = (
                f"The projected population is about "
                f"{forecast_utilisation:.1f}% of the estimated "
                f"support capacity. The forecast therefore "
                "exceeds the estimated support level."
            )

        reasons.append({
            "type": "capacity",
            "title": "Habitat support capacity",
            "message": message
        })

    return reasons


# ============================================================
# FORECAST SPECIES POPULATION
# ============================================================
# ============================================================
# FORECAST CONFIGURATION
# ============================================================

QUARTERS_PER_YEAR = 4

# Maximum increase allowed per 3-month period.
MAX_QUARTERLY_GROWTH = 0.15

# Maximum decline allowed per 3-month period.
MAX_QUARTERLY_DECLINE = -0.15

# Recent observations receive more importance.
RECENT_OBSERVATIONS_USED = 4

# Trend damping.
TREND_DAMPING = 0.50
# ============================================================
# SAFE NUMBER FUNCTIONS
# ============================================================

def safe_float(value, default=None):
    """
    Safely convert a value to float.

    Returns default when the value is invalid.
    """

    if value is None:
        return default

    try:

        value = float(value)

        if not math.isfinite(value):
            return default

        return value

    except (
        ValueError,
        TypeError
    ):

        return default


def safe_int(value, default=0):
    """
    Safely convert a value to a whole number.
    """

    value = safe_float(
        value,
        None
    )

    if value is None:
        return default

    return int(
        round(value)
    )


def clean_text(value):
    """
    Safely convert a value to text.
    """

    if value is None:
        return ""

    return str(value).strip()
# ============================================================
# ENVIRONMENTAL DATA
# ============================================================

def get_latest_environment_for_species_observation(obs):
    """
    Find the environmental observation closest to the
    biodiversity observation date.

    This function tries to match using location first.

    It does NOT modify any historical observation.
    """

    if obs is None:
        return None

    observation_date = getattr(
        obs,
        "observation_date",
        None
    )

    if observation_date is None:
        return None

    try:

        observation_date = pd.to_datetime(
            observation_date,
            errors="coerce"
        )

    except Exception:

        return None

    if pd.isna(observation_date):
        return None

    try:

        query = EnvironmentalObservation.query

        # ----------------------------------------------------
        # Try location matching
        # ----------------------------------------------------

        species_location = None

        if getattr(obs, "species", None):

            species_location = clean_text(
                getattr(
                    obs.species,
                    "location",
                    None
                )
            )

        if species_location:

            location_column = getattr(
                EnvironmentalObservation,
                "location",
                None
            )

            if location_column is not None:

                environment = (
                    query
                    .filter(
                        location_column.ilike(
                            species_location
                        )
                    )
                    .order_by(
                        EnvironmentalObservation.observation_date.desc()
                    )
                    .first()
                )

                if environment is not None:

                    return environment

        # ----------------------------------------------------
        # If location cannot be matched, use latest
        # environmental observation.
        # ----------------------------------------------------

        return (
            query
            .order_by(
                EnvironmentalObservation.observation_date.desc()
            )
            .first()
        )

    except Exception:

        return None


def get_environment_area(environment):
    """
    Extract monitoring area from the environmental record.
    """

    if environment is None:
        return None

    possible_fields = [
        "area_hectares",
        "water_body_hectares"
    ]

    for field in possible_fields:

        value = safe_float(
            getattr(
                environment,
                field,
                None
            ),
            None
        )

        if value is not None and value > 0:

            return value

    # Some systems store area through a monitoring site.
    monitoring_site = getattr(
        environment,
        "monitoring_site",
        None
    )

    if monitoring_site is not None:

        value = safe_float(
            getattr(
                monitoring_site,
                "area_hectares",
                None
            ),
            None
        )

        if value is not None and value > 0:

            return value

    return None
# ============================================================
# RESOURCE SCORE
# ============================================================

def calculate_resource_score(environment):
    """
    Estimate habitat/resource availability from available
    environmental indicators.

    Score:
        0   = very poor
        100 = very good

    This is a transparent heuristic, NOT a scientifically
    validated carrying-capacity measurement.
    """

    if environment is None:

        return {
            "score": None,
            "basis": []
        }

    values = []
    basis = []

    # --------------------------------------------------------
    # VEGETATION COVER
    # --------------------------------------------------------

    vegetation_cover = safe_float(
        getattr(
            environment,
            "vegetation_cover",
            None
        ),
        None
    )

    if vegetation_cover is not None:

        vegetation_cover = np.clip(
            vegetation_cover,
            0,
            100
        )

        values.append(
            float(vegetation_cover)
        )

        basis.append(
            f"vegetation cover: {vegetation_cover:.1f}%"
        )

    # --------------------------------------------------------
    # VEGETATION DENSITY
    # --------------------------------------------------------

    vegetation_density = safe_float(
        getattr(
            environment,
            "vegetation_density",
            None
        ),
        None
    )

    if vegetation_density is not None:

        vegetation_density = np.clip(
            vegetation_density,
            0,
            100
        )

        values.append(
            float(vegetation_density)
        )

        basis.append(
            f"vegetation density: {vegetation_density:.1f}%"
        )

    # --------------------------------------------------------
    # GRASS AVAILABILITY
    # --------------------------------------------------------

    grass = safe_float(
        getattr(
            environment,
            "grass_availability",
            None
        ),
        None
    )

    if grass is not None:

        grass = np.clip(
            grass,
            0,
            100
        )

        values.append(
            float(grass)
        )

        basis.append(
            f"grass availability: {grass:.1f}%"
        )

    # --------------------------------------------------------
    # TREE DENSITY
    # --------------------------------------------------------

    tree_density = safe_float(
        getattr(
            environment,
            "tree_density",
            None
        ),
        None
    )

    if tree_density is not None:

        tree_density = np.clip(
            tree_density,
            0,
            100
        )

        values.append(
            float(tree_density)
        )

        basis.append(
            f"tree density: {tree_density:.1f}%"
        )

    # --------------------------------------------------------
    # SOIL pH
    # --------------------------------------------------------

    soil_ph = safe_float(
        getattr(
            environment,
            "soil_ph",
            None
        ),
        None
    )

    if soil_ph is not None:

        # Broad ecological normalization.
        # pH 5.0 - 8.0 treated as progressively usable.
        soil_score = (
            100
            -
            abs(
                soil_ph - 6.5
            ) * 25
        )

        soil_score = float(
            np.clip(
                soil_score,
                0,
                100
            )
        )

        values.append(
            soil_score
        )

        basis.append(
            f"soil pH: {soil_ph:.2f}"
        )

    # --------------------------------------------------------
    # WATER pH
    # --------------------------------------------------------

    water_ph = safe_float(
        getattr(
            environment,
            "water_ph",
            None
        ),
        None
    )

    if water_ph is not None:

        water_score = (
            100
            -
            abs(
                water_ph - 7.0
            ) * 20
        )

        water_score = float(
            np.clip(
                water_score,
                0,
                100
            )
        )

        values.append(
            water_score
        )

        basis.append(
            f"water pH: {water_ph:.2f}"
        )

    # --------------------------------------------------------
    # WATER TURBIDITY
    # --------------------------------------------------------

    turbidity = safe_float(
        getattr(
            environment,
            "water_turbidity",
            None
        ),
        None
    )

    if turbidity is not None:

        # Lower turbidity is generally better.
        turbidity_score = (
            100
            -
            np.clip(
                turbidity,
                0,
                100
            )
        )

        values.append(
            float(turbidity_score)
        )

        basis.append(
            f"water turbidity: {turbidity:.2f}"
        )

    # --------------------------------------------------------
    # NO DATA
    # --------------------------------------------------------

    if not values:

        return {
            "score": None,
            "basis": []
        }

    score = float(
        np.mean(values)
    )

    return {
        "score": round(
            score,
            2
        ),
        "basis": basis
    }
    # ============================================================
# SPECIES SUPPORT CAPACITY
# ============================================================

def calculate_species_capacity(
    historical_data,
    latest_environment
):
    """
    Estimate species support capacity.

    This is a data-supported heuristic.

    It uses:

        population
        area
        population density
        environmental resource score

    It must NOT be interpreted as a scientifically validated
    ecological carrying-capacity model.
    """

    if not historical_data:

        return {
            "available": False,
            "estimated_capacity": None,
            "area_hectares": None,
            "resource_score": None,
            "resource_factor": None,
            "reference_density": None,
            "message": (
                "No historical population data is available "
                "for capacity estimation."
            ),
            "basis": None,
            "resource_basis": []
        }

    # --------------------------------------------------------
    # AREA
    # --------------------------------------------------------

    latest_area = get_environment_area(
        latest_environment
    )

    # Also check historical area values.
    if latest_area is None:

        valid_areas = []

        for item in historical_data:

            area = safe_float(
                item.get(
                    "area_hectares"
                ),
                None
            )

            if (
                area is not None
                and
                area > 0
            ):

                valid_areas.append(
                    area
                )

        if valid_areas:

            latest_area = valid_areas[-1]

    # --------------------------------------------------------
    # RESOURCE SCORE
    # --------------------------------------------------------

    resource_result = calculate_resource_score(
        latest_environment
    )

    resource_score = resource_result.get(
        "score"
    )

    resource_basis = resource_result.get(
        "basis",
        []
    )

    # --------------------------------------------------------
    # REQUIRE AREA
    # --------------------------------------------------------

    if (
        latest_area is None
        or
        latest_area <= 0
    ):

        return {
            "available": False,
            "estimated_capacity": None,
            "area_hectares": None,
            "resource_score": resource_score,
            "resource_factor": None,
            "reference_density": None,
            "message": (
                "A valid monitoring area is required "
                "to estimate habitat support capacity."
            ),
            "basis": (
                "No valid area was available."
            ),
            "resource_basis": resource_basis
        }

    # --------------------------------------------------------
    # HISTORICAL DENSITY
    # --------------------------------------------------------

    densities = []

    for item in historical_data:

        population = safe_float(
            item.get(
                "population"
            ),
            None
        )

        area = safe_float(
            item.get(
                "area_hectares"
            ),
            None
        )

        if (
            population is None
            or
            area is None
            or
            area <= 0
        ):

            continue

        density = (
            population
            /
            area
        )

        if np.isfinite(density):

            densities.append(
                float(density)
            )

    # --------------------------------------------------------
    # REQUIRE DENSITY
    # --------------------------------------------------------

    if not densities:

        return {
            "available": False,
            "estimated_capacity": None,
            "area_hectares": round(
                latest_area,
                2
            ),
            "resource_score": resource_score,
            "resource_factor": None,
            "reference_density": None,
            "message": (
                "There is not enough historical area and "
                "population information to estimate capacity."
            ),
            "basis": (
                "Historical population density could not "
                "be calculated."
            ),
            "resource_basis": resource_basis
        }

    # --------------------------------------------------------
    # REFERENCE DENSITY
    # --------------------------------------------------------
    #
    # Median is used instead of maximum so one unusual
    # observation does not define the entire capacity.
    # --------------------------------------------------------

    reference_density = float(
        np.median(
            densities
        )
    )

    # --------------------------------------------------------
    # RESOURCE FACTOR
    # --------------------------------------------------------

    if resource_score is None:

        # No environmental information.
        resource_factor = 1.0

    else:

        # Converts:
        #
        # 0 resource score   -> 0.50
        # 50 resource score  -> 0.75
        # 100 resource score -> 1.00
        #
        resource_factor = (
            0.50
            +
            (
                resource_score
                /
                200.0
            )
        )

    # --------------------------------------------------------
    # ESTIMATED CAPACITY
    # --------------------------------------------------------

    estimated_capacity = (
        reference_density
        *
        latest_area
        *
        resource_factor
    )

    estimated_capacity = max(
        0,
        safe_int(
            estimated_capacity
        )
    )

    # Never return zero when valid positive historical
    # density exists.
    if (
        estimated_capacity == 0
        and
        reference_density > 0
    ):

        estimated_capacity = 1

    # --------------------------------------------------------
    # BASIS
    # --------------------------------------------------------

    basis = (
        "Estimated from the median historical "
        "population density, current monitoring area, "
        "and available environmental resource indicators."
    )

    return {

        "available": True,

        "area_hectares": round(
            latest_area,
            2
        ),

        "resource_score": (
            round(
                resource_score,
                2
            )
            if resource_score is not None
            else None
        ),

        "resource_factor": round(
            resource_factor,
            3
        ),

        "reference_density": round(
            reference_density,
            4
        ),

        "estimated_capacity": int(
            estimated_capacity
        ),

        "message": (
            "The estimated support capacity is based on "
            "historical population density and available "
            "habitat/resource information."
        ),

        "basis": basis,

        "resource_basis": resource_basis
    }# ============================================================
# CAPACITY STATUS
# ============================================================

def get_capacity_status(
    population,
    capacity
):
    """
    Determine whether the population is within the
    estimated support capacity.
    """

    population = safe_float(
        population,
        0
    )

    capacity = safe_float(
        capacity,
        None
    )

    if (
        capacity is None
        or
        capacity <= 0
    ):

        return {

            "status":
                "Unavailable",

            "utilisation_percent":
                None,

            "message":
                (
                    "Estimated habitat support capacity "
                    "is unavailable because sufficient "
                    "area/resource data was not available."
                )
        }

    utilisation = (
        population
        /
        capacity
    ) * 100

    utilisation = max(
        0,
        utilisation
    )

    if utilisation <= 80:

        status = (
            "Within estimated support capacity"
        )

        message = (
            "The population is currently within the "
            "estimated habitat support capacity based "
            "on the available area, historical density "
            "and resource information."
        )

    elif utilisation <= 100:

        status = (
            "Approaching estimated support capacity"
        )

        message = (
            "The population is approaching the estimated "
            "habitat support capacity. Continued monitoring "
            "of resources and population change is recommended."
        )

    else:

        status = (
            "Above estimated support capacity"
        )

        message = (
            "The population is above the estimated support "
            "capacity calculated from the available habitat "
            "and resource information."
        )

    return {

        "status":
            status,

        "utilisation_percent":
            round(
                utilisation,
                2
            ),

        "message":
            message
    }
    # ============================================================
# FORECAST REASONS
# ============================================================
# ============================================================
# FORECAST HELPERS
# ============================================================

def safe_float(value, default=None):
    """
    Safely convert a value to float.
    """

    try:

        if value is None:
            return default

        value = float(value)

        if not np.isfinite(value):
            return default

        return value

    except (
        ValueError,
        TypeError
    ):

        return default


def safe_int(value, default=0):
    """
    Safely convert a value to a whole integer.
    """

    try:

        if value is None:
            return default

        value = float(value)

        if not np.isfinite(value):
            return default

        return int(
            round(value)
        )

    except (
        ValueError,
        TypeError
    ):

        return default


def clean_text(value):
    """
    Safely clean text.
    """

    if value is None:
        return ""

    return str(value).strip()


# ============================================================
# FORECAST CONSTANTS
# ============================================================

QUARTERS_PER_YEAR = 4

RECENT_OBSERVATIONS_USED = 4

TREND_DAMPING = 0.50

# Maximum quarterly growth allowed.
MAX_QUARTERLY_GROWTH = 0.20

# Maximum quarterly decline allowed.
MAX_QUARTERLY_DECLINE = -0.20


# ============================================================
# CAPACITY STATUS
# ============================================================

def get_capacity_status(
    population,
    capacity
):
    """
    Determine how the population compares with the
    estimated habitat/resource support capacity.

    IMPORTANT:
    This is an estimated support-capacity assessment,
    not a scientifically established carrying capacity.
    """

    population = safe_float(
        population,
        0
    )

    capacity = safe_float(
        capacity,
        None
    )

    if (
        capacity is None
        or
        capacity <= 0
    ):

        return {

            "available": False,

            "status":
                "Unavailable",

            "utilisation_percent":
                None,

            "message":
                (
                    "Estimated habitat support capacity "
                    "is not available."
                )
        }

    utilisation_percent = (
        population
        /
        capacity
    ) * 100

    if utilisation_percent <= 80:

        status = (
            "Within estimated support capacity"
        )

        message = (
            f"The population is using approximately "
            f"{utilisation_percent:.1f}% of the estimated "
            f"support capacity."
        )

    elif utilisation_percent <= 100:

        status = (
            "Approaching estimated support capacity"
        )

        message = (
            f"The population is using approximately "
            f"{utilisation_percent:.1f}% of the estimated "
            f"support capacity."
        )

    else:

        status = (
            "Above estimated support capacity"
        )

        message = (
            f"The population is approximately "
            f"{utilisation_percent:.1f}% of the estimated "
            f"support capacity."
        )

    return {

        "available":
            True,

        "status":
            status,

        "utilisation_percent":
            round(
                utilisation_percent,
                2
            ),

        "message":
            message
    }



# ============================================================
# SPECIES POPULATION FORECAST
# ============================================================
def forecast_species_population(
    species_name,
    years_ahead=3
):
    """
    Forecast species population from the observed population trend.

    Forecast rules:

    - Approved observations only.
    - Historical observations are never modified.
    - Observations are sorted chronologically.
    - Recent observations have greater importance when determining
      the historical direction.
    - Increasing population -> +5% every forecast quarter.
    - Declining population -> -5% every forecast quarter.
    - Stable population -> no forecast change.
    - The 5% change is compounded from one forecast period to the next.
    - Capacity NEVER reverses the observed direction.
    - Population output is always a whole number.
    - Forecast reasons are removed.
    """

    # ========================================================
    # VALIDATION
    # ========================================================

    species_name = clean_text(species_name)

    if not species_name:
        return {
            "success": False,
            "error": "Please select a species."
        }

    try:
        years_ahead = int(years_ahead)

    except (ValueError, TypeError):
        return {
            "success": False,
            "error": "Forecast period must be a valid number."
        }

    if years_ahead < 1 or years_ahead > 10:
        return {
            "success": False,
            "error": "Forecast period must be between 1 and 10 years."
        }

    # ========================================================
    # APPROVED OBSERVATIONS ONLY
    # ========================================================

    observations = (
        Observation.query
        .filter(
            Observation.status == "Approved"
        )
        .join(
            Observation.species
        )
        .filter(
            Species.specie_Common_Name.ilike(
                species_name
            )
        )
        .order_by(
            Observation.observation_date.asc()
        )
        .all()
    )

    if not observations:
        return {
            "success": False,
            "error": (
                f"No approved observations were found "
                f"for {species_name}."
            )
        }

    # ========================================================
    # BUILD HISTORICAL DATA
    # ========================================================

    historical_data = []

    latest_environment = None

    for obs in observations:

        population = safe_float(
            obs.population_count,
            None
        )

        if (
            population is None
            or
            population < 0
        ):
            continue

        observation_date = pd.to_datetime(
            obs.observation_date,
            errors="coerce"
        )

        if pd.isna(observation_date):
            continue

        # ----------------------------------------------------
        # ENVIRONMENT
        # ----------------------------------------------------

        try:

            environment = (
                get_latest_environment_for_species_observation(
                    obs
                )
            )

        except Exception:

            environment = None

        # ----------------------------------------------------
        # AREA
        # ----------------------------------------------------

        try:

            area = get_environment_area(
                environment
            )

        except Exception:

            area = None

        area = safe_float(
            area,
            None
        )

        historical_data.append({

            "observation_id":
                obs.id,

            "date":
                observation_date,

            "population":
                float(population),

            "area_hectares":
                area
        })

        if environment is not None:
            latest_environment = environment

    # ========================================================
    # VALID HISTORY
    # ========================================================

    if not historical_data:

        return {
            "success": False,
            "error": (
                f"No valid approved population history "
                f"is available for {species_name}."
            )
        }

    df = pd.DataFrame(
        historical_data
    )

    df = (
        df
        .sort_values("date")
        .reset_index(drop=True)
    )

    observation_count = len(df)

    # ========================================================
    # MINIMUM DATA
    # ========================================================

    if observation_count < 3:

        return {
            "success": False,
            "error": (
                f"At least 3 approved observations are "
                f"required to forecast {species_name}. "
                f"Only {observation_count} valid approved "
                f"observation(s) are available."
            ),
            "species": species_name,
            "approved_observations": observation_count,
            "minimum_required": 3
        }

    # ========================================================
    # POPULATION ARRAY
    # ========================================================

    populations = (
        pd.to_numeric(
            df["population"],
            errors="coerce"
        )
        .fillna(0)
        .to_numpy(
            dtype=float
        )
    )

    populations = np.maximum(
        populations,
        0
    )

    # ========================================================
    # RECENT TREND
    # ========================================================
    #
    # We use the most recent observations to determine whether
    # the species is increasing, declining or stable.
    #
    # IMPORTANT:
    #
    # This section determines ONLY the direction.
    #
    # It does NOT determine the forecast percentage.
    #
    # The forecast percentage is always exactly 5%.
    # ========================================================

    recent_population_count = min(
        len(populations),
        RECENT_OBSERVATIONS_USED + 1
    )

    recent_populations = (
        populations[
            -recent_population_count:
        ]
    )

    # ========================================================
    # OBSERVED GROWTH RATES
    # ========================================================

    quarterly_growth_rates = []

    for previous, current in zip(
        recent_populations[:-1],
        recent_populations[1:]
    ):

        if previous > 0:

            rate = (
                current - previous
            ) / previous

            if np.isfinite(rate):

                quarterly_growth_rates.append(
                    float(rate)
                )

    # ========================================================
    # RECENT MEDIAN GROWTH
    # ========================================================

    if quarterly_growth_rates:

        recent_growth_rate = float(
            np.median(
                quarterly_growth_rates
            )
        )

    else:

        recent_growth_rate = 0.0

    # ========================================================
    # RECENT WINDOW CHANGE
    # ========================================================

    if len(recent_populations) >= 2:

        first_recent_population = float(
            recent_populations[0]
        )

        last_recent_population = float(
            recent_populations[-1]
        )

        if first_recent_population > 0:

            recent_window_change = (
                last_recent_population
                -
                first_recent_population
            ) / first_recent_population

        else:

            # If the starting population is zero,
            # percentage growth cannot be calculated.
            #
            # Use the actual direction instead.

            if (
                last_recent_population
                >
                first_recent_population
            ):

                recent_window_change = 1.0

            elif (
                last_recent_population
                <
                first_recent_population
            ):

                recent_window_change = -1.0

            else:

                recent_window_change = 0.0

    else:

        recent_window_change = 0.0

    # ========================================================
    # TREND DIRECTION
    # ========================================================
    #
    # We deliberately use a small threshold here.
    #
    # The old 2% threshold could incorrectly classify a real
    # increase/decrease as Stable.
    # ========================================================

    TREND_THRESHOLD = 0.005

    # --------------------------------------------------------
    # STRONGLY INCREASING
    # --------------------------------------------------------

    if (
        recent_growth_rate > TREND_THRESHOLD
        and
        recent_window_change > 0
    ):

        trend_direction = "Increasing"

    # --------------------------------------------------------
    # STRONGLY DECLINING
    # --------------------------------------------------------

    elif (
        recent_growth_rate < -TREND_THRESHOLD
        and
        recent_window_change < 0
    ):

        trend_direction = "Declining"

    # --------------------------------------------------------
    # WINDOW CLEARLY INCREASING
    # --------------------------------------------------------

    elif (
        recent_window_change > TREND_THRESHOLD
    ):

        trend_direction = "Increasing"

    # --------------------------------------------------------
    # WINDOW CLEARLY DECLINING
    # --------------------------------------------------------

    elif (
        recent_window_change < -TREND_THRESHOLD
    ):

        trend_direction = "Declining"

    # --------------------------------------------------------
    # OTHERWISE STABLE
    # --------------------------------------------------------

    else:

        trend_direction = "Stable"

    # ========================================================
    # FIXED FORECAST RATE
    # ========================================================
    #
    # THIS IS THE IMPORTANT CHANGE.
    #
    # We are no longer using:
    #
    #     recent_growth_rate * TREND_DAMPING
    #
    # because that was causing the forecast percentage to vary
    # and interact incorrectly with capacity.
    #
    # Instead:
    #
    # Increasing = +5%
    # Declining  = -5%
    # Stable     = 0%
    # ========================================================

    FORECAST_CHANGE_RATE = 0.05

    if trend_direction == "Increasing":

        trend_rate = (
            FORECAST_CHANGE_RATE
        )

    elif trend_direction == "Declining":

        trend_rate = (
            -FORECAST_CHANGE_RATE
        )

    else:

        trend_rate = 0.0

    # ========================================================
    # CAPACITY
    # ========================================================
    #
    # Capacity is still calculated and returned for reporting.
    #
    # IMPORTANT:
    #
    # Capacity is NOT allowed to change:
    #
    # Increasing -> Declining
    #
    # or:
    #
    # Declining -> Increasing
    #
    # Therefore capacity will NOT be used to modify the 5%
    # forecast rate.
    # ========================================================

    try:

        capacity_result = (
            calculate_species_capacity(
                historical_data,
                latest_environment
            )
        )

    except Exception:

        app.logger.exception(
            "Capacity calculation failed for %s",
            species_name
        )

        capacity_result = {

            "available":
                False,

            "estimated_capacity":
                None,

            "resource_score":
                None,

            "area_hectares":
                None,

            "reference_density":
                None,

            "resource_factor":
                None,

            "basis":
                None,

            "resource_basis":
                [],

            "message":
                "Capacity calculation was unavailable."
        }

    if not isinstance(
        capacity_result,
        dict
    ):

        capacity_result = {

            "available":
                False,

            "estimated_capacity":
                None,

            "resource_score":
                None,

            "message":
                "Capacity calculation was unavailable."
        }

    capacity = safe_float(
        capacity_result.get(
            "estimated_capacity"
        ),
        None
    )

    # ========================================================
    # CURRENT POPULATION
    # ========================================================

    current_population = max(
        0.0,
        float(
            populations[-1]
        )
    )

    current_population_int = int(
        round(
            current_population
        )
    )

    # ========================================================
    # CURRENT CAPACITY STATUS
    # ========================================================

    current_capacity_status = (
        get_capacity_status(
            current_population,
            capacity
        )
    )

    # ========================================================
    # FORECAST QUARTERS
    # ========================================================

    total_quarters = (
        years_ahead
        *
        QUARTERS_PER_YEAR
    )

    forecast = []

    # ========================================================
    # IMPORTANT
    # ========================================================
    #
    # Keep this as FLOAT internally.
    #
    # Do NOT round this after every forecast.
    #
    # Example:
    #
    # 100
    # 105
    # 110.25
    # 115.7625
    #
    # The final displayed value is rounded to a whole number,
    # but the internal calculation retains precision.
    # ========================================================

    previous_population = (
        current_population
    )

    last_date = (
        df["date"].iloc[-1]
    )

    # ========================================================
    # FORECAST LOOP
    # ========================================================

    for quarter_number in range(
        1,
        total_quarters + 1
    ):

        # ====================================================
        # APPLY EXACT 5% RULE
        # ====================================================

        if trend_direction == "Increasing":

            effective_rate = 0.05

        elif trend_direction == "Declining":

            effective_rate = -0.05

        else:

            effective_rate = 0.0

        # ====================================================
        # CALCULATE POPULATION
        # ====================================================

        predicted_population = (
            previous_population
            *
            (
                1.0
                +
                effective_rate
            )
        )

        # ====================================================
        # PREVENT NEGATIVE POPULATION
        # ====================================================

        predicted_population = max(
            0.0,
            float(
                predicted_population
            )
        )

        # ====================================================
        # DATE
        # ====================================================

        forecast_date = (
            last_date
            +
            pd.DateOffset(
                months=(
                    3
                    *
                    quarter_number
                )
            )
        )

        # ====================================================
        # CALENDAR QUARTER
        # ====================================================

        calendar_quarter = (
            (
                (
                    last_date.month
                    -
                    1
                )
                //
                3
            )
            +
            quarter_number
        )

        forecast_year = (
            last_date.year
            +
            (
                (
                    calendar_quarter
                    -
                    1
                )
                //
                4
            )
        )

        forecast_quarter = (
            (
                calendar_quarter
                -
                1
            )
            %
            4
        ) + 1

        # ====================================================
        # DISPLAY POPULATION
        # ====================================================
        #
        # No decimal places.
        # ====================================================

        predicted_population_display = int(
            round(
                predicted_population
            )
        )

        previous_population_display = int(
            round(
                previous_population
            )
        )

        # ====================================================
        # QUARTERLY CHANGE
        # ====================================================

        quarterly_change = (
            predicted_population_display
            -
            previous_population_display
        )

        # ====================================================
        # CHANGE FROM CURRENT
        # ====================================================

        change_from_current = (
            predicted_population_display
            -
            current_population_int
        )

        # ====================================================
        # PERCENTAGE CHANGE
        # ====================================================

        if previous_population > 0:

            percentage_change = (
                (
                    predicted_population
                    -
                    previous_population
                )
                /
                previous_population
            ) * 100

        else:

            percentage_change = None

        # ====================================================
        # DIRECTION
        # ====================================================

        if predicted_population > previous_population:

            direction = "Increasing"

        elif predicted_population < previous_population:

            direction = "Declining"

        else:

            direction = "Stable"

        # ====================================================
        # APPEND FORECAST
        # ====================================================

        forecast.append({

            "quarter":
                f"Q{forecast_quarter}",

            "year":
                int(
                    forecast_year
                ),

            "date":
                forecast_date.strftime(
                    "%Y-%m-%d"
                ),

            "period":
                (
                    f"Q{forecast_quarter} "
                    f"{forecast_year}"
                ),

            "predicted_population":
                predicted_population_display,

            "change_from_current":
                int(
                    change_from_current
                ),

            "quarterly_change":
                int(
                    quarterly_change
                ),

            "percentage_change":
                (
                    round(
                        percentage_change,
                        2
                    )
                    if percentage_change is not None
                    else None
                ),

            "direction":
                direction,

            "applied_quarterly_rate":
                round(
                    effective_rate * 100,
                    2
                )
        })

        # ====================================================
        # KEEP FULL PRECISION FOR NEXT QUARTER
        # ====================================================

        previous_population = (
            float(
                predicted_population
            )
        )

    # ========================================================
    # FINAL FORECAST
    # ========================================================

    final_population = int(
        round(
            forecast[-1][
                "predicted_population"
            ]
        )
    )

    # ========================================================
    # FINAL CAPACITY STATUS
    # ========================================================

    final_capacity_status = (
        get_capacity_status(
            final_population,
            capacity
        )
    )

    # ========================================================
    # OVERALL TREND
    # ========================================================

    trend = trend_direction

    # ========================================================
    # FORECAST CHANGE
    # ========================================================

    forecast_change = (
        final_population
        -
        current_population_int
    )

    if current_population_int > 0:

        forecast_change_percent = (
            forecast_change
            /
            current_population_int
        ) * 100

    else:

        forecast_change_percent = None

    # ========================================================
    # HISTORICAL DATA
    # ========================================================

    historical = []

    for _, row in df.iterrows():

        population = int(
            round(
                float(
                    row["population"]
                )
            )
        )

        area = safe_float(
            row["area_hectares"],
            None
        )

        historical.append({

            "date":
                row["date"].strftime(
                    "%Y-%m-%d"
                ),

            "year":
                int(
                    row["date"].year
                ),

            "population":
                population,

            "area_hectares":
                (
                    round(
                        area,
                        2
                    )
                    if area is not None
                    else None
                )
        })

    # ========================================================
    # CAPACITY ASSESSMENT
    # ========================================================

    capacity_assessment = {

        "available":
            bool(
                capacity_result.get(
                    "available",
                    False
                )
            ),

        "area_hectares":
            capacity_result.get(
                "area_hectares"
            ),

        "resource_score":
            capacity_result.get(
                "resource_score"
            ),

        "resource_factor":
            capacity_result.get(
                "resource_factor"
            ),

        "estimated_capacity":
            capacity_result.get(
                "estimated_capacity"
            ),

        "reference_density":
            capacity_result.get(
                "reference_density"
            ),

        "utilisation_percent":
            current_capacity_status.get(
                "utilisation_percent"
            ),

        "status":
            current_capacity_status.get(
                "status",
                "Unavailable"
            ),

        "message":
            current_capacity_status.get(

                "message",

                capacity_result.get(

                    "message",

                    "Capacity assessment unavailable."
                )
            ),

        "basis":
            capacity_result.get(
                "basis"
            ),

        "resource_basis":
            capacity_result.get(
                "resource_basis",
                []
            )
    }

    # ========================================================
    # RETURN
    # ========================================================

    return {

        "success":
            True,

        "species":
            species_name,

        "forecast_years":
            int(
                years_ahead
            ),

        "forecast_quarters":
            int(
                total_quarters
            ),

        "observation_interval":
            "3 months",

        "interval_months":
            3,

        "approved_observations":
            int(
                observation_count
            ),

        "minimum_required":
            3,

        "first_observation_date":
            df["date"].iloc[0].strftime(
                "%Y-%m-%d"
            ),

        "last_observation_date":
            last_date.strftime(
                "%Y-%m-%d"
            ),

        "last_observed_population":
            int(
                current_population_int
            ),

        "current_population":
            int(
                current_population_int
            ),

        # ====================================================
        # TREND
        # ====================================================

        "trend":
            trend,

        "quarterly_growth_rate":
            round(
                recent_growth_rate * 100,
                2
            ),

        "raw_quarterly_growth_rate":
            round(
                recent_growth_rate * 100,
                2
            ),

        # IMPORTANT:
        # This is now always:
        #
        # Increasing = 5%
        # Declining  = -5%
        # Stable     = 0%
        #
        "damped_quarterly_rate":
            round(
                trend_rate * 100,
                2
            ),

        "forecast_rate":
            round(
                trend_rate * 100,
                2
            ),

        "recent_window_change_percent":
            round(
                recent_window_change * 100,
                2
            ),

        # ====================================================
        # DATA
        # ====================================================

        "historical":
            historical,

        "forecast":
            forecast,

        # ====================================================
        # FINAL FORECAST
        # ====================================================

        "final_forecast":
            int(
                final_population
            ),

        "forecast_change":
            int(
                forecast_change
            ),

        "forecast_change_percent":
            (
                round(
                    forecast_change_percent,
                    2
                )
                if forecast_change_percent is not None
                else None
            ),

        # ====================================================
        # CAPACITY
        # ====================================================

        "capacity_assessment":
            capacity_assessment,

        "capacity":
            capacity_result,

        "capacity_status":
            current_capacity_status,

        "final_capacity_status":
            final_capacity_status,

        # ====================================================
        # MODEL
        # ====================================================

        "model":
            "5% Population Trend Forecast",

        "mae":
            None,

        "r2":
            None
    }

# ============================================================
# FORECAST API
# ============================================================

@app.route("/analysis/forecast",methods=["GET"])
@app.route( "/analysis/forecast/", methods=["GET"])
@login_required
@role_required("admin","field_officer","viewer")
def population_forecast():
    species_name = request.args.get("species","").strip()

    if not species_name:
        return jsonify({
            "success":False,
            "error":
                "Please select a species."
        }), 400

    years_raw = request.args.get("years","3").strip()

    try:
        years_ahead = int( years_raw)

    except ( ValueError, TypeError):
        return jsonify({
            "success":False,
            "error":"Years must be a valid whole number."
        }), 400

    if ( years_ahead < 1 or years_ahead > 10):
        return jsonify({
            "success": False,
            "error": ( "Forecast period must be between " "1 and 10 years."
                )
        }), 400

    try:
        result = forecast_species_population(
            species_name= species_name,

            years_ahead= years_ahead
        )

        if not result.get("success", False ):
            return jsonify(result), 400

        return jsonify(result), 200

    except Exception as e:
        app.logger.exception("Population forecasting error for species '%s'",species_name)
        return jsonify({
            "success":False,

            "error":( "An error occurred while generating " "the population forecast."  ),

            # Keep this during development.
            # Remove in production if internal details
            # should not be exposed.
            "details":
                str(e)
        }), 500


# ============================================================
# FORECAST PAGE
# ============================================================

@app.route( "/forecast", methods=["GET"])
@app.route(  "/forecast/", methods=["GET"])
@login_required
@role_required("admin","field_officer","viewer")
def forecast_page():
    approved_observations = (Observation.query .filter( Observation.status == "Approved").join(Observation.species).all())
    species_map = {}
    for obs in approved_observations:
        if obs.species is None:
            continue

        species_id = obs.species.id

        if species_id not in species_map:

            species_map[
                species_id
            ] = obs.species

    # --------------------------------------------------------
    # Sort species alphabetically
    # --------------------------------------------------------

    species_list = sorted(

        species_map.values(),

        key=lambda species: (
            clean_text(
                getattr(
                    species,
                    "specie_Common_Name",
                    ""
                )
            ).lower()
        )
    )

    # --------------------------------------------------------
    # Render forecast.html
    # --------------------------------------------------------

    return render_template(

        "forecast.html",

        species_list=
            species_list,

        forecast_api_url=
            url_for(
                "population_forecast"
            )
    )
    
def species_abundance():
    observations = ( Observation.query.filter_by(status="Approved").all())
    abundance = {}

    for observation in observations:
        if not observation.species:
            continue
        species_name = (observation.species.specie_Common_Name)
        abundance[species_name] = (abundance.get(species_name, 0)+ observation.population_count)
    return abundance  
 
def species_richness():
    observations = ( Observation.query.filter_by(status="Approved").all())
    species = set()

    for observation in observations:
        if observation.species:
            species.add( observation.species.specie_Common_Name)
    return len(species)  

def calculate_ecological_statistics():

    observations = (Observation.query.filter_by(status="Approved").all())

    if not observations:
        return {
            "species_richness": 0,
            "shannon": 0,
            "simpson": 0,
            "evenness": 0
        }

    populations = {}

    for obs in observations:
        if not obs.species:
            continue
        name = ( obs.species.specie_Common_Name)
        populations[name] = ( populations.get(name, 0)+ obs.population_count)
    total = sum(populations.values())

    if total <= 0:
        return {
            "species_richness": len(populations),
            "shannon": 0,
            "simpson": 0,
            "evenness": 0
        }

    proportions = np.array( list(populations.values())) / total

    proportions = proportions[ proportions > 0]
    shannon = -np.sum( proportions * np.log(proportions))
    simpson = 1 - np.sum(proportions ** 2)

    richness = len(populations)

    if richness > 1:
        evenness = (shannon /np.log(richness))

    else:
        evenness = 0

    return {
        "species_richness":richness,

        "shannon":round(shannon, 3),

        "simpson": round(simpson, 3),
        "evenness":round(evenness, 3)
    }



@app.route("/analysisPage")
@login_required
@role_required("admin", "field_officer", "viewer")
def analysis_page():
    ai_results = generate_species_effect_report()
    # Convert AI results to dictionaries
    if ai_results is not None and not ai_results.empty:
        ai_results = ai_results.to_dict(orient="records")
    else:
        ai_results = []

    # GET ALL SPECIES
    
    species_list = Species.query.all()
    all_surveys = (EnvironmentalObservation.query .join(MonitoringSite).order_by( EnvironmentalObservation.observation_date.desc(),
        EnvironmentalObservation.id.desc()) .all())

    # Dictionary:
    # monitoring_site_id -> latest environmental survey

    latest_surveys = {}
    for survey in all_surveys:
        if survey.monitoring_site is None:
            continue

        site_id = survey.monitoring_site.id

        # Because records are ordered newest first,
        # the first record we encounter for a site is its
        # current/latest survey.

        if site_id not in latest_surveys:
            latest_surveys[site_id] = survey

    # Convert dictionary to list

    environmental_observations = list(latest_surveys.values())

    # Sort current surveys by newest date

    environmental_observations.sort(
        key=lambda survey: (survey.observation_date,survey.id),
        reverse=True
    )

    # ========================================================
    # POPULATION TREND DATA
    # ========================================================

    trend_data = []

    for species in species_list:
        observations = ( Observation.query .filter_by( species_id=species.id,status="Approved")
            .order_by(Observation.observation_date.asc()).all())

        trend_data.append({
            "id": species.id,
            "name": species.specie_Common_Name,
            "dates": [
                observation.observation_date.strftime(
                    "%Y-%m-%d"
                )
                for observation in observations
                if observation.observation_date
            ],

            "counts": [
                observation.population_count or 0
                for observation in observations
            ]
        })

    # ========================================================
    # DEBUG INFORMATION
    # ========================================================

    print(
        "\n============================================"
    )

    print(
        "CURRENT ENVIRONMENTAL SURVEYS:",
        len(environmental_observations)
    )

    for survey in environmental_observations:

        site_name = (
            survey.monitoring_site.name
            if survey.monitoring_site
            else "NO MONITORING SITE"
        )

        print(
            "CURRENT SURVEY:",
            survey.id,
            "| SITE:",
            site_name,
            "| DATE:",
            survey.observation_date
        )

    print(
        "============================================\n"
    )

    # ========================================================
    # RENDER PAGE
    # ========================================================

    return render_template(
        "analysisPage.html",

        # AI ecological analysis
        ai_results=ai_results,

        # Species
        species_list=species_list,

        # Population trends
        trend_data=trend_data,

        # ONLY CURRENT/LATEST SURVEY FOR EACH SECTION
        environmental_observations=environmental_observations
    )    
    
def create_notification(role, title, message, notification_type="Info", user_id=None):

    notification = Notification(role=role,user_id=user_id, title=title,message=message,
        notification_type=notification_type)

    db.session.add(notification)
    db.session.commit()
    
@app.route("/")
def index ():
    return  redirect(url_for("login"))



@app.route("/registration", methods=['GET', 'POST'])
def registration():

    if request.method == "POST":

        user_first_name = request.form['first_name']
        user_surname= request.form['surname'] 
        user_email = request.form['email']
        user_phone = request.form['phone']
        user_password = request.form['password']
        user_gender = request.form['gender']
        user_DOB = request.form['dob']
        user_confirm_password = request.form['confirm_password']
        
        if user_password != user_confirm_password:
            flash("Passwords does not match", "danger")
            return role_required(url_for("registration"))
        if len(user_password) < 8:
            flash("Password password must be 8 at least 8 characters long", "danger")
            return redirect(url_for("registration"))
        
        user_password = generate_password_hash(request.form['password'])
        new_user = Details(First_name=user_first_name, surname=user_surname, email=user_email, phone=user_phone, DOB=user_DOB, gender=user_gender, password=user_password)   
     
        try:
            db.session.add(new_user)
            db.session.commit()
            create_notification(role="admin", title="New User Registration",message=f"{user_first_name} {user_surname} has registered.", notification_type="Info")
            flash("User registered successfully.", "success")
            return redirect(url_for('login'))

        except Exception as e:
            db.session.rollback()
            return f"Error: {e}"

    return render_template("registration.html")
  

@app.route("/login", methods=['GET', 'POST'])
def login():
    user = None

    if request.method == "POST":

        email = request.form['email']
        password = request.form['password']

        user = Details.query.filter_by(email=email).first()
        if user and check_password_hash(user.password, password):
            session['user_id'] = user.id
            session['user_name'] = f"{user.First_name} {user.surname}"
            session['role'] = user.role
            
            if user.role == "admin":
                return redirect(url_for('admin'))
            
            elif user.role == "field_officer":
                return redirect(url_for('field_Officer'))
            
            elif user.role == "viewer":
                return redirect(url_for('user'))
            else:
                return "Unknown role"
        else:
            flash("Incorrect email or password.", "danger")
    
    return render_template("login.html")

@app.route("/user")
def user():
    return render_template("user.html") 


@app.route("/admin")
@login_required
@role_required("admin")
def admin():
    total_users = Details.query.filter_by(role="viewer").count()
    total_field_officers = Details.query.filter_by(role="field_officer").count()
    total_species = Species.query.count()
    total_pending_observation = Observation.query.filter_by(status="Pending").count()

    # Count unread notifications for admin
    total_isRead_notifications = Notification.query.filter_by(role="admin",is_read=False).count()
    return render_template("admin.html",total_users=total_users, total_field_officers=total_field_officers, total_species=total_species,
        total_pending_observation=total_pending_observation,total_isRead_notifications=total_isRead_notifications)

@app.route("/view_user")
@login_required
@role_required("admin")
def view_user():
    users=Details.query.filter_by(role="viewer")
    return render_template("view_user.html", users=users)

@app.route("/view_field_Officer")
@login_required
@role_required("admin")
def view_field_Officer():
    field_officers=Details.query.filter_by(role="field_officer")
    return render_template("view_field_Officer.html", field_officers=field_officers)    

@app.route("/manageUser")
@login_required
@role_required("admin")
def managerUser():
    users = Details.query.all()
    return render_template("manageUser.html", users=users)



@app.route("/change_role/<int:id>", methods=["POST"])
@login_required
@role_required("admin")
def change_role(id):
    user = Details.query.get_or_404(id)
    new_role = request.form['role']
    user.role = new_role

    db.session.commit()
    create_notification( role=new_role, user_id=user.id, title="Role Updated",message=f"Your role has been changed to {new_role}.",notification_type="Success")
    return redirect(url_for('managerUser'))


@app.route("/delete_user/<int:id>", methods=["POST"])
@login_required
@role_required("admin")
def delete_user(id):
       
    try:
        user=Details.query.get_or_404(id)
        create_notification(role="admin", user_id=user.id, title="User Deleted", message=f"User {user.First_name} has been deleted.", notification_type="Info")
        db.session.delete(user)
        db.session.commit()
        return redirect(url_for('managerUser'))
    
    except Exception as e:
        db.session.rollback()
        return f"Error: {e}"

@app.route("/field_Officer")
@login_required
@role_required("field_officer")
def field_Officer():
    return render_template("field_Officer.html")



@app.route("/record_observation", methods=["GET", "POST"])
@login_required
@role_required("field_officer")
def record_observation():

    # =========================================================
    # GET SPECIES
    # =========================================================

    species_list = Species.query.order_by(
        Species.specie_Common_Name.asc()
    ).all()

    # =========================================================
    # GET ENVIRONMENTAL OBSERVATIONS
    #
    # These are the monitoring surveys that the field officer
    # can attach the biodiversity observation to.
    # =========================================================

    environmental_observations = EnvironmentalObservation.query.order_by(
        EnvironmentalObservation.observation_date.desc()
    ).all()

    # =========================================================
    # POST - RECORD BIODIVERSITY OBSERVATION
    # =========================================================

    if request.method == "POST":

        try:

            # -------------------------------------------------
            # GET FORM VALUES
            # -------------------------------------------------

            environmental_observation_id = request.form.get(
                "environmental_observation_id"
            )

            species_id = request.form.get("species_id")

            population_raw = request.form.get("population")

            notes = request.form.get(
                "note",
                ""
            ).strip()

            image = request.files.get("image")

            # -------------------------------------------------
            # VALIDATE ENVIRONMENTAL OBSERVATION
            # -------------------------------------------------

            if not environmental_observation_id:

                flash(
                    "Please select a monitoring section.",
                    "danger"
                )

                return redirect(url_for("record_observation"))

            try:

                environmental_observation_id = int(
                    environmental_observation_id
                )

            except (ValueError, TypeError):

                flash(
                    "Invalid monitoring section selected.",
                    "danger"
                )

                return redirect(url_for("record_observation"))

            # Get selected environmental survey
            environmental_observation = db.session.get(
                EnvironmentalObservation,
                environmental_observation_id
            )

            if not environmental_observation:

                flash(
                    "The selected environmental survey does not exist.",
                    "danger"
                )

                return redirect(url_for("record_observation"))

            # -------------------------------------------------
            # VALIDATE SPECIES
            # -------------------------------------------------

            if not species_id:

                flash(
                    "Please select a species.",
                    "danger"
                )

                return redirect(url_for("record_observation"))

            try:

                species_id = int(species_id)

            except (ValueError, TypeError):

                flash(
                    "Invalid species selected.",
                    "danger"
                )

                return redirect(url_for("record_observation"))

            species = db.session.get(
                Species,
                species_id
            )

            if not species:
                flash( "Selected species does not exist.", "danger")
                return redirect(url_for("record_observation"))

            # -------------------------------------------------
            # VALIDATE POPULATION
            # -------------------------------------------------

            if not population_raw:

                flash( "Please enter the population count.","danger")

                return redirect(url_for("record_observation"))

            try:

                population = int(population_raw)

            except (ValueError, TypeError):

                flash(
                    "Population must be a valid whole number.",
                    "danger"
                )

                return redirect(url_for("record_observation"))

            if population < 0:

                flash(
                    "Population cannot be a negative number.",
                    "danger"
                )

                return redirect(url_for("record_observation"))

            # -------------------------------------------------
            # HANDLE OPTIONAL IMAGE
            # -------------------------------------------------

            filename = None

            if image and image.filename:

                original_filename = image.filename

                safe_filename = secure_filename(
                    original_filename
                )

                if not safe_filename:

                    flash(
                        "Invalid image filename.",
                        "danger"
                    )

                    return redirect(
                        url_for("record_observation")
                    )

                allowed_extensions = {
                    "jpg",
                    "jpeg",
                    "png",
                    "gif",
                    "webp"
                }

                if "." not in safe_filename:

                    flash(
                        "Image must have a valid file extension.",
                        "danger"
                    )

                    return redirect(
                        url_for("record_observation")
                    )

                extension = safe_filename.rsplit(
                    ".",
                    1
                )[1].lower()

                if extension not in allowed_extensions:

                    flash(
                        "Invalid image type. Please upload "
                        "JPG, JPEG, PNG, GIF or WEBP.",
                        "danger"
                    )

                    return redirect(
                        url_for("record_observation")
                    )

                # Create unique filename
                unique_filename = (
                    f"{uuid.uuid4().hex}_{safe_filename}"
                )

                # Upload directory
                upload_folder = os.path.join(
                    app.root_path,
                    "static",
                    "uploads"
                )

                os.makedirs(
                    upload_folder,
                    exist_ok=True
                )

                image_path = os.path.join(
                    upload_folder,
                    unique_filename
                )

                image.save(image_path)

                filename = unique_filename

            # -------------------------------------------------
            # USE ENVIRONMENTAL SURVEY DATE
            # -------------------------------------------------

            observation_date = (
                environmental_observation.observation_date
            )

            if not observation_date:

                observation_date = datetime.now(
                    timezone.utc
                )

            # -------------------------------------------------
            # CREATE BIODIVERSITY OBSERVATION
            # -------------------------------------------------

            new_observation = Observation(

                species_id=species_id,

                population_count=population,

                notes=notes,

                photo=filename,

                observation_date=observation_date,

                status="Pending",

                environmental_observation_id=(
                    environmental_observation_id
                )
            )

            db.session.add(
                new_observation
            )

            db.session.commit()

            # -------------------------------------------------
            # NOTIFY FIELD OFFICER
            # -------------------------------------------------

            create_notification(
                role="field_officer",
                user_id=session["user_id"],
                title="Observation Recorded",
                message=(
                    f"Your observation of "
                    f"{species.specie_Common_Name} "
                    f"has been successfully recorded."
                ),
                notification_type="Success"
            )

            # -------------------------------------------------
            # NOTIFY ADMINISTRATORS
            # -------------------------------------------------

            create_notification(
                role="admin",
                title="New Observation Submitted",
                message=(
                    f"{session['user_name']} submitted a new "
                    f"biodiversity observation of "
                    f"{species.specie_Common_Name}."
                ),
                notification_type="Info"
            )

            # -------------------------------------------------
            # SUCCESS
            # -------------------------------------------------

            flash(
                "Biodiversity observation successfully recorded "
                "and linked to the monitoring section.",
                "success"
            )

            # IMPORTANT:
            # Use the actual endpoint name.
            return redirect(
                url_for("record_observation")
            )

        # =====================================================
        # ERROR HANDLING
        # =====================================================

        except Exception:

            db.session.rollback()

            app.logger.exception(
                "Error recording biodiversity observation"
            )

            flash(
                "An error occurred while recording the observation. "
                "Please try again.",
                "danger"
            )

            return redirect(
                url_for("record_observation")
            )

    # =========================================================
    # GET REQUEST
    # =========================================================

    return render_template(
        "record_observation.html",
        species_list=species_list,
        environmental_observations=environmental_observations
    )




@app.route("/pending_observations")
@login_required
@role_required("admin")
def pending_observations():
    observations = Observation.query.filter_by(status="Pending").order_by(Observation.observation_date.desc()).all()
    return render_template("pending_observations.html", observations=observations)

@app.route("/approve_observation/<int:id>", methods=["POST"])
@login_required
@role_required("admin")
def approve_observation(id):
    observation = Observation.query.get_or_404(id)
    observation.status = "Approved"
    observation.reviewed_by = session["user_id"]
    observation.reviewed_at = datetime.now(timezone.utc)
    db.session.commit()

    create_notification(role="field_officer", title="Observation Approved", message="One of your observations has been approved.",
        notification_type="Success"
    )

    return redirect(url_for("pending_observations"))

@app.route("/reject_observation/<int:id>", methods=["POST"])
@login_required
@role_required("admin")
def reject_observation(id):

    observation = Observation.query.get_or_404(id)

    observation.status = "Rejected"
    observation.reviewed_by = session["user_id"]
    observation.reviewed_at = datetime.now(timezone.utc)
    
    db.session.delete(observation)
    db.session.commit()

    create_notification(role="field_officer", title="Observation Rejected", message="One of your observations has been rejected.", notification_type="Warning" )

    return redirect(url_for("pending_observations"))

@app.route("/view_observations")
@login_required
@role_required("field_officer","admin","viewer")
def view_observations():
    observations = Observation.query.order_by(Observation.observation_date.desc()).all()
    return render_template("view_observations.html", observations=observations)

@app.route("/view_species")
@login_required
@role_required("field_officer","admin","viewer")
def view_species():
    species_list = Species.query.all()
    return render_template("view_species.html", species_list=species_list)     


@app.route("/report")
@login_required
@role_required("field_officer", "admin", "viewer")
def report():

    # ============================================================
    # FILTER VALUES
    # ============================================================

    species_filter = request.args.get("species_id", "").strip()
    site_filter = request.args.get("site_id", "").strip()
    start_date_filter = request.args.get("start_date", "").strip()
    end_date_filter = request.args.get("end_date", "").strip()
    trend_filter = request.args.get("trend", "").strip()
    risk_filter = request.args.get("risk", "").strip()

    # ============================================================
    # DROPDOWN DATA
    # ============================================================

    species_list = ( Species.query .order_by(Species.specie_Common_Name.asc())  .all())
    monitoring_sites = (MonitoringSite.query.order_by(MonitoringSite.name.asc()).all())

    # ============================================================
    # BASE OBSERVATION QUERY
    #
    # Only APPROVED observations are included.
    # ============================================================

    observation_query = (
        Observation.query
        .join(EnvironmentalObservation)
        .filter(
            Observation.status == "Approved"
        )
    )

    # ============================================================
    # SPECIES FILTER
    # ============================================================

    if species_filter:

        try:
            observation_query = observation_query.filter(
                Observation.species_id == int(species_filter)
            )
        except ValueError:
            species_filter = ""

    # ============================================================
    # MONITORING SITE / AREA FILTER
    # ============================================================

    if site_filter:

        try:
            observation_query = observation_query.filter(
                EnvironmentalObservation.monitoring_site_id
                == int(site_filter)
            )
        except ValueError:
            site_filter = ""

    # ============================================================
    # DATE FILTER
    # ============================================================

    parsed_start_date = None
    parsed_end_date = None

    if start_date_filter:

        try:
            parsed_start_date = datetime.strptime(
                start_date_filter,
                "%Y-%m-%d"
            )

            observation_query = observation_query.filter(
                Observation.observation_date
                >= parsed_start_date
            )

        except ValueError:
            start_date_filter = ""

    if end_date_filter:

        try:
            parsed_end_date = (
                datetime.strptime(
                    end_date_filter,
                    "%Y-%m-%d"
                )
                + timedelta(days=1)
            )

            observation_query = observation_query.filter(
                Observation.observation_date
                < parsed_end_date
            )

        except ValueError:
            end_date_filter = ""

    # ============================================================
    # GET FILTERED OBSERVATIONS
    # ============================================================

    filtered_observations = (
        observation_query
        .order_by(
            Observation.observation_date.asc()
        )
        .all()
    )

    # ============================================================
    # TOTAL STATISTICS
    # ============================================================

    total_observations = len(
        filtered_observations
    )

    total_population = sum(
        observation.population_count or 0
        for observation in filtered_observations
    )

    # Species represented in the filtered data
    filtered_species_ids = {
        observation.species_id
        for observation in filtered_observations
        if observation.species_id
    }

    total_species = len(
        filtered_species_ids
    )

    # ============================================================
    # PROCESS SPECIES
    # ============================================================

    report_data = []

    for species in species_list:

        species_observations = [
            observation
            for observation in filtered_observations
            if observation.species_id == species.id
        ]

        # Skip species that do not belong to the current filter
        if not species_observations:
            continue

        # --------------------------------------------------------
        # FIRST / LATEST
        # --------------------------------------------------------

        species_observations.sort(
            key=lambda observation:
            observation.observation_date
            or datetime.min
        )

        first_observation = (
            species_observations[0]
        )

        latest_observation = (
            species_observations[-1]
        )

        first_population = (
            first_observation.population_count
            or 0
        )

        latest_population = (
            latest_observation.population_count
            or 0
        )

        population_change = (
            latest_population
            - first_population
        )

        # --------------------------------------------------------
        # PERCENTAGE CHANGE
        # --------------------------------------------------------

        if first_population > 0:

            percentage_change = (
                population_change
                / first_population
            ) * 100

        else:

            percentage_change = 0

        # --------------------------------------------------------
        # TREND
        # --------------------------------------------------------

        if population_change > 0:

            trend = "Increasing"

        elif population_change < 0:

            trend = "Decreasing"

        else:

            trend = "Stable"

        # --------------------------------------------------------
        # RISK
        #
        # This keeps your existing population-based thresholds.
        # --------------------------------------------------------

        if latest_population <= 5:

            risk_status = "Critical"

        elif latest_population <= 10:

            risk_status = "Endangered"

        elif latest_population <= 20:

            risk_status = "Vulnerable"

        else:

            risk_status = "Stable"

        # --------------------------------------------------------
        # TREND FILTER
        # --------------------------------------------------------

        if trend_filter and trend != trend_filter:

            continue

        # --------------------------------------------------------
        # RISK FILTER
        # --------------------------------------------------------

        if risk_filter and risk_status != risk_filter:

            continue

        # --------------------------------------------------------
        # CHART
        # --------------------------------------------------------

        chart_file = (
            f"charts/chart_{species.id}.png"
        )

        chart_path = os.path.join(
            app.static_folder,
            chart_file
        )

        chart = (
            chart_file
            if os.path.exists(chart_path)
            else None
        )

        # --------------------------------------------------------
        # AREA INFORMATION
        # --------------------------------------------------------

        areas = []

        for observation in species_observations:

            environmental = (
                observation.environmental_observation
            )

            if (
                environmental
                and environmental.monitoring_site
            ):

                site_name = (
                    environmental
                    .monitoring_site
                    .name
                )

                if site_name not in areas:
                    areas.append(site_name)

        # --------------------------------------------------------
        # REPORT ITEM
        # --------------------------------------------------------

        report_data.append({

            "id": species.id,

            "name":
                species.specie_Common_Name,

            "scientific":
                getattr(
                    species,
                    "scientificName",
                    "N/A"
                ),

            "habitat":
                species.specie_Habitat,

            "location":
                species.location,

            "areas":
                areas,

            "observations":
                species_observations,

            "observation_count":
                len(species_observations),

            "first_population":
                first_population,

            "latest_population":
                latest_population,

            "population_change":
                population_change,

            "percentage_change":
                round(
                    percentage_change,
                    2
                ),

            "trend":
                trend,

            "risk_status":
                risk_status,

            "chart":
                chart
        })

    # ============================================================
    # SUMMARY COUNTS
    # ============================================================

    increasing_species = sum(
        1
        for item in report_data
        if item["trend"] == "Increasing"
    )

    decreasing_species = sum(
        1
        for item in report_data
        if item["trend"] == "Decreasing"
    )

    stable_species = sum(
        1
        for item in report_data
        if item["trend"] == "Stable"
    )

    critical_species = sum(
        1
        for item in report_data
        if item["risk_status"] == "Critical"
    )

    endangered_species = sum(
        1
        for item in report_data
        if item["risk_status"] == "Endangered"
    )

    vulnerable_species = sum(
        1
        for item in report_data
        if item["risk_status"] == "Vulnerable"
    )

    # ============================================================
    # RENDER
    # ============================================================

    return render_template("report.html", report_data=report_data, species_list=species_list,
        monitoring_sites=monitoring_sites,
        total_species=total_species,
        total_observations=total_observations,
        total_population=total_population,
        increasing_species=increasing_species,
        decreasing_species=decreasing_species,
        stable_species=stable_species,
        critical_species=critical_species,
        endangered_species=endangered_species,
        vulnerable_species=vulnerable_species,

        # Current filters
        selected_species=species_filter,
        selected_site=site_filter,
        selected_start_date=start_date_filter,
        selected_end_date=end_date_filter,
        selected_trend=trend_filter,
        selected_risk=risk_filter
    )

@app.route("/download_report")
@login_required
@role_required("field_officer","admin","viewer")
def download_report():
    
    species_list = Species.query.all()
    story = []
    styles = getSampleStyleSheet()
    story.append(Paragraph("BIODIVERSITY MONITORING REPORT", styles["Title"]))
    story.append(Spacer(1, 20))

    # ensure chart folder exists
    chart_folder = os.path.join("static", "charts")
    os.makedirs(chart_folder, exist_ok=True)

    for species in species_list:
        story.append(Paragraph(f"Species: {species.specie_Common_Name}",styles["Heading2"]))
        story.append(Paragraph(f"Scientific Name: {species.scientificName}", styles["Normal"]))
        story.append(Spacer(1, 10))
        observations = Observation.query.filter_by(species_id=species.id).order_by(Observation.observation_date.asc()).all()
        # collect data for GRAPH
        dates = []
        counts = []

        for obs in observations:
            dates.append(obs.observation_date.strftime('%Y-%m-%d'))
            counts.append(obs.population_count)
            story.append(Paragraph(f"Date: {obs.observation_date.strftime('%Y-%m-%d %H:%M')}",styles["Normal"]))
            story.append(Paragraph(f"Population: {obs.population_count}",styles["Normal"]))
            story.append(Paragraph(f"Notes: {obs.notes}", styles["Normal"]))
            story.append(Spacer(1, 8))

            # IMAGE
            if obs.photo:
                image_path = os.path.join("static", "uploads", obs.photo)
                if os.path.exists(image_path):
                    img = Image(image_path)
                    img.drawHeight = 2 * inch
                    img.drawWidth = 3 * inch
                    story.append(img)
            story.append(Spacer(1, 15))

        # ---------------------------
        # CREATE GRAPH PER SPECIES
        # ---------------------------
        if len(dates) > 0:
            chart_path = os.path.join(chart_folder, f"chart_{species.id}.png")
            plt.figure(figsize=(7,4))
            plt.plot(dates, counts, marker='o', linewidth=2 )
            plt.title(f"{species.specie_Common_Name} Population Trend", fontsize=14)
            plt.xlabel("Date", fontsize=12)
            plt.ylabel("Population Count", fontsize=12)
            plt.xticks(rotation=45)
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
            plt.savefig(chart_path, dpi=300)
            plt.close()
            # add chart to PDF
            if os.path.exists(chart_path):
                chart_img = Image(chart_path)
                chart_img.drawHeight = 2.5 * inch
                chart_img.drawWidth = 4 * inch
                story.append(Paragraph("Population Trend", styles["Heading3"]))
                story.append(chart_img)
                story.append(Spacer(1, 25))


    # BUILD PDF
    filename = "CBU_NATURE PARK_Biodiversity_Report.pdf"
    doc = SimpleDocTemplate(filename)
    doc.build(story)
    return send_file(filename, as_attachment=True)
@app.route("/notifications")
@login_required
def notifications():
    notifications = Notification.query.filter(
        (Notification.role == session["role"]) | (Notification.user_id == session["user_id"])
    ).order_by(Notification.created_at.desc()).all()

    unread_count = sum(1 for n in notifications if not n.is_read)

    return render_template("notifications.html", notifications=notifications,unread_count=unread_count)

@app.route("/manageSpecies", methods=["GET", "POST"])
@login_required
@role_required("field_officer")
def manageSpecies():
    species = Species.query.all()
    return render_template("manageSpecies.html", species=species)

@app.route("/delete_species/<int:id>", methods=["POST"])
@login_required
@role_required("admin", "field_officer")
def delete_species(id):
    species = Species.query.get_or_404(id)
    db.session.delete(species)
    db.session.commit()
    create_notification( role="admin", title="Species Deleted", message=f"{session['user_name']} removed {species.specie_Common_Name}.", notification_type="Warning")
    return redirect(url_for("view_species"))

@app.route("/add_species", methods=["GET", "POST"])
@login_required
@role_required("field_officer")
def add_species():

    if request.method == "POST":
        scientific_name = request.form["scientificName"]
        common_name = request.form["commonName"]
        habitat = request.form["habitat"]
        location = request.form["location"]

        new_species = Species(scientificName=scientific_name, specie_Common_Name=common_name,
            specie_Habitat=habitat, location=location)

        db.session.add(new_species)
        db.session.commit()
        return redirect(url_for("view_species"))

    return render_template("add_species.html")

@app.route("/search")
@login_required
@role_required("admin", "field_officer", "viewer")
def search():
    search = request.args.get("search", "").strip()
    role = session.get("role")
    results = { "users": [], "species": [],"observations": []}

    if search:
        # Admin can search everything
        if role == "admin":
            results["users"] = Details.query.filter((Details.First_name.contains(search)) |(Details.surname.contains(search)) |
                (Details.email.contains(search)) | (Details.role.contains(search))).all()

            results["species"] = Species.query.filter((Species.scientificName.contains(search)) | (Species.specie_Common_Name.contains(search)) |
                (Species.location.contains(search)) ).all()

            results["observations"] = Observation.query.join(Species).filter(
                (Species.scientificName.contains(search)) |
                (Species.specie_Common_Name.contains(search)) |
                (Observation.notes.contains(search))
            ).all()

        # Field officers cannot search users
        elif role == "field_officer":
            results["species"] = Species.query.filter((Species.scientificName.contains(search)) |(Species.specie_Common_Name.contains(search)) |(Species.location.contains(search))
            ).all()

            results["observations"] = Observation.query.join(Species).filter( (Species.scientificName.contains(search)) |(Species.specie_Common_Name.contains(search)) |
                (Observation.notes.contains(search))).all()

        # Viewers can search species only
        elif role == "viewer":
            results["species"] = Species.query.filter((Species.scientificName.contains(search)) | (Species.specie_Common_Name.contains(search)) |
                (Species.location.contains(search))).all()

    return render_template("search_results.html",search=search,  results=results)

@app.route("/forgot_password", methods=["GET", "POST"])
def forgot_password():

    if request.method == "POST":
        email = request.form["email"]
        user = Details.query.filter_by(email=email).first()
        if not user:
            return "Email not found"
        # Generate token
        token = secrets.token_urlsafe(32)
        user.reset_token = token
        user.reset_token_expiry = datetime.now(timezone.utc) + timedelta(minutes=30)
        db.session.commit()

        reset_link = url_for("resetPassword", token=token, _external=True)

        # Instead of sending email, display link
        return f"""Password reset link created:<br><br><a href="{reset_link}">
        {reset_link}
        </a>
        """
    return render_template("forgot_password.html")

@app.route("/gallery")
@role_required("admin", "field_officer", "viewer")
def gallery():
    observations = Observation.query.filter(Observation.photo.isnot(None)).order_by(Observation.observation_date.desc()).all()
    return render_template("gallery.html",observations=observations)


# ============================================================
# MANAGE CBU NATURE PARK MONITORING SITES
# ============================================================
# ============================================================
# MANAGE MONITORING SITES
# ============================================================

@app.route("/manage_sites", methods=["GET", "POST"])
@login_required
#@role_required("admin")
def manage_sites():

    PARK_AREA = 9.12

    # ========================================================
    # POST - ADD NEW MONITORING SITE
    # ========================================================

    if request.method == "POST":

        # ----------------------------------------------------
        # GET FORM VALUES
        # ----------------------------------------------------

        name = request.form.get("name", "").strip()
        area_raw = request.form.get("area_hectares", "").strip()
        description = request.form.get("description", "").strip()

        # ----------------------------------------------------
        # VALIDATE SITE NAME
        # ----------------------------------------------------

        if not name:
            flash(
                "Please enter the monitoring site or section name.",
                "danger"
            )

            return redirect(url_for("manage_sites"))

        # ----------------------------------------------------
        # VALIDATE AREA
        # ----------------------------------------------------

        if not area_raw:
            flash( "Please enter the area of the monitoring site.", "danger")
            return redirect(url_for("manage_sites"))

        try:
            area_hectares = float(area_raw)

        except (ValueError, TypeError):
            flash("Area must be a valid number.",  "danger")
            return redirect(url_for("manage_sites"))

        # ----------------------------------------------------
        # CHECK AREA VALUE
        # ----------------------------------------------------

        if area_hectares <= 0:
            flash( "Area must be greater than 0 hectares.","danger")
            return redirect(url_for("manage_sites"))

        # ----------------------------------------------------
        # CHECK DUPLICATE SITE NAME
        # ----------------------------------------------------

        existing_site = (MonitoringSite.query.filter( func.lower(MonitoringSite.name) == name.lower()).first())

        if existing_site:
            flash(f"A monitoring site named '{name}' already exists.","danger" )
            return redirect(url_for("manage_sites"))

        # ----------------------------------------------------
        # GET CURRENTLY ALLOCATED AREA
        # ----------------------------------------------------

        current_area = (db.session.query( func.coalesce( func.sum(MonitoringSite.area_hectares), 0 )).scalar())

        current_area = float(current_area or 0)
        
        new_total_area = current_area + area_hectares

        if new_total_area > PARK_AREA:
            remaining_area = PARK_AREA - current_area
            
            flash( f"Cannot add this monitoring site. " f"The total park area is {PARK_AREA:.2f} ha. "  f"Currently allocated: {current_area:.2f} ha. "
                f"Remaining area: {max(remaining_area, 0):.2f} ha.", "danger")
            return redirect(url_for("manage_sites"))

        # CREATE MONITORING SITE
        try:

            new_site = MonitoringSite(name=name, area_hectares=area_hectares, description=description)
            db.session.add(new_site)
            db.session.commit()

        except Exception:
            db.session.rollback()
            app.logger.exception("Error adding monitoring site")
            flash("An error occurred while adding the monitoring site.","danger")
            return redirect(url_for("manage_sites"))

        # ----------------------------------------------------
        # CREATE NOTIFICATION
        # ----------------------------------------------------

        try:
            user_name = session.get( "user_name","Administrator" )
            create_notification( role="admin", title="Monitoring Site Added",
                message=( f"{user_name} added the monitoring site "  f"'{name}' covering " f"{area_hectares:.2f} hectares."),notification_type="Success")

        except Exception:
            app.logger.exception("Monitoring site was added, but notification failed."  )

        # SUCCESS MESSAGE
        flash( f"Monitoring site '{name}' was added successfully.", "success")
        return redirect(url_for("manage_sites"))

    # ========================================================
    # GET - DISPLAY MONITORING SITES
    # ========================================================

    sites = (MonitoringSite.query .order_by(MonitoringSite.name.asc()).all())
    # CALCULATE TOTAL ALLOCATED AREA
    total_area = (db.session.query( func.coalesce( func.sum(MonitoringSite.area_hectares),0)).scalar())

    total_area = float(total_area or 0)
    remaining_area = max(PARK_AREA - total_area,0)

    return render_template( "manage_sites.html", sites=sites, total_area=total_area, remaining_area=remaining_area,
        park_area=PARK_AREA)

@app.route("/delete_site/<int:id>", methods=["POST"])
@login_required
@role_required("admin")
def delete_site(id):

    site = MonitoringSite.query.get_or_404(id)

    site_name = site.name

    try:

        # Check whether environmental observations
        # are linked to this monitoring site.
        environmental_count = (
            EnvironmentalObservation.query
            .filter_by(
                monitoring_site_id=site.id
            )
            .count()
        )

        if environmental_count > 0:

            flash(
                f"Cannot delete '{site_name}' because "
                f"{environmental_count} environmental "
                f"observation(s) are linked to this "
                f"monitoring site.",
                "danger"
            )

            return redirect(
                url_for("manage_sites")
            )

        # Delete the monitoring site
        db.session.delete(site)

        db.session.commit()

    except Exception:

        db.session.rollback()

        app.logger.exception(
            "Error deleting monitoring site"
        )

        flash(
            "Unable to delete the monitoring site.",
            "danger"
        )

        return redirect(
            url_for("manage_sites")
        )

    # Optional notification
    try:

        user_name = session.get(
            "user_name",
            "Administrator"
        )

        create_notification(
            role="admin",
            title="Monitoring Site Deleted",
            message=(
                f"{user_name} deleted the monitoring site "
                f"'{site_name}'."
            ),
            notification_type="Warning"
        )

    except Exception:

        app.logger.exception(
            "Monitoring site was deleted, "
            "but notification failed."
        )

    flash(
        f"Monitoring site '{site_name}' "
        f"was deleted successfully.",
        "success"
    )

    return redirect(
        url_for("manage_sites")
    )

SASSCAL_STATION_ID = 61014

SASSCAL_DAILY_URL = (
    "https://sasscalweathernet.org/"
    "weatherstat_daily_we.php"
    "?loggerid_crit=61014"
)

def get_weekly_weather():
    """
    Get a weekly summary (last 7 available daily records)
    from SASSCAL WeatherNet Station 61014 - CBU Kitwe.

    Returns average temperature and total rainfall over
    the most recent 7 days that have data.
    """

    try:
        response = requests.get(
            SASSCAL_DAILY_URL,
            timeout=30,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/153.0 Safari/537.36"
                ),
                "Accept": "text/html,application/xhtml+xml",
            },
        )
        response.raise_for_status()

        tables = pd.read_html(StringIO(response.text))
        if not tables:
            raise ValueError("SASSCAL returned no HTML tables.")

        weather_df = None
        for table in tables:
            if isinstance(table.columns, pd.MultiIndex):
                new_columns = []
                for column in table.columns:
                    parts = [
                        str(p).strip() for p in column
                        if str(p).strip() and str(p).strip().lower() not in ("nan", "none")
                    ]
                    new_columns.append(" ".join(parts))
                table.columns = new_columns
            else:
                table.columns = [str(c).strip() for c in table.columns]

            column_names = [str(c).lower() for c in table.columns]
            has_date = any("date" in c for c in column_names)
            has_temperature = any("air temp" in c and "avg" in c for c in column_names)
            has_rainfall = any("precip" in c and "total" in c for c in column_names)

            if has_date and has_temperature and has_rainfall:
                weather_df = table.copy()
                break

        if weather_df is None:
            raise ValueError("Could not find the SASSCAL daily weather table.")

        date_column = temperature_column = rainfall_column = None
        for column in weather_df.columns:
            name = str(column).strip().lower()
            if date_column is None and "date" in name:
                date_column = column
            if temperature_column is None and "air temp" in name and "avg" in name:
                temperature_column = column
            if rainfall_column is None and "precip" in name and "total" in name:
                rainfall_column = column

        if not all([date_column, temperature_column, rainfall_column]):
            raise ValueError("Required SASSCAL columns were not found.")

        weather_df = weather_df[[date_column, temperature_column, rainfall_column]].copy()
        weather_df.columns = ["date", "temperature", "rainfall"]

        weather_df["date"] = pd.to_datetime(weather_df["date"], errors="coerce")
        weather_df["temperature"] = pd.to_numeric(weather_df["temperature"], errors="coerce")
        weather_df["rainfall"] = pd.to_numeric(weather_df["rainfall"], errors="coerce")

        weather_df = weather_df.dropna(subset=["date", "temperature", "rainfall"])
        weather_df = weather_df.sort_values(by="date").drop_duplicates(subset=["date"], keep="last")

        if weather_df.empty:
            raise ValueError("SASSCAL returned no valid daily weather records.")

        # ---- weekly aggregation: last 7 available days ----
        last_week = weather_df.tail(7)

        return {
            "start_date": last_week["date"].min().date(),
            "end_date": last_week["date"].max().date(),
            "days_used": len(last_week),
            "temperature": round(last_week["temperature"].mean(), 2),
            "rainfall": round(last_week["rainfall"].sum(), 2),
            "source": "SASSCAL WeatherNet",
            "station_id": SASSCAL_STATION_ID,
        }

    except requests.RequestException as error:
        print("\nSASSCAL CONNECTION ERROR:", error)
        return None
    except (ValueError, KeyError, IndexError) as error:
        print("\nSASSCAL DATA ERROR:", error)
        return None
    except Exception as error:
        print("\nUNEXPECTED SASSCAL ERROR:", type(error).__name__, error)
        return None   
# ============================================================
# RECORD ENVIRONMENTAL OBSERVATION
# ============================================================
# ============================================================
# RECORD ENVIRONMENTAL OBSERVATION
# ============================================================

@app.route("/record_environment", methods=["GET", "POST"])
@login_required
@role_required("field_officer")
def record_environment():

    # --------------------------------------------------------
    # MONITORING SITES
    # --------------------------------------------------------

    sites = (
        MonitoringSite.query
        .order_by(MonitoringSite.name.asc())
        .all()
    )


    # --------------------------------------------------------
    # GET WEATHER DATA
    # --------------------------------------------------------

    weekly_weather = get_weekly_weather()


    # ========================================================
    # GET REQUEST
    # ========================================================

    if request.method == "GET":

        return render_template(
            "record_environment.html",
            sites=sites,
            weekly_weather=weekly_weather
        )


    # ========================================================
    # POST REQUEST
    # ========================================================

    # IMPORTANT:
    #
    # We TRY SASSCAL again when saving.
    #
    # If SASSCAL works:
    #     use automatic temperature/rainfall.
    #
    # If SASSCAL fails:
    #     use temperature/rainfall entered by the user.
    #
    # DO NOT redirect immediately when SASSCAL fails.

    weekly_weather = get_weekly_weather()


    # ========================================================
    # MONITORING SITE
    # ========================================================

    monitoring_site_id = request.form.get(
        "monitoring_site_id",
        ""
    ).strip()


    if not monitoring_site_id:

        flash(
            "Please select a monitoring site.",
            "danger"
        )

        return render_template(
            "record_environment.html",
            sites=sites,
            weekly_weather=weekly_weather
        )


    try:

        monitoring_site_id = int(
            monitoring_site_id
        )

    except (
        TypeError,
        ValueError
    ):

        flash(
            "Invalid monitoring site selected.",
            "danger"
        )

        return render_template(
            "record_environment.html",
            sites=sites,
            weekly_weather=weekly_weather
        )


    # ========================================================
    # FIND MONITORING SITE
    # ========================================================

    monitoring_site = db.session.get(
        MonitoringSite,
        monitoring_site_id
    )


    if monitoring_site is None:

        flash(
            "The selected monitoring site does not exist.",
            "danger"
        )

        return render_template(
            "record_environment.html",
            sites=sites,
            weekly_weather=weekly_weather
        )


    # ========================================================
    # NUMBER HELPER
    # ========================================================

    def get_float(field_name):

        value = request.form.get(
            field_name,
            ""
        ).strip()


        if value == "":
            return None


        try:

            number = float(value)


            if not math.isfinite(number):

                raise ValueError


            return number


        except (
            TypeError,
            ValueError
        ):

            raise ValueError(
                f"{field_name.replace('_', ' ').title()} "
                "must be a valid number."
            )


    # ========================================================
    # READ FORM VALUES
    # ========================================================

    try:

        soil_ph = get_float(
            "soil_ph"
        )

        soil_moisture = get_float(
            "soil_moisture"
        )

        water_ph = get_float(
            "water_ph"
        )

        water_turbidity = get_float(
            "water_turbidity"
        )

        vegetation_cover = get_float(
            "vegetation_cover"
        )

        vegetation_density = get_float(
            "vegetation_density"
        )

        grass_availability = get_float(
            "grass_availability"
        )

        tree_density = get_float(
            "tree_density"
        )

    except ValueError as error:

        flash(
            str(error),
            "danger"
        )

        return render_template(
            "record_environment.html",
            sites=sites,
            weekly_weather=weekly_weather
        )


    # ========================================================
    # WEATHER
    # ========================================================

    weather_source = None
    weather_start_date = None
    weather_end_date = None


    # --------------------------------------------------------
    # AUTOMATIC SASSCAL WEATHER
    # --------------------------------------------------------

    if weekly_weather is not None:

        temperature = weekly_weather.get(
            "temperature"
        )

        rainfall = weekly_weather.get(
            "rainfall"
        )

        weather_start_date = weekly_weather.get(
            "start_date"
        )

        weather_end_date = weekly_weather.get(
            "end_date"
        )

        weather_source = (
            "SASSCAL WeatherNet"
        )


    # --------------------------------------------------------
    # MANUAL WEATHER FALLBACK
    # --------------------------------------------------------

    else:

        try:

            temperature = get_float(
                "temperature"
            )

            rainfall = get_float(
                "rainfall"
            )

        except ValueError as error:

            flash(
                str(error),
                "danger"
            )

            return render_template(
                "record_environment.html",
                sites=sites,
                weekly_weather=None
            )


        # ----------------------------------------------------
        # MANUAL TEMPERATURE REQUIRED
        # ----------------------------------------------------

        if temperature is None:

            flash(
                "SASSCAL WeatherNet is unavailable. "
                "Please enter the weekly average temperature manually.",
                "danger"
            )

            return render_template(
                "record_environment.html",
                sites=sites,
                weekly_weather=None
            )


        # ----------------------------------------------------
        # MANUAL RAINFALL REQUIRED
        # ----------------------------------------------------

        if rainfall is None:

            flash(
                "SASSCAL WeatherNet is unavailable. "
                "Please enter the weekly total rainfall manually.",
                "danger"
            )

            return render_template(
                "record_environment.html",
                sites=sites,
                weekly_weather=None
            )


        # ----------------------------------------------------
        # MANUAL RAINFALL VALIDATION
        # ----------------------------------------------------

        if rainfall < 0:

            flash(
                "Rainfall cannot be negative.",
                "danger"
            )

            return render_template(
                "record_environment.html",
                sites=sites,
                weekly_weather=None
            )


        weather_source = (
            "Manual Entry - SASSCAL unavailable"
        )


    # ========================================================
    # WEATHER VALIDATION
    # ========================================================

    if temperature is None:

        flash(
            "Temperature is required.",
            "danger"
        )

        return render_template(
            "record_environment.html",
            sites=sites,
            weekly_weather=weekly_weather
        )


    if rainfall is None:

        flash(
            "Rainfall is required.",
            "danger"
        )

        return render_template(
            "record_environment.html",
            sites=sites,
            weekly_weather=weekly_weather
        )


    if rainfall < 0:

        flash(
            "Rainfall cannot be negative.",
            "danger"
        )

        return render_template(
            "record_environment.html",
            sites=sites,
            weekly_weather=weekly_weather
        )


    # ========================================================
    # SOIL PH VALIDATION
    # ========================================================

    if (
        soil_ph is not None
        and not 0 <= soil_ph <= 14
    ):

        flash(
            "Soil pH must be between 0 and 14.",
            "danger"
        )

        return render_template(
            "record_environment.html",
            sites=sites,
            weekly_weather=weekly_weather
        )


    # ========================================================
    # WATER PH VALIDATION
    # ========================================================

    if (
        water_ph is not None
        and not 0 <= water_ph <= 14
    ):

        flash(
            "Water pH must be between 0 and 14.",
            "danger"
        )

        return render_template(
            "record_environment.html",
            sites=sites,
            weekly_weather=weekly_weather
        )


    # ========================================================
    # SOIL MOISTURE
    # ========================================================

    if (
        soil_moisture is not None
        and not 0 <= soil_moisture <= 100
    ):

        flash(
            "Soil moisture must be between 0 and 100%.",
            "danger"
        )

        return render_template(
            "record_environment.html",
            sites=sites,
            weekly_weather=weekly_weather
        )


    # ========================================================
    # WATER TURBIDITY
    # ========================================================

    if (
        water_turbidity is not None
        and water_turbidity < 0
    ):

        flash(
            "Water turbidity cannot be negative.",
            "danger"
        )

        return render_template(
            "record_environment.html",
            sites=sites,
            weekly_weather=weekly_weather
        )


    # ========================================================
    # PERCENTAGE VALIDATION
    # ========================================================

    percentage_fields = {

        "Vegetation cover":
            vegetation_cover,

        "Vegetation density":
            vegetation_density,

        "Grass availability":
            grass_availability,

        "Tree density":
            tree_density

    }


    for field_name, value in percentage_fields.items():

        if (
            value is not None
            and not 0 <= value <= 100
        ):

            flash(
                f"{field_name} must be between 0 and 100%.",
                "danger"
            )

            return render_template(
                "record_environment.html",
                sites=sites,
                weekly_weather=weekly_weather
            )


    # ========================================================
    # NOTES
    # ========================================================

    notes = request.form.get(
        "notes",
        ""
    ).strip()


    # ========================================================
    # USER
    # ========================================================

    recorded_by = session.get(
        "user_id"
    )


    if not recorded_by:

        flash(
            "Your user session could not be identified.",
            "danger"
        )

        return render_template(
            "record_environment.html",
            sites=sites,
            weekly_weather=weekly_weather
        )


    # ========================================================
    # CREATE ENVIRONMENTAL OBSERVATION
    # ========================================================

    environmental_observation = EnvironmentalObservation(

        monitoring_site_id=
            monitoring_site.id,

        location=
            monitoring_site.name,

        area_hectares=
            monitoring_site.area_hectares,

        observation_date=
            datetime.now(timezone.utc),

        # ----------------------------------------------------
        # WEATHER
        # ----------------------------------------------------

        temperature=
            temperature,

        rainfall=
            rainfall,

        # ----------------------------------------------------
        # SOIL
        # ----------------------------------------------------

        soil_ph=
            soil_ph,

        soil_moisture=
            soil_moisture,

        # ----------------------------------------------------
        # WATER
        # ----------------------------------------------------

        water_ph=
            water_ph,

        water_turbidity=
            water_turbidity,

        # ----------------------------------------------------
        # VEGETATION
        # ----------------------------------------------------

        vegetation_cover=
            vegetation_cover,

        vegetation_density=
            vegetation_density,

        grass_availability=
            grass_availability,

        tree_density=
            tree_density,

        # ----------------------------------------------------
        # NOTES
        # ----------------------------------------------------

        notes=
            notes,

        # ----------------------------------------------------
        # USER
        # ----------------------------------------------------

        recorded_by=
            recorded_by
    )


    # ========================================================
    # SAVE
    # ========================================================

    try:

        db.session.add(
            environmental_observation
        )

        db.session.commit()


    except Exception as error:

        db.session.rollback()

        print(
            "Environmental observation database error:",
            error
        )

        flash(
            "An error occurred while saving "
            "the environmental observation.",
            "danger"
        )

        return render_template(
            "record_environment.html",
            sites=sites,
            weekly_weather=weekly_weather
        )


    # ========================================================
    # SUCCESS MESSAGE
    # ========================================================

    if weather_source == "SASSCAL WeatherNet":

        flash(
            f"Environmental data for "
            f"'{monitoring_site.name}' "
            f"was recorded successfully. "
            f"Weekly weather data from "
            f"{weather_start_date} to "
            f"{weather_end_date} "
            f"was obtained automatically from "
            f"SASSCAL WeatherNet Station "
            f"{SASSCAL_STATION_ID}.",
            "success"
        )

    else:

        flash(
            f"Environmental data for "
            f"'{monitoring_site.name}' "
            f"was recorded successfully. "
            f"SASSCAL WeatherNet was unavailable, "
            f"so the temperature and rainfall were "
            f"entered manually.",
            "warning"
        )


    return redirect(
        url_for("record_environment")
    )    
@app.route("/resetPassword", methods=["GET", "POST"])
def resetPassword():
    token = request.args.get("token")
    user = Details.query.filter_by(reset_token=token).first()
    
    if not user:
        return "Invalid reset link"

    if request.method == "POST":
        password = request.form["password"]
        confirm_password = request.form["confirm_password"]
        
        if password != confirm_password:
            return "Passwords do not match"

        user.password = generate_password_hash(password)
        user.reset_token = None
        user.reset_token_expiry = None
        db.session.commit()
        return redirect(url_for("login"))
    return render_template( "resetPassword.html")

@app.route ("/back")
@login_required
def back ():
    role = session.get('role')
    if role == "admin":
        return redirect(url_for("admin"))
    elif role == "field_officer":
        return redirect(url_for("field_Officer"))
    elif role == "viewer":
        return redirect(url_for("user"))
    else: 
        flash("Unexpected Error as occurred ")
        return redirect(url_for ("login"))
    
@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
        # Create a default admin account if one doesn't exist yet
        if not Details.query.filter_by(email="admin@cbunaturepark.com").first():
            default_admin = Details(
                First_name="Admin",
                surname="User",
                email="admin@cbunaturepark.com",
                phone="0000000000",
                DOB="2000-01-01",
                gender="N/A",
                password=generate_password_hash("admin1234"),
                role="admin"
            )
            db.session.add(default_admin)
            db.session.commit()
    app.run(debug=True) 
