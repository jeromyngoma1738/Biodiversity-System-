from datetime import datetime, timezone

from extensions import db


# ============================================================
# USER / DETAILS
# ============================================================

class Details(db.Model):
    __tablename__ = "details"

    id = db.Column(db.Integer, primary_key=True)

    First_name = db.Column(
        db.String(50),
        nullable=False
    )

    surname = db.Column(
        db.String(50),
        nullable=False
    )

    email = db.Column(
        db.String(50),
        unique=True,
        nullable=False
    )

    phone = db.Column(db.String(20))

    DOB = db.Column(db.String(20))

    gender = db.Column(db.String(10))

    password = db.Column(
        db.String(255),
        nullable=False
    )

    role = db.Column(
        db.String(50),
        nullable=False,
        default="viewer"
    )

    date_Created = db.Column(
        db.DateTime,
        default=lambda: datetime.now(timezone.utc)
    )

    reset_token = db.Column(
        db.String(200),
        nullable=True
    )

    reset_token_expiry = db.Column(
        db.DateTime,
        nullable=True
    )

    reviewed_observations = db.relationship(
        "Observation",
        foreign_keys="Observation.reviewed_by",
        backref="reviewer",
        lazy=True
    )

    def __repr__(self):
        return f"<User {self.First_name} {self.surname}>"


# ============================================================
# SPECIES
# ============================================================

class Species(db.Model):
    __tablename__ = "species"

    id = db.Column(
        db.Integer,
        primary_key=True
    )

    scientificName = db.Column(
        db.String(100),
        nullable=False
    )

    specie_Common_Name = db.Column(
        db.String(100),
        nullable=False
    )

    specie_Habitat = db.Column(
        db.String(100),
        nullable=False,
        default="savannah"
    )

    # General/default location.
    # The actual analysis section comes from MonitoringSite.
    location = db.Column(
        db.String(100),
        nullable=False,
        default="CBU Nature Park"
    )

    observations = db.relationship(
        "Observation",
        backref="species",
        lazy=True,
        cascade="all, delete-orphan"
    )

    analyses = db.relationship(
        "Analysis",
        backref="species",
        lazy=True,
        cascade="all, delete-orphan"
    )

    def __repr__(self):
        return f"<Species {self.specie_Common_Name}>"


# ============================================================
# MONITORING SITE / LAND SECTION
# ============================================================

class MonitoringSite(db.Model):
    __tablename__ = "monitoring_sites"

    id = db.Column(
        db.Integer,
        primary_key=True
    )

    # Name of the actual land section
    name = db.Column(
        db.String(100),
        nullable=False,
        unique=True
    )

    # Size of this particular section
    area_hectares = db.Column(
        db.Float,
        nullable=False
    )

    description = db.Column(
        db.Text,
        nullable=True
    )

    created_at = db.Column(
        db.DateTime,
        default=lambda: datetime.now(timezone.utc),
        nullable=False
    )

    environmental_observations = db.relationship(
        "EnvironmentalObservation",
        backref="monitoring_site",
        lazy=True,
        cascade="all, delete-orphan"
    )

    def __repr__(self):
        return f"<MonitoringSite {self.name}>"


# ============================================================
# ENVIRONMENTAL OBSERVATION
# ============================================================

class EnvironmentalObservation(db.Model):
    __tablename__ = "environmental_observations"

    id = db.Column(
        db.Integer,
        primary_key=True
    )

    # ========================================================
    # LAND SECTION
    # ========================================================

    monitoring_site_id = db.Column(
        db.Integer,
        db.ForeignKey("monitoring_sites.id"),
        nullable=False,
        index=True
    )

    location = db.Column(
        db.String(150),
        nullable=False
    )

    observation_date = db.Column(
        db.DateTime,
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        index=True
    )

    # ========================================================
    # AREA
    # ========================================================

    area_hectares = db.Column(
        db.Float,
        nullable=True
    )

    # ========================================================
    # WEATHER
    # ========================================================

    temperature = db.Column(
        db.Float,
        nullable=True
    )

    rainfall = db.Column(
        db.Float,
        nullable=True
    )

    # ========================================================
    # SOIL
    # ========================================================

    soil_ph = db.Column(db.Float)

    soil_moisture = db.Column(db.Float)

    soil_quality = db.Column(db.Float)

    # ========================================================
    # WATER
    # ========================================================

    water_ph = db.Column(db.Float)

    water_turbidity = db.Column(db.Float)

    water_quality = db.Column(db.Float)

    # ========================================================
    # VEGETATION
    # ========================================================

    vegetation_cover = db.Column(db.Float)

    vegetation_density = db.Column(db.Float)

    grass_availability = db.Column(db.Float)

    tree_density = db.Column(db.Float)

    # ========================================================
    # NOTES
    # ========================================================

    notes = db.Column(db.Text)

    # ========================================================
    # FIELD OFFICER
    # ========================================================

    recorded_by = db.Column(
        db.Integer,
        db.ForeignKey("details.id"),
        nullable=True
    )

    def __repr__(self):
        return f"<EnvironmentalObservation {self.id}>"


# ============================================================
# BIODIVERSITY OBSERVATION
# ============================================================

class Observation(db.Model):
    __tablename__ = "observation"

    id = db.Column(
        db.Integer,
        primary_key=True
    )

    # Species
    species_id = db.Column(
        db.Integer,
        db.ForeignKey("species.id"),
        nullable=False,
        index=True
    )

    # Environmental survey / land section
    environmental_observation_id = db.Column(
        db.Integer,
        db.ForeignKey("environmental_observations.id"),
        nullable=False,
        index=True
    )

    observation_date = db.Column(
        db.DateTime,
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
        index=True
    )

    population_count = db.Column(
        db.Integer,
        nullable=False
    )

    notes = db.Column(
        db.Text
    )

    photo = db.Column(
        db.String(555)
    )

    status = db.Column(
        db.String(20),
        default="Pending",
        index=True
    )

    reviewed_by = db.Column(
        db.Integer,
        db.ForeignKey("details.id"),
        nullable=True
    )

    reviewed_at = db.Column(
        db.DateTime,
        nullable=True
    )

    environmental_observation = db.relationship(
        "EnvironmentalObservation",
        backref=db.backref(
            "biodiversity_observations",
            lazy=True
        )
    )

    def __repr__(self):
        return f"<Observation {self.id}>"


# ============================================================
# ANALYSIS RESULT
# ============================================================

class Analysis(db.Model):
    __tablename__ = "analysis"

    id = db.Column(
        db.Integer,
        primary_key=True
    )

    species_id = db.Column(
        db.Integer,
        db.ForeignKey("species.id"),
        nullable=False
    )

    environmental_observation_id = db.Column(
        db.Integer,
        db.ForeignKey("environmental_observations.id"),
        nullable=True
    )

    monitoring_site_id = db.Column(
        db.Integer,
        db.ForeignKey("monitoring_sites.id"),
        nullable=True,
        index=True
    )

    current_population = db.Column(
        db.Integer,
        nullable=False
    )

    previous_population = db.Column(
        db.Integer,
        nullable=False
    )

    population_change = db.Column(
        db.Integer,
        nullable=False
    )

    percentage_change = db.Column(
        db.Float,
        nullable=False
    )

    trend = db.Column(
        db.String(50),
        nullable=False
    )

    risk_level = db.Column(
        db.String(50),
        nullable=False
    )

    analysis_date = db.Column(
        db.DateTime,
        default=lambda: datetime.now(timezone.utc)
    )

    def __repr__(self):
        return f"<Analysis {self.id}>"


# ============================================================
# NOTIFICATIONS
# ============================================================

class Notification(db.Model):
    __tablename__ = "notification"

    id = db.Column(
        db.Integer,
        primary_key=True
    )

    user_id = db.Column(
        db.Integer,
        db.ForeignKey("details.id"),
        nullable=True
    )

    role = db.Column(
        db.String(30),
        nullable=False
    )

    title = db.Column(
        db.String(150),
        nullable=False
    )

    message = db.Column(
        db.Text,
        nullable=False
    )

    notification_type = db.Column(
        db.String(20),
        default="Info"
    )

    is_read = db.Column(
        db.Boolean,
        default=False
    )

    created_at = db.Column(
        db.DateTime,
        default=lambda: datetime.now(timezone.utc)
    )

    def __repr__(self):
        return f"<Notification {self.id}>"