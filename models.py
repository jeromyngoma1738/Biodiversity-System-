from datetime import datetime, timezone
from extensions import db

class Details(db.Model):
    __tablename__ = "details"

    id = db.Column(db.Integer, primary_key=True)
    First_name = db.Column(db.String(50), nullable=False)
    surname = db.Column(db.String(50), nullable=False)
    email = db.Column(db.String(50), unique=True, nullable=False)
    phone = db.Column(db.String(20))
    DOB = db.Column(db.String(20))
    gender = db.Column(db.String(10))
    password = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(50), nullable=False, default="viewer")
    date_Created = db.Column(db.DateTime,default=lambda: datetime.now(timezone.utc))
    reset_token = db.Column(db.String(200),nullable=True)
    reset_token_expiry = db.Column(db.DateTime,nullable=True)

    # Relationship with Observation
    reviewed_observations = db.relationship( "Observation",foreign_keys="Observation.reviewed_by",
        backref="reviewer", lazy=True
    )

    def __repr__(self):
        return f"<User {self.First_name} {self.surname}>"


# SPECIES TABLE
class Species(db.Model):
    __tablename__ = "species"

    id = db.Column(db.Integer, primary_key=True)
    scientificName = db.Column(db.String(50), nullable=False)
    specie_Common_Name = db.Column(db.String(50), nullable=False)
    specie_Habitat = db.Column(db.String(50), nullable=False, default="savannah")
    location = db.Column(db.String(50), nullable=False, default="CBU Nature Park")

    observations = db.relationship("Observation",backref="species",lazy=True,
        cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Species {self.specie_Common_Name}>"


# OBSERVATION TABLE
class Observation(db.Model):
    __tablename__ = "observation"

    id = db.Column(db.Integer, primary_key=True)
    species_id = db.Column(db.Integer,db.ForeignKey("species.id"), nullable=False)
    observation_date = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    population_count = db.Column(db.Integer, nullable=False)
    notes = db.Column(db.Text)
    photo = db.Column(db.String(555))
    status = db.Column(db.String(20), default="Pending")

    reviewed_by = db.Column(db.Integer, db.ForeignKey("details.id"),nullable=True)

    reviewed_at = db.Column(db.DateTime)

    def __repr__(self):
        return f"<Observation {self.id}>"



# NOTIFICATION TABLE

class Notification(db.Model):
    __tablename__ = "notification"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("details.id"), nullable=True)
    role = db.Column(db.String(30), nullable=False)
    title = db.Column(db.String(150), nullable=False)
    message = db.Column(db.Text, nullable=False)
    notification_type = db.Column( db.String(20), default="Info")
    is_read = db.Column(db.Boolean, default=False)
    created_at = db.Column( db.DateTime, default=lambda: datetime.now(timezone.utc))
class Analysis(db.Model):
    id = db.Column(db.Integer, primary_key=True)

    species_id = db.Column(db.Integer,db.ForeignKey("species.id"),nullable=False)

    current_population = db.Column(db.Integer, nullable=False)
    previous_population = db.Column(db.Integer, nullable=False)
    population_change = db.Column(db.Integer, nullable=False)
    percentage_change = db.Column(db.Float, nullable=False)
    trend = db.Column(db.String(50), nullable=False)
    risk_level = db.Column(db.String(50), nullable=False)

    analysis_date = db.Column(db.DateTime,default=lambda: datetime.now(timezone.utc))

    # Relationship
    species = db.relationship("Species",backref="analyses")