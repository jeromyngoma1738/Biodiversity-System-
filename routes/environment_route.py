from flask import (
    Blueprint,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    session
)

from extensions import db
from models import EnvironmentalObservation

from app.decorators import (
    login_required,
    role_required
)

environment_bp = Blueprint(
    "environment",
    __name__
)