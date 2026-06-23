from flask import Flask, redirect, url_for, request, render_template, session
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timezone
from werkzeug.security import generate_password_hash
from werkzeug.security import check_password_hash
import os 
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle 
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.shapes import Drawing   
from functools import wraps



app = Flask(__name__)
app.secret_key = "my_super_secret_key_12345"

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///Details.db'      # default db
app.config['SQLALCHEMY_BINDS'] = {'species_db': 'sqlite:///Species.db'}

db = SQLAlchemy(app)
class Details(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    First_name = db.Column(db.String(50), nullable=False)
    surname = db.Column(db.String(50), nullable=False)
    email = db.Column(db.String(50), unique=True, nullable=False)
    phone = db.Column(db.String(20))
    DOB = db.Column(db.String(20))
    gender = db.Column(db.String(10))
    password = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(50), nullable=False, default='viewer')
    date_Created = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))

    def __repr__(self):
        return '<Name %r>' % self.id

#creating a database for the species to be observed
class Species(db.Model):
    __bind_key__ = 'species_db'
    id = db.Column(db.Integer, primary_key=True)
    scientificName = db.Column(db.String(50), nullable=False)
    specie_Common_Name = db.Column(db.String(50), nullable=False)
    specie_Habitat = db.Column(db.String(50), nullable=False, default='savannah')
    location = db.Column(db.String(50), nullable=False, default='CBU nature Park')

    # relationship
    observations = db.relationship('Observation', backref='species', lazy=True)

    def __repr__(self):
        return f"<Species {self.specie_Common_Name}>"
    
class Observation(db.Model):
    __bind_key__ = 'species_db'
    id = db.Column(db.Integer, primary_key=True)

    species_id = db.Column(db.Integer, db.ForeignKey('species.id'), nullable=False)

    observation_date = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    population_count = db.Column(db.Integer, nullable=False)

    notes = db.Column(db.String)
    photo = db.Column(db.String(555))

    def __repr__(self):
        return f"<Observation {self.id} - Species {self.species_id}>"

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
        
        if user_password == user_confirm_password: 
            user_password = generate_password_hash(request.form['password'])  
            new_user = Details( First_name = user_first_name, surname=user_surname,email=user_email, phone=user_phone, DOB=user_DOB, gender=user_gender,
            password=user_password)
            
        else:
            return "Passwords do not match."

        try:
            db.session.add(new_user)
            db.session.commit()

            return redirect(url_for('login'))

        except Exception as e:
            db.session.rollback()
            return f"Error: {e}"

    return render_template("registration.html")

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
            return redirect(url_for('home'))

        else:
                return "Unknown role"
    return render_template("login.html")

@app.route("/home")
def home():
    return render_template("homepage.html")


@app.route("/admin")
@login_required
@role_required("admin")
def admin():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    else:
            return render_template("admin.html")

@app.route("/manageUser")
@login_required
@role_required("admin")
def managerUser():
    users = Details.query.all()
    return render_template("manageUser.html", users=users)

@app.route("/change_role/<int:id>", methods=["POST"])
def change_role(id):
    user = Details.query.get_or_404(id)

    new_role = request.form['role']
    user.role = new_role

    db.session.commit()

    return redirect(url_for('managerUser'))

@app.route("/delete_user/<int:id>", methods=["POST"])
def delete_user(id):
    user = Details.query.get_or_404(id)

    db.session.delete(user)
    db.session.commit()

    return redirect(url_for('managerUser'))

@app.route("/field_Officer")
@login_required
@role_required("field_officer")
def field_Officer():
    return render_template("field_Officer.html")

@app.route("/record_observation", methods=["GET", "POST"])
@login_required
@role_required("field_officer")
def record_observation():
    species_list = Species.query.all()

    if request.method == "POST":
        species_id = request.form['species_id']
        population = request.form['population']
        notes = request.form['note']

        image = request.files.get('image')
        filename = None

        if image and image.filename:
            filename = image.filename
            image.save(os.path.join("static/uploads", filename))

        try:
            new_observation = Observation(
                species_id=int(species_id),
                population_count=int(population),
                notes=notes,
                photo=filename,
                observation_date=datetime.now(timezone.utc)
            )

            db.session.add(new_observation)
            db.session.commit()

            return redirect(url_for('view_observations'))

        except Exception as e:
            db.session.rollback()
            return f"Error: {e}"

    return render_template("record_observation.html", species_list=species_list)

@app.route("/view_observations")
@login_required
@role_required("field_officer","admin","viewer")
def view_observations():
    observations = Observation.query.order_by(Observation.observation_date.desc()).all()
    return render_template("view_observations.html", observations=observations)

@app.route("/view_species")
def view_species():
    species_list = Species.query.all()
    return render_template("view_species.html", species_list=species_list)     

@app.route("/report")
@login_required
@role_required("field_officer","admin","viewer")
def report():
    return render_template("report.html")

@app.route("/trends")
@login_required
@role_required("field_officer","admin","viewer")    
def trends():
    species_list = Species.query.all()

    trend_data = []
    for species in species_list:
        observations = Observation.query.filter_by(species_id=species.id)\
            .order_by(Observation.observation_date.asc()).all()
        
        trend_data.append({
            "name": species.specie_Common_Name,
            "dates": [obs.observation_date.strftime('%Y-%m-%d') for obs in observations],
            "counts": [obs.population_count for obs in observations]
        })

    return render_template("trends.html", species_list=species_list, trend_data=trend_data)

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True) 