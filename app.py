from flask import Flask, redirect, url_for, request, render_template, session, send_file
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timezone
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
import os
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
import matplotlib.pyplot as plt
from reportlab.lib.units import inch


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


class Notification(db.Model):
    id = db.Column(db.Integer, primary_key=True)

    user_id = db.Column(db.Integer, db.ForeignKey('details.id'), nullable=True)

    role = db.Column(db.String(30), nullable=False)

    title = db.Column(db.String(150), nullable=False)

    message = db.Column(db.Text, nullable=False)

    notification_type = db.Column(db.String(20), default="Info")
    # Info, Success, Warning, Critical

    is_read = db.Column(db.Boolean, default=False)

    created_at = db.Column(db.DateTime,
                           default=lambda: datetime.now(timezone.utc))
    
def create_notification(role, title, message,notification_type="Info", user_id=None):

    notification = Notification(role=role,user_id=user_id, title=title,message=message,notification_type=notification_type )

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
        
        if user_password == user_confirm_password: 
            user_password = generate_password_hash(request.form['password'])  
            new_user = Details( First_name = user_first_name, surname=user_surname,email=user_email, phone=user_phone, DOB=user_DOB, gender=user_gender,
            password=user_password)
            
        else:
            return "Passwords do not match."

        try:
            db.session.add(new_user)
            db.session.commit()
            create_notification(role="admin", title="New User Registration",message=f"{user_first_name} {user_surname} has registered.",
                                notification_type="Info")

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
            return redirect(url_for('user'))

        else:
                return "Unknown role"
    return render_template("login.html")

@app.route("/user")
def user():
    return render_template("user.html") 

@app.route("/admin")
@login_required
@role_required("admin")
def admin():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    else:
            total_users = Details.query.filter_by(role="viewer").count()
            total_field_officers = Details.query.filter_by(role="field_officer").count()
            total_species = Species.query.count()
            total_notifications = Notification.query.count()
            return render_template( "admin.html", total_users=total_users, total_field_officers=total_field_officers, total_species=total_species, total_notifications=total_notifications)

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
def change_role(id):
    user = Details.query.get_or_404(id)

    new_role = request.form['role']
    user.role = new_role

    db.session.commit()
    create_notification( role=new_role, user_id=user.id, title="Role Updated",
    message=f"Your role has been changed to {new_role}.",notification_type="Success")

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
            new_observation = Observation(species_id=int(species_id), population_count=int(population), 
                            notes=notes,photo=filename,observation_date=datetime.now(timezone.utc) )

            db.session.add(new_observation)
            db.session.commit()
            # Notify the field officer
            create_notification(role="field_officer", user_id=session["user_id"],
                title="Observation Recorded",
                message="Your observation has been successfully recorded.",
                notification_type="Success")
            # Notify all admins
            create_notification(role="admin",title="New Observation Submitted",
                                message=f"{session['user_name']} submitted a new observation.",notification_type="Info")

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
@login_required
@role_required("field_officer","admin","viewer")
def view_species():
    species_list = Species.query.all()
    return render_template("view_species.html", species_list=species_list)     

@app.route("/report")
@login_required
@role_required("field_officer","admin","viewer")
def report():

    species_list = Species.query.all()

    report_data = []

    for species in species_list:

        observations = Observation.query.filter_by(species_id=species.id).order_by( Observation.observation_date.asc()).all()

        chart_file = f"charts/chart_{species.id}.png"
        chart_path = os.path.join("static", chart_file)

        report_data.append({"name": species.specie_Common_Name, "scientific": species.scientificName, "observations": observations,
            "chart": chart_file if os.path.exists(chart_path) else None})
        
    return render_template("report.html", report_data=report_data)

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

        # ---------------------------
        # collect data for GRAPH
        # ---------------------------
        dates = []
        counts = []

        for obs in observations:

            dates.append(obs.observation_date.strftime('%Y-%m-%d'))
            counts.append(obs.population_count)

            story.append(
                Paragraph(
                    f"Date: {obs.observation_date.strftime('%Y-%m-%d %H:%M')}",
                    styles["Normal"]
                )
            )

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
        (Notification.role == session["role"]) |
        (Notification.user_id == session["user_id"])
    ).order_by(Notification.created_at.desc()).all()

    return render_template("notifications.html", notifications=notifications)

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

    create_notification(
        role="admin",
        title="Species Deleted",
        message=f"{session['user_name']} removed {species.specie_Common_Name}.",
        notification_type="Warning"
    )

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

        new_species = Species(
            scientificName=scientific_name,
            specie_Common_Name=common_name,
            specie_Habitat=habitat,
            location=location
        )

        db.session.add(new_species)
        db.session.commit()

        return redirect(url_for("view_species"))

    return render_template("add_species.html")
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True) 