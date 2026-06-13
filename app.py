from flask import Flask, redirect, url_for, request, render_template, session
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timezone
from werkzeug.security import generate_password_hash
from werkzeug.security import check_password_hash
import os 


app = Flask(__name__)
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
class Species (db.Model):
    id = db.Column(db.Integer, primary_key=True)
    specie_Scientific_Name = db.Column(db.String(50), nullable=False)
    specie_Common_Name = db.Column(db.String(50), nullable=False)
    specie_Habitat =db.Column(db.String(50), nullable=False)
    location =db.Column(db.String(50), nullable=False)
    date_Created = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    population_Count = db.Column(db.String(50),nullable=False)
    notes = db.Column(db.String)
    photo = db.Column(db.String (555))
    
    def __repr__(self):
        return '<Species %r>' % self.id
    
    
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
    

@app.route("/login", methods=['GET', 'POST'])
def login():

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

        return "Invalid email or password"

    return render_template("login.html")

@app.route("/home")
def home():
    return render_template("homepage.html")


@app.route("/admin")
def admin():
    return render_template("admin.html")

@app.route("/manageUser")
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
def field_Officer():
    return render_template("field_Officer.html")


@app.route("/Species", methods=["GET", "POST"])
def species():
    if request.method == "POST":
        specieScientificName = request.form['Specie_Scientific_Name']
        specieCommonName = request.form['Specie_Common_Name']
        speciePopulation = request.form['population']
        notes = request.form['note']
        specieImage = request.files['image']

        new_species = Species( specie_Scientific_Name=specieScientificName, specie_Common_Name=specieCommonName,
            population_Count=speciePopulation, notes=notes, photo=specieImage)

        try:
            db.session.add(new_species)
            db.session.commit()
            return redirect(url_for('species'))

        except Exception as e:
            db.session.rollback()
            return f"Error: {e}"

    return render_template('Species.html')

        

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)