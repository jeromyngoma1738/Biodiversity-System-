
from flask import Flask, flash, redirect, url_for, request, render_template, session, send_file
from flask_sqlalchemy import SQLAlchemy
from extensions import db
from models import Details, Species, Observation, Notification, Analysis
from datetime import datetime, timezone
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
import os
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
import matplotlib.pyplot as plt
from reportlab.lib.units import inch
from sqlalchemy import or_
from mail_config import mail, configure_mail 
import secrets
from sqlalchemy import func
from datetime import datetime, timedelta
from datetime import datetime, timezone
from flask import Blueprint
app = Flask(__name__)

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
            if len(user_password) >= 8:
                user_password = generate_password_hash(request.form['password'])  
                new_user = Details( First_name = user_first_name, surname=user_surname,email=user_email, phone=user_phone, DOB=user_DOB, gender=user_gender,
                password=user_password)
                flash ("Password must be at least 8 characters long.")  
        else:
            return "Passwords do not match."

        try:
            db.session.add(new_user)
            db.session.commit()
            create_notification(role="admin", title="New User Registration",message=f"{user_first_name} {user_surname} has registered.",
                                notification_type="Info")
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
