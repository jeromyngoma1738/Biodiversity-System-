from flask import Flask, flash, redirect, url_for, request, render_template, session, send_file
from flask_sqlalchemy import SQLAlchemy
from extensions import db
from models import Details, Species, Observation, Notification, Analysis
from datetime import datetime, timezone
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
import os
from sqlalchemy import or_
from mail_config import mail, configure_mail 
import secrets
from sqlalchemy import func


app = Flask(__name__)

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


@app.route("/admin")
@login_required
@role_required("admin")
def admin():
    total_users = Details.query.filter_by(role="viewer").count()
    total_field_officers = Details.query.filter_by(role="field_officer").count()
    total_species = Species.query.count()
    total_notifications = Notification.query.filter_by(role="admin",  is_read=True).count()
    total_pending_observation=Observation.query.filter_by(status="Pending").count()

    return render_template("admin.html", total_users=total_users, total_field_officers=total_field_officers, total_species=total_species,
     total_notifications=total_notifications, total_pending_observation=total_pending_observation)

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
