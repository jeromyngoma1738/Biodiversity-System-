from flask_mail import Mail, Message

mail = Mail()


def configure_mail(app):

    app.config['MAIL_SERVER'] = 'smtp.gmail.com'
    app.config['MAIL_PORT'] = 587
    app.config['MAIL_USE_TLS'] = True

    app.config['MAIL_USERNAME'] = "yourgmail@gmail.com"
    app.config['MAIL_PASSWORD'] = "your_app_password"

    mail.init_app(app)