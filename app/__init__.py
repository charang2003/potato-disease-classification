from flask_app import Flask

app = Flask(__name__)

from app import routes
