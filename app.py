from flask import Flask
from flask_cors import CORS
import logging
from apis.recommendation import recommendation_bp
from apis.sentiment import sentiment_bp


def create_app():
    app = Flask(__name__)

    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    CORS(app, resources={r"/*": {"origins": "*"}})

    logger.debug("Registering blueprints")
    app.register_blueprint(recommendation_bp, url_prefix="/api/recommend")
    app.register_blueprint(sentiment_bp, url_prefix="/api/sentiment")

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(debug=True, host="0.0.0.0", port=5000)
