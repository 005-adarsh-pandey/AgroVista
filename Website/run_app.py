# Run the Flask app in the background
from app import app

if __name__ == "__main__":
    # app.run(debug=True)
    host = os.environ.get('HOST', '0.0.0.0')
    try:
        port = int(os.environ.get('PORT', 5000))
    except (TypeError, ValueError):
        port = 5000
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() in ('1', 'true', 'yes')

    app.run(host=host, port=port, debug=debug)
