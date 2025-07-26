"""
Main Routes Module for AgroVista
Handles home, about, and navigation routes
"""

from flask import Blueprint, render_template, request, jsonify
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create blueprint
main_bp = Blueprint('main', __name__)

@main_bp.route('/')
def home():
    """Home page route"""
    try:
        return render_template('home.html')
    except Exception as e:
        logger.error(f"❌ Error in home route: {e}")
        return "Error loading home page", 500

@main_bp.route('/projects')
def projects():
    """Projects page route"""
    try:
        return render_template('projects.html')
    except Exception as e:
        logger.error(f"❌ Error in projects route: {e}")
        return "Error loading projects page", 500

@main_bp.route('/projects2')
def projects2():
    """Secondary projects page route"""
    try:
        return render_template('projects2.html')
    except Exception as e:
        logger.error(f"❌ Error in projects2 route: {e}")
        return "Error loading projects2 page", 500

@main_bp.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'AgroVista',
        'timestamp': datetime.now().isoformat()
    })

@main_bp.route('/api/status')
def api_status():
    """API status endpoint"""
    return jsonify({
        'api_version': '1.0',
        'status': 'operational',
        'endpoints': [
            '/predict',
            '/weather',
            '/market'
        ]
    })
