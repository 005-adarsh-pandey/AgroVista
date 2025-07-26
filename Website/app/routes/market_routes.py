"""
Market Routes Module for AgroVista
Handles market data routes and API endpoints
"""

from flask import Blueprint, render_template, request, jsonify
import logging
from datetime import datetime
import sys
import os

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from app.services.market_service import market_service

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create blueprint
market_bp = Blueprint('market', __name__)

@market_bp.route('/market')
def market_page():
    """Market data page"""
    try:
        return render_template('market.html')
    except Exception as e:
        logger.error(f"❌ Error in market route: {e}")
        return "Error loading market page", 500

@market_bp.route('/api/market/msp/<crop_name>')
def get_msp_data(crop_name):
    """Get MSP data for a specific crop"""
    try:
        msp_data = market_service.get_msp_data(crop_name)
        
        if msp_data:
            return jsonify({
                'success': True,
                'data': msp_data,
                'crop': crop_name,
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify({
                'success': False,
                'message': f'MSP data not found for {crop_name}',
                'crop': crop_name
            }), 404
            
    except Exception as e:
        logger.error(f"❌ Error getting MSP data for {crop_name}: {e}")
        return jsonify({
            'success': False,
            'error': 'MSP data service unavailable'
        }), 500

@market_bp.route('/api/market/mandi')
def get_mandi_prices():
    """Get current mandi prices"""
    try:
        state = request.args.get('state')
        limit = int(request.args.get('limit', 100))
        
        # Limit the number of records
        limit = max(1, min(limit, 500))
        
        # Try to get data from API first, then fallback to local files
        mandi_data = market_service.get_mandi_prices(state, limit)
        
        if not mandi_data:
            # Fallback to local data
            mandi_data = market_service.get_local_mandi_data(limit)
        
        return jsonify({
            'success': True,
            'data': mandi_data,
            'count': len(mandi_data),
            'state_filter': state,
            'timestamp': datetime.now().isoformat()
        })
        
    except ValueError:
        return jsonify({
            'success': False,
            'error': 'Invalid limit parameter'
        }), 400
        
    except Exception as e:
        logger.error(f"❌ Error getting mandi prices: {e}")
        return jsonify({
            'success': False,
            'error': 'Mandi price service unavailable'
        }), 500

@market_bp.route('/api/market/trends/<crop_name>')
def get_price_trends(crop_name):
    """Get price trends for a crop"""
    try:
        days = int(request.args.get('days', 30))
        days = max(7, min(days, 365))  # Limit between 7 days and 1 year
        
        trend_data = market_service.get_price_trends(crop_name, days)
        
        return jsonify({
            'success': True,
            'data': trend_data,
            'crop': crop_name,
            'period_days': days,
            'timestamp': datetime.now().isoformat()
        })
        
    except ValueError:
        return jsonify({
            'success': False,
            'error': 'Invalid days parameter'
        }), 400
        
    except Exception as e:
        logger.error(f"❌ Error getting price trends for {crop_name}: {e}")
        return jsonify({
            'success': False,
            'error': 'Price trends service unavailable'
        }), 500

@market_bp.route('/api/market/summary')
def market_summary():
    """Get market summary with key statistics"""
    try:
        # Get recent mandi data for summary
        recent_data = market_service.get_mandi_prices(limit=50)
        
        if not recent_data:
            recent_data = market_service.get_local_mandi_data(50)
        
        # Calculate basic statistics
        summary = {
            'total_markets': len(set(item.get('market', '') for item in recent_data)),
            'total_commodities': len(set(item.get('commodity', '') for item in recent_data)),
            'states_covered': len(set(item.get('state', '') for item in recent_data)),
            'last_updated': recent_data[0].get('price_date', 'N/A') if recent_data else 'N/A'
        }
        
        # Top commodities by frequency
        commodity_count = {}
        for item in recent_data:
            commodity = item.get('commodity', 'Unknown')
            commodity_count[commodity] = commodity_count.get(commodity, 0) + 1
        
        top_commodities = sorted(commodity_count.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return jsonify({
            'success': True,
            'summary': summary,
            'top_commodities': [{'name': name, 'count': count} for name, count in top_commodities],
            'data_source': 'API' if recent_data else 'Local',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Error getting market summary: {e}")
        return jsonify({
            'success': False,
            'error': 'Market summary service unavailable'
        }), 500

@market_bp.route('/api/market/search')
def search_market_data():
    """Search market data by commodity, state, or district"""
    try:
        query = request.args.get('q', '').strip().lower()
        search_type = request.args.get('type', 'commodity')  # commodity, state, district
        limit = int(request.args.get('limit', 50))
        
        if not query:
            return jsonify({
                'success': False,
                'error': 'Search query is required'
            }), 400
        
        # Get market data
        market_data = market_service.get_mandi_prices(limit=200)
        
        if not market_data:
            market_data = market_service.get_local_mandi_data(200)
        
        # Filter based on search criteria
        filtered_data = []
        for item in market_data:
            search_field = item.get(search_type, '').lower()
            if query in search_field:
                filtered_data.append(item)
                
                if len(filtered_data) >= limit:
                    break
        
        return jsonify({
            'success': True,
            'data': filtered_data[:limit],
            'query': query,
            'search_type': search_type,
            'results_count': len(filtered_data),
            'timestamp': datetime.now().isoformat()
        })
        
    except ValueError:
        return jsonify({
            'success': False,
            'error': 'Invalid limit parameter'
        }), 400
        
    except Exception as e:
        logger.error(f"❌ Error searching market data: {e}")
        return jsonify({
            'success': False,
            'error': 'Market search service unavailable'
        }), 500
