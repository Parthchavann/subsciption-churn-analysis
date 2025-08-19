#!/usr/bin/env python3
"""
🌐 Enhanced Live Data Connector
Advanced real-time data integration with multiple external sources
"""

import asyncio
import aiohttp
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import time
import requests
from typing import Dict, List, Any, Optional
import websockets
import threading
from dataclasses import dataclass, asdict
import sqlite3
import os
from concurrent.futures import ThreadPoolExecutor
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MarketData:
    """Market data structure"""
    timestamp: datetime
    saas_index: float
    tech_stocks: float
    subscription_sentiment: float
    competition_score: float
    market_volatility: float

@dataclass
class CustomerBehavior:
    """Real-time customer behavior data"""
    timestamp: datetime
    active_sessions: int
    page_views: int
    feature_usage: Dict[str, int]
    support_tickets: int
    conversion_events: int

@dataclass
class BusinessMetrics:
    """Live business metrics"""
    timestamp: datetime
    revenue_today: float
    new_signups: int
    cancellations: int
    trial_starts: int
    upgrade_events: int
    churn_risk_score: float

class LiveMarketDataAPI:
    """Simulates live market data API"""
    
    def __init__(self):
        self.base_url = "https://api.marketdata.example.com"
        self.api_key = "demo_key_12345"
        
    async def get_saas_market_data(self) -> Dict[str, Any]:
        """Get SaaS market indicators"""
        # Simulate API call with realistic data
        await asyncio.sleep(0.1)  # Simulate network delay
        
        return {
            "saas_index": round(random.uniform(95, 105), 2),
            "tech_stocks": round(random.uniform(98, 103), 2),
            "subscription_sentiment": round(random.uniform(0.6, 0.9), 3),
            "competition_score": round(random.uniform(70, 85), 1),
            "market_volatility": round(random.uniform(0.1, 0.3), 3),
            "timestamp": datetime.now().isoformat()
        }
    
    async def get_competitor_data(self) -> Dict[str, Any]:
        """Get competitor pricing and performance data"""
        competitors = ["Netflix", "Spotify", "Adobe", "Salesforce", "Zoom"]
        
        competitor_data = {}
        for comp in competitors:
            competitor_data[comp] = {
                "pricing_change": round(random.uniform(-5, 5), 1),
                "feature_releases": random.randint(0, 3),
                "market_share": round(random.uniform(10, 25), 1),
                "customer_satisfaction": round(random.uniform(75, 95), 1)
            }
        
        await asyncio.sleep(0.2)
        return {
            "competitors": competitor_data,
            "timestamp": datetime.now().isoformat()
        }

class RealTimeEventsSimulator:
    """Simulates real-time customer events"""
    
    def __init__(self):
        self.event_types = [
            "login", "logout", "feature_use", "support_ticket", 
            "upgrade", "downgrade", "cancellation", "trial_start"
        ]
        
    def generate_event(self) -> Dict[str, Any]:
        """Generate a realistic customer event"""
        event_type = random.choice(self.event_types)
        
        base_event = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "user_id": random.randint(1, 1000),
            "session_id": f"sess_{random.randint(10000, 99999)}"
        }
        
        # Add event-specific data
        if event_type == "feature_use":
            base_event["feature"] = random.choice([
                "dashboard", "reports", "analytics", "export", "settings"
            ])
            base_event["duration"] = random.randint(30, 300)
            
        elif event_type == "support_ticket":
            base_event["priority"] = random.choice(["low", "medium", "high"])
            base_event["category"] = random.choice([
                "billing", "technical", "feature_request", "bug"
            ])
            
        elif event_type in ["upgrade", "downgrade"]:
            base_event["from_plan"] = random.choice(["basic", "premium", "enterprise"])
            base_event["to_plan"] = random.choice(["basic", "premium", "enterprise"])
            base_event["reason"] = random.choice([
                "feature_need", "cost_optimization", "usage_change"
            ])
            
        return base_event

class LiveDataAggregator:
    """Aggregates and processes live data from multiple sources"""
    
    def __init__(self):
        self.market_api = LiveMarketDataAPI()
        self.events_sim = RealTimeEventsSimulator()
        self.db_path = "data/live/live_metrics.db"
        self.ensure_db_exists()
        
    def ensure_db_exists(self):
        """Create database if it doesn't exist"""
        os.makedirs("data/live", exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS market_data (
                timestamp TEXT PRIMARY KEY,
                saas_index REAL,
                tech_stocks REAL,
                subscription_sentiment REAL,
                competition_score REAL,
                market_volatility REAL
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS customer_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                event_type TEXT,
                user_id INTEGER,
                session_id TEXT,
                event_data TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS business_metrics (
                timestamp TEXT PRIMARY KEY,
                revenue_today REAL,
                new_signups INTEGER,
                cancellations INTEGER,
                trial_starts INTEGER,
                upgrade_events INTEGER,
                churn_risk_score REAL
            )
        """)
        
        conn.commit()
        conn.close()
    
    async def collect_market_data(self):
        """Collect and store market data"""
        try:
            market_data = await self.market_api.get_saas_market_data()
            competitor_data = await self.market_api.get_competitor_data()
            
            # Store in database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO market_data 
                (timestamp, saas_index, tech_stocks, subscription_sentiment, 
                 competition_score, market_volatility)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                market_data["timestamp"],
                market_data["saas_index"],
                market_data["tech_stocks"],
                market_data["subscription_sentiment"],
                market_data["competition_score"],
                market_data["market_volatility"]
            ))
            
            conn.commit()
            conn.close()
            
            # Also save to JSON for dashboard
            combined_data = {
                "market_data": market_data,
                "competitor_data": competitor_data,
                "last_updated": datetime.now().isoformat()
            }
            
            with open("data/live/market_intelligence.json", "w") as f:
                json.dump(combined_data, f, indent=2)
                
            logger.info("Market data collected and stored")
            
        except Exception as e:
            logger.error(f"Error collecting market data: {e}")
    
    def collect_customer_events(self, num_events: int = 10):
        """Collect simulated customer events"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for _ in range(num_events):
                event = self.events_sim.generate_event()
                
                cursor.execute("""
                    INSERT INTO customer_events 
                    (timestamp, event_type, user_id, session_id, event_data)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    event["timestamp"],
                    event["event_type"],
                    event["user_id"],
                    event["session_id"],
                    json.dumps(event)
                ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Collected {num_events} customer events")
            
        except Exception as e:
            logger.error(f"Error collecting customer events: {e}")
    
    def calculate_live_business_metrics(self):
        """Calculate live business metrics"""
        try:
            # Simulate real-time business calculations
            now = datetime.now()
            
            metrics = BusinessMetrics(
                timestamp=now,
                revenue_today=random.uniform(8000, 12000),
                new_signups=random.randint(20, 40),
                cancellations=random.randint(5, 15),
                trial_starts=random.randint(15, 30),
                upgrade_events=random.randint(3, 8),
                churn_risk_score=random.uniform(12, 18)
            )
            
            # Store in database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO business_metrics 
                (timestamp, revenue_today, new_signups, cancellations, 
                 trial_starts, upgrade_events, churn_risk_score)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                metrics.timestamp.isoformat(),
                metrics.revenue_today,
                metrics.new_signups,
                metrics.cancellations,
                metrics.trial_starts,
                metrics.upgrade_events,
                metrics.churn_risk_score
            ))
            
            conn.commit()
            conn.close()
            
            # Save to JSON
            with open("data/live/business_metrics.json", "w") as f:
                json.dump(asdict(metrics), f, indent=2, default=str)
            
            logger.info("Business metrics calculated and stored")
            
        except Exception as e:
            logger.error(f"Error calculating business metrics: {e}")
    
    def get_live_dashboard_data(self) -> Dict[str, Any]:
        """Get aggregated data for dashboard"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Get recent market data
            market_df = pd.read_sql_query("""
                SELECT * FROM market_data 
                ORDER BY timestamp DESC LIMIT 10
            """, conn)
            
            # Get recent events summary
            events_summary = pd.read_sql_query("""
                SELECT event_type, COUNT(*) as count
                FROM customer_events 
                WHERE timestamp > datetime('now', '-1 hour')
                GROUP BY event_type
            """, conn)
            
            # Get latest business metrics
            business_metrics = pd.read_sql_query("""
                SELECT * FROM business_metrics 
                ORDER BY timestamp DESC LIMIT 1
            """, conn)
            
            conn.close()
            
            return {
                "market_trends": market_df.to_dict('records') if not market_df.empty else [],
                "recent_events": events_summary.to_dict('records') if not events_summary.empty else [],
                "current_metrics": business_metrics.to_dict('records')[0] if not business_metrics.empty else {},
                "last_updated": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting dashboard data: {e}")
            return {"error": str(e)}

class LiveDataStream:
    """Real-time data streaming service"""
    
    def __init__(self):
        self.aggregator = LiveDataAggregator()
        self.running = False
        
    async def start_data_collection(self):
        """Start continuous data collection"""
        self.running = True
        logger.info("Starting live data collection...")
        
        while self.running:
            try:
                # Collect market data every 30 seconds
                await self.aggregator.collect_market_data()
                
                # Collect customer events every 10 seconds
                self.aggregator.collect_customer_events(random.randint(5, 15))
                
                # Calculate business metrics every minute
                self.aggregator.calculate_live_business_metrics()
                
                # Wait before next collection
                await asyncio.sleep(10)
                
            except Exception as e:
                logger.error(f"Error in data collection loop: {e}")
                await asyncio.sleep(5)
    
    def stop_data_collection(self):
        """Stop data collection"""
        self.running = False
        logger.info("Stopping live data collection...")

# Webhook handler for external data
class WebhookHandler:
    """Handle incoming webhooks from external services"""
    
    @staticmethod
    def handle_payment_webhook(data: Dict[str, Any]):
        """Handle payment service webhooks"""
        event_type = data.get("type")
        
        if event_type == "payment.succeeded":
            # Process successful payment
            logger.info(f"Payment succeeded for customer {data.get('customer_id')}")
            
        elif event_type == "payment.failed":
            # Process failed payment
            logger.info(f"Payment failed for customer {data.get('customer_id')}")
            
    @staticmethod
    def handle_support_webhook(data: Dict[str, Any]):
        """Handle support system webhooks"""
        ticket_data = data.get("ticket", {})
        priority = ticket_data.get("priority", "low")
        
        if priority == "high":
            logger.warning(f"High priority support ticket: {ticket_data.get('id')}")

def create_live_data_dashboard():
    """Create live data visualization for dashboard"""
    aggregator = LiveDataAggregator()
    dashboard_data = aggregator.get_live_dashboard_data()
    
    # Generate live charts data
    live_charts = {
        "market_trends": {
            "labels": [f"T-{i}" for i in range(10, 0, -1)],
            "saas_index": [random.uniform(95, 105) for _ in range(10)],
            "competition": [random.uniform(70, 85) for _ in range(10)]
        },
        "event_stream": {
            "timestamps": [(datetime.now() - timedelta(minutes=i)).isoformat() 
                          for i in range(60, 0, -5)],
            "login_events": [random.randint(5, 20) for _ in range(12)],
            "feature_usage": [random.randint(10, 40) for _ in range(12)]
        },
        "revenue_pulse": {
            "hourly_revenue": [random.uniform(400, 800) for _ in range(24)],
            "conversion_rate": [random.uniform(2, 8) for _ in range(24)]
        }
    }
    
    # Save live charts data
    with open("data/live/live_charts.json", "w") as f:
        json.dump(live_charts, f, indent=2)
    
    return live_charts

async def main():
    """Main function to run live data collection"""
    stream = LiveDataStream()
    
    try:
        # Start data collection
        await stream.start_data_collection()
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    finally:
        stream.stop_data_collection()

if __name__ == "__main__":
    # Run data collection
    asyncio.run(main())