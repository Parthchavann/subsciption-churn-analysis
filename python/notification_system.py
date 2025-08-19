#!/usr/bin/env python3
"""
🚨 Advanced Notification & Alert System
Real-time notifications with multiple channels and intelligent alerting
"""

import smtplib
import json
import requests
import asyncio
import websockets
from datetime import datetime, timedelta
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import logging
from enum import Enum
import sqlite3
import os
from jinja2 import Template
import threading
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AlertChannel(Enum):
    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    SMS = "sms"
    BROWSER = "browser"

@dataclass
class Alert:
    """Alert data structure"""
    id: str
    title: str
    message: str
    severity: AlertSeverity
    metric: str
    current_value: float
    threshold_value: float
    timestamp: datetime
    resolved: bool = False
    channels: List[AlertChannel] = None
    metadata: Dict[str, Any] = None

@dataclass
class NotificationRule:
    """Notification rule configuration"""
    id: str
    name: str
    metric: str
    condition: str  # "gt", "lt", "eq", "change_pct"
    threshold: float
    severity: AlertSeverity
    channels: List[AlertChannel]
    enabled: bool = True
    cooldown_minutes: int = 30
    last_triggered: Optional[datetime] = None

class EmailNotifier:
    """Email notification handler"""
    
    def __init__(self, smtp_host: str = "smtp.gmail.com", smtp_port: int = 587):
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.username = os.getenv("EMAIL_USERNAME", "your-email@example.com")
        self.password = os.getenv("EMAIL_PASSWORD", "your-app-password")
        
    async def send_alert_email(self, alert: Alert, recipients: List[str]):
        """Send alert via email"""
        try:
            # Create email template
            template = Template("""
            <html>
            <body>
                <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
                    <div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px 10px 0 0;">
                        <h1 style="margin: 0;">🚨 {{ alert.severity.value.upper() }} ALERT</h1>
                        <h2 style="margin: 10px 0 0 0;">{{ alert.title }}</h2>
                    </div>
                    
                    <div style="background: #f8f9fa; padding: 20px; border-radius: 0 0 10px 10px; border: 1px solid #dee2e6;">
                        <div style="background: white; padding: 15px; border-radius: 5px; margin-bottom: 15px;">
                            <h3 style="color: #333; margin-top: 0;">Alert Details</h3>
                            <p><strong>Metric:</strong> {{ alert.metric }}</p>
                            <p><strong>Current Value:</strong> {{ alert.current_value }}</p>
                            <p><strong>Threshold:</strong> {{ alert.threshold_value }}</p>
                            <p><strong>Time:</strong> {{ alert.timestamp.strftime('%Y-%m-%d %H:%M:%S') }}</p>
                        </div>
                        
                        <div style="background: white; padding: 15px; border-radius: 5px;">
                            <h3 style="color: #333; margin-top: 0;">Description</h3>
                            <p>{{ alert.message }}</p>
                        </div>
                        
                        {% if alert.metadata %}
                        <div style="background: white; padding: 15px; border-radius: 5px; margin-top: 15px;">
                            <h3 style="color: #333; margin-top: 0;">Additional Information</h3>
                            {% for key, value in alert.metadata.items() %}
                            <p><strong>{{ key }}:</strong> {{ value }}</p>
                            {% endfor %}
                        </div>
                        {% endif %}
                        
                        <div style="text-align: center; margin-top: 20px;">
                            <a href="http://localhost:8081/dashboard.html" 
                               style="background: #667eea; color: white; padding: 12px 24px; text-decoration: none; border-radius: 5px; display: inline-block;">
                                View Dashboard
                            </a>
                        </div>
                    </div>
                </div>
            </body>
            </html>
            """)
            
            html_content = template.render(alert=alert)
            
            # Create email message
            msg = MimeMultipart('alternative')
            msg['Subject'] = f"[ALERT] {alert.title}"
            msg['From'] = self.username
            msg['To'] = ", ".join(recipients)
            
            # Add HTML content
            html_part = MimeText(html_content, 'html')
            msg.attach(html_part)
            
            # Send email (simulated for demo)
            logger.info(f"📧 Email alert sent to {recipients}: {alert.title}")
            
            # In production, uncomment below:
            # with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
            #     server.starttls()
            #     server.login(self.username, self.password)
            #     server.send_message(msg)
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")

class SlackNotifier:
    """Slack notification handler"""
    
    def __init__(self, webhook_url: str = None):
        self.webhook_url = webhook_url or os.getenv("SLACK_WEBHOOK_URL")
        
    async def send_alert_slack(self, alert: Alert):
        """Send alert to Slack"""
        try:
            # Determine color based on severity
            color_map = {
                AlertSeverity.LOW: "#36a64f",      # Green
                AlertSeverity.MEDIUM: "#ff9500",   # Orange  
                AlertSeverity.HIGH: "#ff0000",     # Red
                AlertSeverity.CRITICAL: "#8b0000"  # Dark Red
            }
            
            # Determine emoji based on severity
            emoji_map = {
                AlertSeverity.LOW: "ℹ️",
                AlertSeverity.MEDIUM: "⚠️",
                AlertSeverity.HIGH: "🚨",
                AlertSeverity.CRITICAL: "🔥"
            }
            
            payload = {
                "attachments": [
                    {
                        "color": color_map[alert.severity],
                        "pretext": f"{emoji_map[alert.severity]} *{alert.severity.value.upper()} ALERT*",
                        "title": alert.title,
                        "text": alert.message,
                        "fields": [
                            {
                                "title": "Metric",
                                "value": alert.metric,
                                "short": True
                            },
                            {
                                "title": "Current Value",
                                "value": str(alert.current_value),
                                "short": True
                            },
                            {
                                "title": "Threshold",
                                "value": str(alert.threshold_value),
                                "short": True
                            },
                            {
                                "title": "Time",
                                "value": alert.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                                "short": True
                            }
                        ],
                        "actions": [
                            {
                                "type": "button",
                                "text": "View Dashboard",
                                "url": "http://localhost:8081/dashboard.html"
                            }
                        ],
                        "footer": "Subscription Analytics",
                        "ts": int(alert.timestamp.timestamp())
                    }
                ]
            }
            
            # Simulate Slack notification (replace with actual webhook call)
            logger.info(f"💬 Slack alert sent: {alert.title}")
            
            # In production, uncomment below:
            # if self.webhook_url:
            #     response = requests.post(self.webhook_url, json=payload)
            #     if response.status_code != 200:
            #         raise Exception(f"Slack API error: {response.status_code}")
            
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")

class WebhookNotifier:
    """Generic webhook notification handler"""
    
    def __init__(self, webhook_urls: List[str] = None):
        self.webhook_urls = webhook_urls or []
        
    async def send_webhook_alert(self, alert: Alert):
        """Send alert via webhook"""
        try:
            payload = {
                "event": "alert_triggered",
                "alert": asdict(alert),
                "timestamp": alert.timestamp.isoformat()
            }
            
            for url in self.webhook_urls:
                # Simulate webhook call
                logger.info(f"🔗 Webhook alert sent to {url}: {alert.title}")
                
                # In production, uncomment below:
                # response = requests.post(url, json=payload, timeout=10)
                # if response.status_code not in [200, 201, 202]:
                #     logger.warning(f"Webhook failed for {url}: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")

class BrowserNotifier:
    """Browser push notification handler"""
    
    def __init__(self):
        self.active_connections = set()
        
    async def send_browser_alert(self, alert: Alert):
        """Send real-time browser notification"""
        try:
            notification_data = {
                "type": "alert",
                "data": {
                    "id": alert.id,
                    "title": alert.title,
                    "message": alert.message,
                    "severity": alert.severity.value,
                    "timestamp": alert.timestamp.isoformat(),
                    "icon": self._get_severity_icon(alert.severity)
                }
            }
            
            # Save to file for browser to pick up
            os.makedirs("data/live", exist_ok=True)
            with open("data/live/browser_notifications.json", "a") as f:
                f.write(json.dumps(notification_data) + "\n")
            
            logger.info(f"🌐 Browser alert queued: {alert.title}")
            
        except Exception as e:
            logger.error(f"Failed to send browser alert: {e}")
    
    def _get_severity_icon(self, severity: AlertSeverity) -> str:
        """Get icon for severity level"""
        icons = {
            AlertSeverity.LOW: "ℹ️",
            AlertSeverity.MEDIUM: "⚠️", 
            AlertSeverity.HIGH: "🚨",
            AlertSeverity.CRITICAL: "🔥"
        }
        return icons.get(severity, "📢")

class AlertEngine:
    """Main alert processing engine"""
    
    def __init__(self):
        self.rules: List[NotificationRule] = []
        self.active_alerts: Dict[str, Alert] = {}
        self.db_path = "data/live/alerts.db"
        
        # Initialize notifiers
        self.email_notifier = EmailNotifier()
        self.slack_notifier = SlackNotifier()
        self.webhook_notifier = WebhookNotifier()
        self.browser_notifier = BrowserNotifier()
        
        self.ensure_db_exists()
        self.load_default_rules()
        
    def ensure_db_exists(self):
        """Create alerts database if it doesn't exist"""
        os.makedirs("data/live", exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                id TEXT PRIMARY KEY,
                title TEXT,
                message TEXT,
                severity TEXT,
                metric TEXT,
                current_value REAL,
                threshold_value REAL,
                timestamp TEXT,
                resolved INTEGER,
                metadata TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS notification_rules (
                id TEXT PRIMARY KEY,
                name TEXT,
                metric TEXT,
                condition TEXT,
                threshold REAL,
                severity TEXT,
                channels TEXT,
                enabled INTEGER,
                cooldown_minutes INTEGER,
                last_triggered TEXT
            )
        """)
        
        conn.commit()
        conn.close()
    
    def load_default_rules(self):
        """Load default notification rules"""
        default_rules = [
            NotificationRule(
                id="churn_rate_high",
                name="High Churn Rate Alert",
                metric="churn_rate",
                condition="gt",
                threshold=18.0,
                severity=AlertSeverity.HIGH,
                channels=[AlertChannel.EMAIL, AlertChannel.SLACK, AlertChannel.BROWSER],
                cooldown_minutes=60
            ),
            NotificationRule(
                id="revenue_drop",
                name="Revenue Drop Alert", 
                metric="monthly_revenue",
                condition="change_pct",
                threshold=-10.0,
                severity=AlertSeverity.MEDIUM,
                channels=[AlertChannel.EMAIL, AlertChannel.BROWSER],
                cooldown_minutes=30
            ),
            NotificationRule(
                id="critical_churn",
                name="Critical Churn Rate",
                metric="churn_rate",
                condition="gt",
                threshold=25.0,
                severity=AlertSeverity.CRITICAL,
                channels=[AlertChannel.EMAIL, AlertChannel.SLACK, AlertChannel.WEBHOOK, AlertChannel.BROWSER],
                cooldown_minutes=15
            ),
            NotificationRule(
                id="low_arpu",
                name="Low ARPU Warning",
                metric="arpu",
                condition="lt", 
                threshold=15.0,
                severity=AlertSeverity.LOW,
                channels=[AlertChannel.BROWSER],
                cooldown_minutes=120
            ),
            NotificationRule(
                id="user_growth_negative",
                name="Negative User Growth",
                metric="active_users",
                condition="change_pct",
                threshold=-5.0,
                severity=AlertSeverity.MEDIUM,
                channels=[AlertChannel.EMAIL, AlertChannel.BROWSER],
                cooldown_minutes=60
            )
        ]
        
        self.rules = default_rules
        self.save_rules_to_db()
    
    def save_rules_to_db(self):
        """Save notification rules to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for rule in self.rules:
                cursor.execute("""
                    INSERT OR REPLACE INTO notification_rules 
                    (id, name, metric, condition, threshold, severity, channels, enabled, cooldown_minutes, last_triggered)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    rule.id,
                    rule.name,
                    rule.metric,
                    rule.condition,
                    rule.threshold,
                    rule.severity.value,
                    json.dumps([c.value for c in rule.channels]),
                    1 if rule.enabled else 0,
                    rule.cooldown_minutes,
                    rule.last_triggered.isoformat() if rule.last_triggered else None
                ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to save rules to database: {e}")
    
    async def evaluate_metrics(self, metrics: Dict[str, float]):
        """Evaluate metrics against alert rules"""
        for rule in self.rules:
            if not rule.enabled:
                continue
                
            # Check cooldown
            if rule.last_triggered:
                time_since_last = datetime.now() - rule.last_triggered
                if time_since_last < timedelta(minutes=rule.cooldown_minutes):
                    continue
            
            metric_value = metrics.get(rule.metric)
            if metric_value is None:
                continue
                
            # Evaluate condition
            should_alert = False
            
            if rule.condition == "gt" and metric_value > rule.threshold:
                should_alert = True
            elif rule.condition == "lt" and metric_value < rule.threshold:
                should_alert = True
            elif rule.condition == "eq" and abs(metric_value - rule.threshold) < 0.01:
                should_alert = True
            elif rule.condition == "change_pct":
                # Calculate percentage change (simplified)
                previous_value = metrics.get(f"{rule.metric}_previous", metric_value)
                if previous_value > 0:
                    change_pct = ((metric_value - previous_value) / previous_value) * 100
                    if abs(change_pct) >= abs(rule.threshold):
                        should_alert = True
            
            if should_alert:
                await self.trigger_alert(rule, metric_value, metrics)
    
    async def trigger_alert(self, rule: NotificationRule, current_value: float, all_metrics: Dict[str, float]):
        """Trigger an alert"""
        try:
            alert_id = f"{rule.id}_{int(datetime.now().timestamp())}"
            
            alert = Alert(
                id=alert_id,
                title=rule.name,
                message=f"{rule.metric} is {current_value:.2f}, which exceeds threshold of {rule.threshold:.2f}",
                severity=rule.severity,
                metric=rule.metric,
                current_value=current_value,
                threshold_value=rule.threshold,
                timestamp=datetime.now(),
                channels=rule.channels,
                metadata={
                    "rule_id": rule.id,
                    "all_metrics": all_metrics
                }
            )
            
            # Save alert to database
            self.save_alert_to_db(alert)
            
            # Send notifications
            await self.send_notifications(alert)
            
            # Update rule last triggered time
            rule.last_triggered = datetime.now()
            self.save_rules_to_db()
            
            # Add to active alerts
            self.active_alerts[alert_id] = alert
            
            logger.info(f"🚨 Alert triggered: {alert.title}")
            
        except Exception as e:
            logger.error(f"Failed to trigger alert: {e}")
    
    def save_alert_to_db(self, alert: Alert):
        """Save alert to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO alerts 
                (id, title, message, severity, metric, current_value, threshold_value, timestamp, resolved, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                alert.id,
                alert.title,
                alert.message,
                alert.severity.value,
                alert.metric,
                alert.current_value,
                alert.threshold_value,
                alert.timestamp.isoformat(),
                0,
                json.dumps(alert.metadata or {})
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to save alert to database: {e}")
    
    async def send_notifications(self, alert: Alert):
        """Send notifications through specified channels"""
        tasks = []
        
        for channel in alert.channels:
            if channel == AlertChannel.EMAIL:
                recipients = ["admin@example.com", "alerts@example.com"]
                tasks.append(self.email_notifier.send_alert_email(alert, recipients))
            elif channel == AlertChannel.SLACK:
                tasks.append(self.slack_notifier.send_alert_slack(alert))
            elif channel == AlertChannel.WEBHOOK:
                tasks.append(self.webhook_notifier.send_webhook_alert(alert))
            elif channel == AlertChannel.BROWSER:
                tasks.append(self.browser_notifier.send_browser_alert(alert))
        
        # Send all notifications concurrently
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    def get_active_alerts(self) -> List[Alert]:
        """Get currently active alerts"""
        return list(self.active_alerts.values())
    
    def resolve_alert(self, alert_id: str):
        """Mark an alert as resolved"""
        if alert_id in self.active_alerts:
            self.active_alerts[alert_id].resolved = True
            
            # Update in database
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute("UPDATE alerts SET resolved = 1 WHERE id = ?", (alert_id,))
                conn.commit()
                conn.close()
                
                logger.info(f"✅ Alert resolved: {alert_id}")
                
            except Exception as e:
                logger.error(f"Failed to resolve alert in database: {e}")

class LiveMetricsMonitor:
    """Continuous metrics monitoring service"""
    
    def __init__(self):
        self.alert_engine = AlertEngine()
        self.running = False
        
    async def start_monitoring(self):
        """Start continuous monitoring"""
        self.running = True
        logger.info("🔍 Starting live metrics monitoring...")
        
        while self.running:
            try:
                # Get current metrics (simulate live data)
                metrics = self.get_current_metrics()
                
                # Evaluate against alert rules
                await self.alert_engine.evaluate_metrics(metrics)
                
                # Wait before next check
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(10)
    
    def get_current_metrics(self) -> Dict[str, float]:
        """Get current metrics (simulated)"""
        import random
        
        # Simulate realistic metric variations
        base_metrics = {
            "churn_rate": 14.9,
            "monthly_revenue": 10887,
            "arpu": 18.05,
            "active_users": 1000
        }
        
        # Add random variations that might trigger alerts
        current_metrics = {}
        for metric, base_value in base_metrics.items():
            variation = random.uniform(-0.2, 0.2)  # ±20% variation
            current_metrics[metric] = base_value * (1 + variation)
            
            # Occasionally simulate alert conditions
            if random.random() < 0.1:  # 10% chance of alert condition
                if metric == "churn_rate":
                    current_metrics[metric] = random.uniform(19, 26)  # High churn
                elif metric == "monthly_revenue":
                    current_metrics[metric] = base_value * 0.85  # Revenue drop
                elif metric == "arpu":
                    current_metrics[metric] = random.uniform(12, 14)  # Low ARPU
        
        return current_metrics
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.running = False
        logger.info("⏹️ Stopping live metrics monitoring...")

# Webhook server for external integrations
class WebhookServer:
    """Webhook server for receiving external alerts"""
    
    def __init__(self, port: int = 8082):
        self.port = port
        self.alert_engine = AlertEngine()
    
    async def handle_webhook(self, websocket, path):
        """Handle incoming webhook"""
        try:
            async for message in websocket:
                data = json.loads(message)
                
                # Process external alert
                if data.get("type") == "external_alert":
                    await self.process_external_alert(data)
                    
        except Exception as e:
            logger.error(f"Webhook handling error: {e}")
    
    async def process_external_alert(self, data: Dict[str, Any]):
        """Process external alert data"""
        try:
            alert = Alert(
                id=f"external_{int(datetime.now().timestamp())}",
                title=data.get("title", "External Alert"),
                message=data.get("message", "External system alert"),
                severity=AlertSeverity(data.get("severity", "medium")),
                metric=data.get("metric", "external"),
                current_value=data.get("current_value", 0),
                threshold_value=data.get("threshold_value", 0),
                timestamp=datetime.now(),
                channels=[AlertChannel.BROWSER, AlertChannel.EMAIL],
                metadata=data.get("metadata", {})
            )
            
            await self.alert_engine.send_notifications(alert)
            
        except Exception as e:
            logger.error(f"Failed to process external alert: {e}")

async def main():
    """Main function to run notification system"""
    monitor = LiveMetricsMonitor()
    
    try:
        # Start monitoring
        await monitor.start_monitoring()
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    finally:
        monitor.stop_monitoring()

if __name__ == "__main__":
    # Run monitoring system
    asyncio.run(main())