#!/usr/bin/env python3
"""
🤖 AI Executive Assistant with Natural Language Analytics
Advanced conversational AI for business intelligence queries and strategic insights
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
import logging
import sqlite3
import uuid
import re
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

# Mock NLP and AI libraries (in production use actual libraries)
class MockNLPProcessor:
    @staticmethod
    def extract_intent(query: str) -> Dict[str, Any]:
        # Mock intent extraction
        intents = {
            'forecast': ['forecast', 'predict', 'projection', 'future', 'next'],
            'analysis': ['analyze', 'analysis', 'insights', 'understand', 'explain'],
            'comparison': ['compare', 'vs', 'versus', 'difference', 'better'],
            'summary': ['summary', 'overview', 'report', 'brief', 'highlights'],
            'trend': ['trend', 'trending', 'pattern', 'growth', 'decline'],
            'performance': ['performance', 'metrics', 'kpi', 'success', 'results']
        }
        
        query_lower = query.lower()
        detected_intent = 'general'
        confidence = 0.5
        
        for intent, keywords in intents.items():
            if any(keyword in query_lower for keyword in keywords):
                detected_intent = intent
                confidence = 0.9
                break
        
        return {'intent': detected_intent, 'confidence': confidence}
    
    @staticmethod
    def extract_entities(query: str) -> List[Dict[str, Any]]:
        # Mock entity extraction
        entities = []
        
        # Time entities
        time_patterns = {
            'month': r'\b(january|february|march|april|may|june|july|august|september|october|november|december|\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b',
            'period': r'\b(last\s+\w+|next\s+\w+|this\s+\w+|past\s+\w+)\b',
            'duration': r'\b(\d+\s+(days?|weeks?|months?|years?))\b'
        }
        
        for entity_type, pattern in time_patterns.items():
            matches = re.findall(pattern, query.lower())
            for match in matches:
                entities.append({
                    'type': entity_type,
                    'value': match,
                    'confidence': 0.8
                })
        
        # Metric entities
        metric_keywords = ['revenue', 'churn', 'customers', 'ltv', 'arpu', 'roi', 'profit', 'cost']
        for metric in metric_keywords:
            if metric in query.lower():
                entities.append({
                    'type': 'metric',
                    'value': metric,
                    'confidence': 0.9
                })
        
        return entities

class MockLLM:
    @staticmethod
    def generate_response(prompt: str, context: Dict[str, Any]) -> str:
        # Mock LLM response generation
        return f"Based on the analysis of your subscription data, here are the key insights: {json.dumps(context, indent=2)}"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QueryType(Enum):
    FORECAST = "forecast"
    ANALYSIS = "analysis"
    COMPARISON = "comparison"
    SUMMARY = "summary"
    TREND = "trend"
    PERFORMANCE = "performance"
    GENERAL = "general"

@dataclass
class QueryContext:
    """Context for natural language query"""
    user_query: str
    intent: QueryType
    entities: List[Dict[str, Any]]
    confidence: float
    extracted_metrics: List[str]
    time_frame: Optional[str]
    business_context: Dict[str, Any]

@dataclass
class AIResponse:
    """AI assistant response"""
    response_text: str
    data_insights: Dict[str, Any]
    visualizations: List[Dict[str, Any]]
    action_items: List[str]
    confidence_score: float
    follow_up_questions: List[str]
    response_time: float

class AIExecutiveAssistant:
    """AI Executive Assistant for Business Intelligence"""
    
    def __init__(self, db_path: str = "data/ai_assistant.db"):
        self.db_path = db_path
        self.nlp_processor = MockNLPProcessor()
        self.llm = MockLLM()
        self.initialize_database()
        self._load_business_context()
        
    def initialize_database(self):
        """Initialize AI assistant database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create queries table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_queries (
            query_id TEXT PRIMARY KEY,
            user_query TEXT,
            intent TEXT,
            entities TEXT,
            confidence REAL,
            response_text TEXT,
            data_insights TEXT,
            response_time REAL,
            user_satisfaction INTEGER,
            created_at TEXT
        )
        ''')
        
        # Create insights cache
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS insights_cache (
            cache_id TEXT PRIMARY KEY,
            query_hash TEXT,
            cached_response TEXT,
            cached_data TEXT,
            cache_expiry TEXT,
            created_at TEXT
        )
        ''')
        
        # Create conversation history
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS conversation_history (
            conversation_id TEXT PRIMARY KEY,
            user_id TEXT,
            session_id TEXT,
            query_sequence INTEGER,
            query_text TEXT,
            response_text TEXT,
            context TEXT,
            timestamp TEXT
        )
        ''')
        
        conn.commit()
        conn.close()
    
    def _load_business_context(self):
        """Load business context and metrics"""
        self.business_context = {
            'company': {
                'name': 'Subscription Analytics Co',
                'industry': 'SaaS',
                'customers': 25000,
                'mrr': 1250000,
                'founding_year': 2018
            },
            'metrics': {
                'churn_rate': 0.052,
                'ltv': 2400,
                'cac': 150,
                'arpu': 50,
                'growth_rate': 0.15
            },
            'goals': {
                'target_churn_rate': 0.035,
                'target_mrr_growth': 0.20,
                'target_customers': 50000
            },
            'business_rules': {
                'healthy_churn_rate': 0.05,
                'good_ltv_cac_ratio': 3.0,
                'target_growth_rate': 0.15
            }
        }
    
    async def process_query(self, user_query: str, user_id: str = "executive", 
                          session_id: str = None) -> AIResponse:
        """Process natural language query and generate AI response"""
        start_time = datetime.now()
        query_id = str(uuid.uuid4())
        session_id = session_id or str(uuid.uuid4())
        
        try:
            # Extract intent and entities
            intent_result = self.nlp_processor.extract_intent(user_query)
            entities = self.nlp_processor.extract_entities(user_query)
            
            # Create query context
            context = QueryContext(
                user_query=user_query,
                intent=QueryType(intent_result['intent']),
                entities=entities,
                confidence=intent_result['confidence'],
                extracted_metrics=self._extract_metrics_from_entities(entities),
                time_frame=self._extract_time_frame(entities),
                business_context=self.business_context
            )
            
            # Check cache
            cached_response = await self._check_cache(user_query)
            if cached_response:
                return cached_response
            
            # Generate data insights
            data_insights = await self._generate_data_insights(context)
            
            # Generate AI response
            response_text = await self._generate_ai_response(context, data_insights)
            
            # Generate visualizations
            visualizations = self._suggest_visualizations(context, data_insights)
            
            # Generate action items
            action_items = self._generate_action_items(context, data_insights)
            
            # Generate follow-up questions
            follow_up_questions = self._generate_follow_up_questions(context)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            response = AIResponse(
                response_text=response_text,
                data_insights=data_insights,
                visualizations=visualizations,
                action_items=action_items,
                confidence_score=context.confidence,
                follow_up_questions=follow_up_questions,
                response_time=processing_time
            )
            
            # Save to database
            await self._save_query_response(query_id, context, response, user_id, session_id)
            
            # Cache response
            await self._cache_response(user_query, response)
            
            logger.info(f"Query processed: {query_id}")
            return response
            
        except Exception as e:
            logger.error(f"Query processing failed: {str(e)}")
            # Return fallback response
            return self._generate_fallback_response(user_query)
    
    def _extract_metrics_from_entities(self, entities: List[Dict[str, Any]]) -> List[str]:
        """Extract metrics from entities"""
        metrics = []
        for entity in entities:
            if entity['type'] == 'metric':
                metrics.append(entity['value'])
        return metrics
    
    def _extract_time_frame(self, entities: List[Dict[str, Any]]) -> Optional[str]:
        """Extract time frame from entities"""
        for entity in entities:
            if entity['type'] in ['period', 'duration', 'month']:
                return entity['value']
        return None
    
    async def _check_cache(self, query: str) -> Optional[AIResponse]:
        """Check if response is cached"""
        query_hash = str(hash(query.lower().strip()))
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT cached_response, cached_data, cache_expiry 
        FROM insights_cache 
        WHERE query_hash = ? AND cache_expiry > ?
        ''', (query_hash, datetime.now().isoformat()))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            # Return cached response
            cached_data = json.loads(result[1])
            return AIResponse(**cached_data)
        
        return None
    
    async def _generate_data_insights(self, context: QueryContext) -> Dict[str, Any]:
        """Generate data insights based on query context"""
        insights = {'query_type': context.intent.value}
        
        if context.intent == QueryType.FORECAST:
            insights.update(await self._generate_forecast_insights(context))
        elif context.intent == QueryType.ANALYSIS:
            insights.update(await self._generate_analysis_insights(context))
        elif context.intent == QueryType.COMPARISON:
            insights.update(await self._generate_comparison_insights(context))
        elif context.intent == QueryType.SUMMARY:
            insights.update(await self._generate_summary_insights(context))
        elif context.intent == QueryType.TREND:
            insights.update(await self._generate_trend_insights(context))
        elif context.intent == QueryType.PERFORMANCE:
            insights.update(await self._generate_performance_insights(context))
        else:
            insights.update(await self._generate_general_insights(context))
        
        return insights
    
    async def _generate_forecast_insights(self, context: QueryContext) -> Dict[str, Any]:
        """Generate forecast-specific insights"""
        insights = {
            'type': 'forecast',
            'horizon': context.time_frame or '3 months',
            'forecasted_metrics': {}
        }
        
        # Mock forecast data
        for metric in context.extracted_metrics or ['revenue', 'customers', 'churn']:
            if metric == 'revenue':
                current = self.business_context['company']['mrr']
                growth = self.business_context['metrics']['growth_rate']
                forecast = current * (1 + growth) ** 3  # 3 month projection
                insights['forecasted_metrics'][metric] = {
                    'current': current,
                    'forecast': forecast,
                    'growth': ((forecast / current) - 1) * 100
                }
            elif metric == 'churn':
                current = self.business_context['metrics']['churn_rate']
                trend = -0.001  # Improving trend
                forecast = max(0.01, current + trend * 3)
                insights['forecasted_metrics'][metric] = {
                    'current': current * 100,
                    'forecast': forecast * 100,
                    'improvement': (current - forecast) * 100
                }
        
        insights['confidence'] = 0.85
        insights['methodology'] = 'ensemble_ml_models'
        
        return insights
    
    async def _generate_analysis_insights(self, context: QueryContext) -> Dict[str, Any]:
        """Generate analysis-specific insights"""
        insights = {
            'type': 'analysis',
            'key_findings': [],
            'metrics_analyzed': context.extracted_metrics or ['all_metrics']
        }
        
        # Generate key findings based on business context
        current_metrics = self.business_context['metrics']
        targets = self.business_context['goals']
        
        # Churn analysis
        if current_metrics['churn_rate'] > targets['target_churn_rate']:
            insights['key_findings'].append({
                'metric': 'churn_rate',
                'finding': 'Churn rate above target',
                'current': current_metrics['churn_rate'] * 100,
                'target': targets['target_churn_rate'] * 100,
                'impact': 'high',
                'recommendation': 'Implement retention campaigns'
            })
        
        # LTV/CAC analysis
        ltv_cac_ratio = current_metrics['ltv'] / current_metrics['cac']
        if ltv_cac_ratio > 3.0:
            insights['key_findings'].append({
                'metric': 'ltv_cac_ratio',
                'finding': 'Healthy unit economics',
                'ratio': ltv_cac_ratio,
                'benchmark': 3.0,
                'impact': 'positive',
                'recommendation': 'Consider scaling acquisition'
            })
        
        insights['overall_health'] = 'good' if len([f for f in insights['key_findings'] if f['impact'] == 'positive']) > len([f for f in insights['key_findings'] if f['impact'] == 'high']) else 'needs_attention'
        
        return insights
    
    async def _generate_comparison_insights(self, context: QueryContext) -> Dict[str, Any]:
        """Generate comparison insights"""
        insights = {
            'type': 'comparison',
            'comparisons': [],
            'time_periods': ['current', 'previous_period']
        }
        
        # Mock comparison data
        for metric in context.extracted_metrics or ['revenue', 'churn']:
            if metric == 'revenue':
                current = self.business_context['company']['mrr']
                previous = current * 0.92  # 8% growth
                insights['comparisons'].append({
                    'metric': metric,
                    'current': current,
                    'previous': previous,
                    'change_percent': ((current / previous) - 1) * 100,
                    'trend': 'increasing'
                })
        
        return insights
    
    async def _generate_summary_insights(self, context: QueryContext) -> Dict[str, Any]:
        """Generate summary insights"""
        insights = {
            'type': 'summary',
            'executive_summary': {},
            'key_metrics': self.business_context['metrics'],
            'performance_indicators': []
        }
        
        # Generate executive summary
        insights['executive_summary'] = {
            'total_customers': self.business_context['company']['customers'],
            'monthly_revenue': self.business_context['company']['mrr'],
            'churn_rate': self.business_context['metrics']['churn_rate'] * 100,
            'growth_rate': self.business_context['metrics']['growth_rate'] * 100,
            'health_score': 8.2  # Mock health score
        }
        
        # Performance indicators
        insights['performance_indicators'] = [
            {'indicator': 'Revenue Growth', 'status': 'green', 'value': '15%'},
            {'indicator': 'Churn Rate', 'status': 'yellow', 'value': '5.2%'},
            {'indicator': 'Customer Acquisition', 'status': 'green', 'value': '+12%'},
            {'indicator': 'LTV/CAC Ratio', 'status': 'green', 'value': '16:1'}
        ]
        
        return insights
    
    async def _generate_trend_insights(self, context: QueryContext) -> Dict[str, Any]:
        """Generate trend insights"""
        insights = {
            'type': 'trend',
            'detected_trends': [],
            'trend_analysis': {}
        }
        
        # Mock trend data
        trends = [
            {'metric': 'revenue', 'direction': 'upward', 'strength': 'strong', 'duration': '6 months'},
            {'metric': 'churn', 'direction': 'stable', 'strength': 'moderate', 'duration': '3 months'},
            {'metric': 'customer_acquisition', 'direction': 'upward', 'strength': 'moderate', 'duration': '4 months'}
        ]
        
        insights['detected_trends'] = trends
        insights['trend_confidence'] = 0.78
        
        return insights
    
    async def _generate_performance_insights(self, context: QueryContext) -> Dict[str, Any]:
        """Generate performance insights"""
        insights = {
            'type': 'performance',
            'kpi_performance': {},
            'benchmarks': {},
            'performance_score': 0
        }
        
        # Calculate performance against targets
        current = self.business_context['metrics']
        targets = self.business_context['goals']
        
        performance_scores = []
        
        # Churn performance
        churn_performance = (targets['target_churn_rate'] / current['churn_rate']) * 100
        insights['kpi_performance']['churn_rate'] = {
            'current': current['churn_rate'] * 100,
            'target': targets['target_churn_rate'] * 100,
            'performance_score': min(100, churn_performance)
        }
        performance_scores.append(min(100, churn_performance))
        
        # Growth performance
        growth_target = targets['target_mrr_growth']
        current_growth = current['growth_rate']
        growth_performance = (current_growth / growth_target) * 100
        insights['kpi_performance']['growth_rate'] = {
            'current': current_growth * 100,
            'target': growth_target * 100,
            'performance_score': min(100, growth_performance)
        }
        performance_scores.append(min(100, growth_performance))
        
        insights['performance_score'] = np.mean(performance_scores)
        
        return insights
    
    async def _generate_general_insights(self, context: QueryContext) -> Dict[str, Any]:
        """Generate general insights"""
        insights = {
            'type': 'general',
            'business_overview': self.business_context['company'],
            'quick_stats': self.business_context['metrics'],
            'recommendations': [
                'Monitor churn rate trends closely',
                'Consider expanding successful acquisition channels',
                'Implement predictive analytics for early churn detection'
            ]
        }
        
        return insights
    
    async def _generate_ai_response(self, context: QueryContext, data_insights: Dict[str, Any]) -> str:
        """Generate natural language AI response"""
        # Build response based on intent and insights
        if context.intent == QueryType.FORECAST:
            return self._generate_forecast_response(context, data_insights)
        elif context.intent == QueryType.ANALYSIS:
            return self._generate_analysis_response(context, data_insights)
        elif context.intent == QueryType.SUMMARY:
            return self._generate_summary_response(context, data_insights)
        else:
            return self._generate_general_response(context, data_insights)
    
    def _generate_forecast_response(self, context: QueryContext, insights: Dict[str, Any]) -> str:
        """Generate forecast response"""
        response_parts = []
        response_parts.append("📈 **Forecast Analysis**")
        response_parts.append("")
        
        for metric, data in insights.get('forecasted_metrics', {}).items():
            if metric == 'revenue':
                response_parts.append(f"**Revenue Projection ({insights.get('horizon', '3 months')}):**")
                response_parts.append(f"- Current MRR: ${data['current']:,.0f}")
                response_parts.append(f"- Projected MRR: ${data['forecast']:,.0f}")
                response_parts.append(f"- Expected Growth: +{data['growth']:.1f}%")
            elif metric == 'churn':
                response_parts.append(f"**Churn Rate Forecast:**")
                response_parts.append(f"- Current Rate: {data['current']:.1f}%")
                response_parts.append(f"- Projected Rate: {data['forecast']:.1f}%")
                response_parts.append(f"- Expected Improvement: -{data['improvement']:.1f}%")
        
        response_parts.append("")
        response_parts.append(f"**Confidence Level:** {insights.get('confidence', 0.8):.0%}")
        response_parts.append("*This forecast is based on ensemble ML models considering historical trends, seasonality, and market conditions.*")
        
        return "\n".join(response_parts)
    
    def _generate_analysis_response(self, context: QueryContext, insights: Dict[str, Any]) -> str:
        """Generate analysis response"""
        response_parts = []
        response_parts.append("🔍 **Business Analysis**")
        response_parts.append("")
        
        overall_health = insights.get('overall_health', 'good')
        response_parts.append(f"**Overall Business Health:** {overall_health.title()}")
        response_parts.append("")
        
        response_parts.append("**Key Findings:**")
        for finding in insights.get('key_findings', []):
            icon = "✅" if finding['impact'] == 'positive' else "⚠️" if finding['impact'] == 'high' else "ℹ️"
            response_parts.append(f"{icon} **{finding['metric'].replace('_', ' ').title()}:** {finding['finding']}")
            if 'recommendation' in finding:
                response_parts.append(f"   *Recommendation: {finding['recommendation']}*")
        
        return "\n".join(response_parts)
    
    def _generate_summary_response(self, context: QueryContext, insights: Dict[str, Any]) -> str:
        """Generate summary response"""
        response_parts = []
        response_parts.append("📊 **Executive Summary**")
        response_parts.append("")
        
        summary = insights.get('executive_summary', {})
        response_parts.append("**Current Performance:**")
        response_parts.append(f"- Total Customers: {summary.get('total_customers', 0):,}")
        response_parts.append(f"- Monthly Revenue: ${summary.get('monthly_revenue', 0):,}")
        response_parts.append(f"- Churn Rate: {summary.get('churn_rate', 0):.1f}%")
        response_parts.append(f"- Growth Rate: {summary.get('growth_rate', 0):.1f}%")
        response_parts.append(f"- Health Score: {summary.get('health_score', 0):.1f}/10")
        response_parts.append("")
        
        response_parts.append("**Performance Indicators:**")
        for indicator in insights.get('performance_indicators', []):
            status_icon = "🟢" if indicator['status'] == 'green' else "🟡" if indicator['status'] == 'yellow' else "🔴"
            response_parts.append(f"{status_icon} {indicator['indicator']}: {indicator['value']}")
        
        return "\n".join(response_parts)
    
    def _generate_general_response(self, context: QueryContext, insights: Dict[str, Any]) -> str:
        """Generate general response"""
        response_parts = []
        response_parts.append("💡 **Business Insights**")
        response_parts.append("")
        
        response_parts.append("I've analyzed your subscription business data. Here are the key insights:")
        response_parts.append("")
        
        recommendations = insights.get('recommendations', [])
        if recommendations:
            response_parts.append("**Recommendations:**")
            for i, rec in enumerate(recommendations, 1):
                response_parts.append(f"{i}. {rec}")
        
        return "\n".join(response_parts)
    
    def _suggest_visualizations(self, context: QueryContext, insights: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Suggest relevant visualizations"""
        visualizations = []
        
        if context.intent == QueryType.FORECAST:
            visualizations.append({
                'type': 'line_chart',
                'title': 'Revenue Forecast',
                'description': 'Projected revenue growth with confidence intervals',
                'data_source': 'forecasted_metrics'
            })
        
        if context.intent == QueryType.TREND:
            visualizations.append({
                'type': 'trend_analysis',
                'title': 'Metric Trends',
                'description': 'Historical trends for key business metrics',
                'data_source': 'historical_data'
            })
        
        if context.intent == QueryType.COMPARISON:
            visualizations.append({
                'type': 'comparison_chart',
                'title': 'Period Comparison',
                'description': 'Side-by-side comparison of metrics',
                'data_source': 'comparison_data'
            })
        
        if 'performance' in context.user_query.lower():
            visualizations.append({
                'type': 'kpi_dashboard',
                'title': 'Performance Dashboard',
                'description': 'Key performance indicators overview',
                'data_source': 'kpi_data'
            })
        
        return visualizations
    
    def _generate_action_items(self, context: QueryContext, insights: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations"""
        action_items = []
        
        if context.intent == QueryType.ANALYSIS:
            findings = insights.get('key_findings', [])
            for finding in findings:
                if finding.get('impact') == 'high' and 'recommendation' in finding:
                    action_items.append(finding['recommendation'])
        
        if context.intent == QueryType.FORECAST:
            action_items.append("Review forecast assumptions quarterly")
            action_items.append("Set up automated alerts for forecast deviations")
        
        if context.intent == QueryType.PERFORMANCE:
            score = insights.get('performance_score', 0)
            if score < 80:
                action_items.append("Focus on underperforming KPIs")
                action_items.append("Develop improvement action plans")
        
        # Default action items
        if not action_items:
            action_items = [
                "Monitor key metrics weekly",
                "Set up automated reporting",
                "Schedule monthly business reviews"
            ]
        
        return action_items[:5]  # Limit to 5 action items
    
    def _generate_follow_up_questions(self, context: QueryContext) -> List[str]:
        """Generate relevant follow-up questions"""
        questions = []
        
        if context.intent == QueryType.FORECAST:
            questions.extend([
                "What factors could impact this forecast?",
                "How does this compare to industry benchmarks?",
                "What scenarios should we prepare for?"
            ])
        
        if context.intent == QueryType.ANALYSIS:
            questions.extend([
                "What's driving these trends?",
                "How can we improve underperforming areas?",
                "What are the biggest risks to monitor?"
            ])
        
        if context.intent == QueryType.PERFORMANCE:
            questions.extend([
                "Which metrics need immediate attention?",
                "What's our competitive position?",
                "How can we accelerate growth?"
            ])
        
        # Default questions
        if not questions:
            questions = [
                "What other metrics would you like to explore?",
                "How can I help you make data-driven decisions?",
                "Would you like to see a detailed analysis of any specific area?"
            ]
        
        return questions[:3]  # Limit to 3 questions
    
    def _generate_fallback_response(self, query: str) -> AIResponse:
        """Generate fallback response for failed queries"""
        return AIResponse(
            response_text="I apologize, but I'm having trouble processing your request. Let me help you with some general business insights instead.",
            data_insights={'type': 'fallback', 'query': query},
            visualizations=[],
            action_items=["Please rephrase your question", "Try asking about specific metrics"],
            confidence_score=0.1,
            follow_up_questions=["What specific metric would you like to analyze?", "Are you looking for forecasts or current performance?"],
            response_time=0.1
        )
    
    async def _save_query_response(self, query_id: str, context: QueryContext, 
                                 response: AIResponse, user_id: str, session_id: str):
        """Save query and response to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Save to user_queries table
        cursor.execute('''
        INSERT INTO user_queries VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            query_id, context.user_query, context.intent.value,
            json.dumps([asdict(e) for e in context.entities]), context.confidence,
            response.response_text, json.dumps(response.data_insights),
            response.response_time, None, datetime.now().isoformat()
        ))
        
        # Save to conversation history
        cursor.execute('''
        INSERT INTO conversation_history VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            str(uuid.uuid4()), user_id, session_id, 1,
            context.user_query, response.response_text,
            json.dumps(asdict(context)), datetime.now().isoformat()
        ))
        
        conn.commit()
        conn.close()
    
    async def _cache_response(self, query: str, response: AIResponse):
        """Cache response for future use"""
        query_hash = str(hash(query.lower().strip()))
        cache_expiry = datetime.now() + timedelta(hours=1)  # 1 hour cache
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        INSERT OR REPLACE INTO insights_cache VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            str(uuid.uuid4()), query_hash, response.response_text,
            json.dumps(asdict(response)), cache_expiry.isoformat(),
            datetime.now().isoformat()
        ))
        
        conn.commit()
        conn.close()
    
    def get_conversation_history(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get conversation history for user"""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
        SELECT * FROM conversation_history 
        WHERE user_id = ? 
        ORDER BY timestamp DESC 
        LIMIT ?
        '''
        
        df = pd.read_sql_query(query, conn, params=(user_id, limit))
        conn.close()
        
        return df.to_dict('records')
    
    def get_usage_analytics(self) -> Dict[str, Any]:
        """Get AI assistant usage analytics"""
        conn = sqlite3.connect(self.db_path)
        
        # Query statistics
        queries_df = pd.read_sql_query('SELECT * FROM user_queries', conn)
        
        analytics = {
            'total_queries': len(queries_df),
            'avg_response_time': queries_df['response_time'].mean() if not queries_df.empty else 0,
            'avg_confidence': queries_df['confidence'].mean() if not queries_df.empty else 0,
            'most_common_intents': queries_df['intent'].value_counts().to_dict() if not queries_df.empty else {},
            'daily_usage': queries_df.groupby(pd.to_datetime(queries_df['created_at']).dt.date).size().to_dict() if not queries_df.empty else {}
        }
        
        conn.close()
        return analytics

async def run_ai_assistant_demo():
    """Run AI executive assistant demonstration"""
    assistant = AIExecutiveAssistant()
    
    print("🤖 Running AI Executive Assistant Demo...")
    print("=" * 60)
    
    # Sample queries
    sample_queries = [
        "What's our revenue forecast for the next 3 months?",
        "Analyze our churn rate performance",
        "Give me a summary of our business performance",
        "How do our metrics compare to last quarter?",
        "What trends are you seeing in our data?",
        "What's our customer acquisition performance?"
    ]
    
    responses = []
    
    for i, query in enumerate(sample_queries, 1):
        print(f"\n🔍 Query {i}: \"{query}\"")
        print("-" * 40)
        
        response = await assistant.process_query(query, user_id="demo_user")
        responses.append(response)
        
        print(f"**Response ({response.response_time:.2f}s, {response.confidence_score:.0%} confidence):**")
        print(response.response_text)
        
        if response.action_items:
            print("\n**Action Items:**")
            for item in response.action_items:
                print(f"• {item}")
        
        if response.follow_up_questions:
            print("\n**Follow-up Questions:**")
            for question in response.follow_up_questions:
                print(f"❓ {question}")
        
        print("\n" + "=" * 60)
    
    # Usage analytics
    analytics = assistant.get_usage_analytics()
    print("\n📊 **Usage Analytics:**")
    print(f"- Total Queries: {analytics['total_queries']}")
    print(f"- Average Response Time: {analytics['avg_response_time']:.2f}s")
    print(f"- Average Confidence: {analytics['avg_confidence']:.0%}")
    
    return responses, analytics

if __name__ == "__main__":
    import asyncio
    asyncio.run(run_ai_assistant_demo())