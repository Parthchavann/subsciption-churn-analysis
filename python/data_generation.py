"""
Synthetic Subscription Data Generator
Generates realistic subscription data for streaming service analysis
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from faker import Faker
import json

fake = Faker()
np.random.seed(42)
random.seed(42)

class SubscriptionDataGenerator:
    def __init__(self, n_users=100000, start_date='2023-01-01', n_months=24):
        self.n_users = n_users
        self.start_date = pd.to_datetime(start_date)
        self.n_months = n_months
        self.end_date = self.start_date + pd.DateOffset(months=n_months)
        
        # Subscription plans
        self.plans = {
            'Basic': {'price': 9.99, 'weight': 0.4},
            'Standard': {'price': 14.99, 'weight': 0.35},
            'Premium': {'price': 19.99, 'weight': 0.2},
            'Family': {'price': 24.99, 'weight': 0.05}
        }
        
        # Acquisition channels
        self.channels = {
            'Organic': 0.25,
            'Paid Search': 0.20,
            'Social Media': 0.15,
            'Email': 0.10,
            'Referral': 0.15,
            'Partnership': 0.10,
            'Direct': 0.05
        }
        
        # Device types
        self.devices = {
            'Mobile': 0.45,
            'Desktop': 0.25,
            'Smart TV': 0.20,
            'Tablet': 0.10
        }
        
        # Payment methods
        self.payment_methods = {
            'Credit Card': 0.60,
            'PayPal': 0.20,
            'Debit Card': 0.15,
            'Digital Wallet': 0.05
        }
        
    def generate_users(self):
        """Generate user demographics and account details"""
        users = []
        
        for user_id in range(1, self.n_users + 1):
            # Random signup date within the period
            days_in_period = (self.end_date - self.start_date).days
            signup_date = self.start_date + timedelta(days=random.randint(0, days_in_period - 30))
            
            # User demographics
            age = np.random.normal(35, 12)
            age = max(18, min(75, int(age)))
            
            user = {
                'user_id': user_id,
                'email': fake.email(),
                'age': age,
                'gender': np.random.choice(['M', 'F', 'Other'], p=[0.48, 0.48, 0.04]),
                'country': np.random.choice(['US', 'UK', 'CA', 'AU', 'DE', 'FR', 'JP', 'BR'], 
                                           p=[0.40, 0.15, 0.10, 0.08, 0.07, 0.07, 0.07, 0.06]),
                'city': fake.city(),
                'signup_date': signup_date,
                'acquisition_channel': np.random.choice(list(self.channels.keys()), 
                                                       p=list(self.channels.values())),
                'device_type': np.random.choice(list(self.devices.keys()), 
                                               p=list(self.devices.values())),
                'payment_method': np.random.choice(list(self.payment_methods.keys()), 
                                                  p=list(self.payment_methods.values()))
            }
            users.append(user)
        
        return pd.DataFrame(users)
    
    def generate_subscriptions(self, users_df):
        """Generate subscription history for each user"""
        subscriptions = []
        
        for _, user in users_df.iterrows():
            # Initial subscription
            plan_weights = list(self.plans[p]['weight'] for p in self.plans.keys())
            initial_plan = np.random.choice(list(self.plans.keys()), p=plan_weights)
            
            subscription_start = user['signup_date']
            
            # Determine if user churns and when
            base_churn_prob = 0.05  # 5% monthly churn rate
            
            # Factors affecting churn
            if user['age'] < 25:
                churn_modifier = 1.3
            elif user['age'] > 50:
                churn_modifier = 0.7
            else:
                churn_modifier = 1.0
            
            if initial_plan == 'Basic':
                churn_modifier *= 1.2
            elif initial_plan == 'Premium' or initial_plan == 'Family':
                churn_modifier *= 0.8
            
            if user['acquisition_channel'] == 'Referral':
                churn_modifier *= 0.6
            elif user['acquisition_channel'] == 'Paid Search':
                churn_modifier *= 1.1
            
            # Generate subscription periods
            current_date = subscription_start
            is_active = True
            plan = initial_plan
            
            while current_date < self.end_date and is_active:
                # Monthly subscription record
                sub_record = {
                    'subscription_id': f"{user['user_id']}_{current_date.strftime('%Y%m')}",
                    'user_id': user['user_id'],
                    'plan': plan,
                    'price': self.plans[plan]['price'],
                    'billing_date': current_date,
                    'status': 'active'
                }
                
                # Check for churn
                monthly_churn_prob = base_churn_prob * churn_modifier
                if random.random() < monthly_churn_prob:
                    is_active = False
                    sub_record['status'] = 'churned'
                    sub_record['churn_date'] = current_date + timedelta(days=random.randint(1, 28))
                    
                    # Churn reason
                    churn_reasons = {
                        'Price': 0.25,
                        'Content': 0.20,
                        'Competition': 0.15,
                        'Technical Issues': 0.10,
                        'No Longer Needed': 0.15,
                        'Customer Service': 0.10,
                        'Other': 0.05
                    }
                    sub_record['churn_reason'] = np.random.choice(list(churn_reasons.keys()),
                                                                 p=list(churn_reasons.values()))
                
                # Check for plan changes (10% chance per month)
                elif random.random() < 0.10:
                    new_plan = np.random.choice(list(self.plans.keys()), p=plan_weights)
                    if new_plan != plan:
                        sub_record['plan_change'] = True
                        sub_record['previous_plan'] = plan
                        plan = new_plan
                
                subscriptions.append(sub_record)
                current_date += pd.DateOffset(months=1)
        
        return pd.DataFrame(subscriptions)
    
    def generate_engagement(self, users_df, subscriptions_df):
        """Generate user engagement metrics"""
        engagement_data = []
        
        active_subs = subscriptions_df[subscriptions_df['status'] == 'active']
        
        for _, sub in active_subs.iterrows():
            user = users_df[users_df['user_id'] == sub['user_id']].iloc[0]
            
            # Base engagement based on plan
            if sub['plan'] == 'Premium' or sub['plan'] == 'Family':
                base_engagement = 0.8
            elif sub['plan'] == 'Standard':
                base_engagement = 0.6
            else:
                base_engagement = 0.4
            
            # Age factor
            if user['age'] < 30:
                age_factor = 1.2
            elif user['age'] > 50:
                age_factor = 0.8
            else:
                age_factor = 1.0
            
            engagement_score = min(1.0, base_engagement * age_factor * np.random.uniform(0.7, 1.3))
            
            engagement = {
                'user_id': sub['user_id'],
                'month': sub['billing_date'],
                'login_days': int(30 * engagement_score * np.random.uniform(0.5, 1.0)),
                'hours_watched': round(50 * engagement_score * np.random.uniform(0.3, 1.5), 1),
                'content_items': int(20 * engagement_score * np.random.uniform(0.4, 1.2)),
                'downloads': int(10 * engagement_score * np.random.uniform(0, 0.8)),
                'shares': int(5 * engagement_score * np.random.uniform(0, 0.5)),
                'ratings_given': int(10 * engagement_score * np.random.uniform(0, 0.7)),
                'engagement_score': round(engagement_score, 3)
            }
            
            engagement_data.append(engagement)
        
        return pd.DataFrame(engagement_data)
    
    def generate_support_tickets(self, users_df, subscriptions_df):
        """Generate customer support interactions"""
        tickets = []
        
        for _, user in users_df.iterrows():
            # Users who churned are more likely to have had support issues
            user_subs = subscriptions_df[subscriptions_df['user_id'] == user['user_id']]
            churned = any(user_subs['status'] == 'churned')
            
            if churned:
                ticket_prob = 0.4
            else:
                ticket_prob = 0.15
            
            if random.random() < ticket_prob:
                n_tickets = np.random.poisson(2)
                
                for _ in range(n_tickets):
                    ticket_date = user['signup_date'] + timedelta(days=random.randint(1, 180))
                    
                    ticket = {
                        'ticket_id': fake.uuid4(),
                        'user_id': user['user_id'],
                        'date': ticket_date,
                        'category': np.random.choice(['Billing', 'Technical', 'Content', 'Account', 'Other'],
                                                   p=[0.30, 0.25, 0.20, 0.15, 0.10]),
                        'priority': np.random.choice(['Low', 'Medium', 'High'], p=[0.50, 0.35, 0.15]),
                        'resolution_time_hours': np.random.exponential(24),
                        'satisfaction_score': np.random.choice([1, 2, 3, 4, 5], 
                                                              p=[0.05, 0.10, 0.20, 0.35, 0.30])
                    }
                    tickets.append(ticket)
        
        return pd.DataFrame(tickets)
    
    def generate_all_data(self):
        """Generate complete dataset"""
        print("Generating users...")
        users_df = self.generate_users()
        
        print("Generating subscriptions...")
        subscriptions_df = self.generate_subscriptions(users_df)
        
        print("Generating engagement data...")
        engagement_df = self.generate_engagement(users_df, subscriptions_df)
        
        print("Generating support tickets...")
        tickets_df = self.generate_support_tickets(users_df, subscriptions_df)
        
        # Save to CSV
        users_df.to_csv('../data/raw/users.csv', index=False)
        subscriptions_df.to_csv('../data/raw/subscriptions.csv', index=False)
        engagement_df.to_csv('../data/raw/engagement.csv', index=False)
        tickets_df.to_csv('../data/raw/support_tickets.csv', index=False)
        
        # Generate summary statistics
        stats = {
            'total_users': len(users_df),
            'active_users': len(subscriptions_df[subscriptions_df['status'] == 'active']['user_id'].unique()),
            'churned_users': len(subscriptions_df[subscriptions_df['status'] == 'churned']['user_id'].unique()),
            'total_revenue': subscriptions_df['price'].sum(),
            'avg_engagement_score': engagement_df['engagement_score'].mean(),
            'total_support_tickets': len(tickets_df)
        }
        
        with open('../data/raw/data_summary.json', 'w') as f:
            json.dump(stats, f, indent=4, default=str)
        
        print("\nData Generation Complete!")
        print(f"Total Users: {stats['total_users']:,}")
        print(f"Active Users: {stats['active_users']:,}")
        print(f"Churned Users: {stats['churned_users']:,}")
        print(f"Total Revenue: ${stats['total_revenue']:,.2f}")
        print(f"Files saved to ../data/raw/")
        
        return users_df, subscriptions_df, engagement_df, tickets_df

if __name__ == "__main__":
    generator = SubscriptionDataGenerator(n_users=100000, n_months=24)
    generator.generate_all_data()