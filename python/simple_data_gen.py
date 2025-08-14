#!/usr/bin/env python3
"""
Simple Data Generator (No External Dependencies)
Creates sample CSV files for demonstration
"""

import csv
import random
import json
from datetime import datetime, timedelta

# Set random seed for reproducibility
random.seed(42)

def generate_sample_data():
    """Generate sample datasets without external libraries"""
    
    print("Generating sample subscription data...")
    
    # Sample data parameters
    n_users = 1000  # Smaller dataset for demo
    start_date = datetime(2023, 1, 1)
    
    # Create data directory
    import os
    os.makedirs('../data/raw', exist_ok=True)
    
    # Generate users
    users_data = []
    for user_id in range(1, n_users + 1):
        signup_days = random.randint(0, 365)
        signup_date = start_date + timedelta(days=signup_days)
        
        user = {
            'user_id': user_id,
            'email': f'user{user_id}@email.com',
            'age': random.randint(18, 65),
            'gender': random.choice(['M', 'F', 'Other']),
            'country': random.choice(['US', 'UK', 'CA', 'AU', 'DE']),
            'city': f'City_{random.randint(1, 100)}',
            'signup_date': signup_date.strftime('%Y-%m-%d'),
            'acquisition_channel': random.choice(['Organic', 'Paid Search', 'Social Media', 'Referral']),
            'device_type': random.choice(['Mobile', 'Desktop', 'Smart TV', 'Tablet']),
            'payment_method': random.choice(['Credit Card', 'PayPal', 'Debit Card'])
        }
        users_data.append(user)
    
    # Write users CSV
    with open('../data/raw/users.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=users_data[0].keys())
        writer.writeheader()
        writer.writerows(users_data)
    
    # Generate subscriptions
    subscriptions_data = []
    plans = ['Basic', 'Standard', 'Premium', 'Family']
    prices = {'Basic': 9.99, 'Standard': 14.99, 'Premium': 19.99, 'Family': 24.99}
    
    for user in users_data:
        user_id = user['user_id']
        signup_date = datetime.strptime(user['signup_date'], '%Y-%m-%d')
        plan = random.choice(plans)
        
        # Generate 3-12 months of subscription history
        months_active = random.randint(3, 12)
        is_churned = random.random() < 0.15  # 15% churn rate
        
        for month in range(months_active):
            billing_date = signup_date + timedelta(days=30 * month)
            status = 'churned' if (is_churned and month == months_active - 1) else 'active'
            
            subscription = {
                'subscription_id': f"{user_id}_{billing_date.strftime('%Y%m')}",
                'user_id': user_id,
                'plan': plan,
                'price': prices[plan],
                'billing_date': billing_date.strftime('%Y-%m-%d'),
                'status': status
            }
            
            if status == 'churned':
                subscription['churn_date'] = (billing_date + timedelta(days=random.randint(1, 28))).strftime('%Y-%m-%d')
                subscription['churn_reason'] = random.choice(['Price', 'Content', 'Competition', 'Technical Issues'])
            
            subscriptions_data.append(subscription)
    
    # Write subscriptions CSV
    with open('../data/raw/subscriptions.csv', 'w', newline='') as f:
        fieldnames = ['subscription_id', 'user_id', 'plan', 'price', 'billing_date', 'status', 'churn_date', 'churn_reason']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(subscriptions_data)
    
    # Generate engagement data
    engagement_data = []
    for subscription in subscriptions_data:
        if subscription['status'] == 'active':
            engagement = {
                'user_id': subscription['user_id'],
                'month': subscription['billing_date'],
                'login_days': random.randint(5, 30),
                'hours_watched': round(random.uniform(10, 100), 1),
                'content_items': random.randint(5, 50),
                'downloads': random.randint(0, 20),
                'shares': random.randint(0, 10),
                'ratings_given': random.randint(0, 15),
                'engagement_score': round(random.uniform(0.2, 1.0), 3)
            }
            engagement_data.append(engagement)
    
    # Write engagement CSV
    with open('../data/raw/engagement.csv', 'w', newline='') as f:
        if engagement_data:
            writer = csv.DictWriter(f, fieldnames=engagement_data[0].keys())
            writer.writeheader()
            writer.writerows(engagement_data)
    
    # Generate support tickets
    support_data = []
    for user in random.sample(users_data, min(300, len(users_data))):  # 30% have tickets
        num_tickets = random.randint(1, 3)
        for _ in range(num_tickets):
            signup_date = datetime.strptime(user['signup_date'], '%Y-%m-%d')
            ticket_date = signup_date + timedelta(days=random.randint(1, 200))
            
            ticket = {
                'ticket_id': f"TKT_{random.randint(10000, 99999)}",
                'user_id': user['user_id'],
                'date': ticket_date.strftime('%Y-%m-%d'),
                'category': random.choice(['Billing', 'Technical', 'Content', 'Account']),
                'priority': random.choice(['Low', 'Medium', 'High']),
                'resolution_time_hours': round(random.uniform(1, 72), 2),
                'satisfaction_score': random.randint(1, 5)
            }
            support_data.append(ticket)
    
    # Write support tickets CSV
    with open('../data/raw/support_tickets.csv', 'w', newline='') as f:
        if support_data:
            writer = csv.DictWriter(f, fieldnames=support_data[0].keys())
            writer.writeheader()
            writer.writerows(support_data)
    
    # Generate summary statistics
    active_users = len([s for s in subscriptions_data if s['status'] == 'active'])
    churned_users = len(set(s['user_id'] for s in subscriptions_data if s['status'] == 'churned'))
    total_revenue = sum(float(s['price']) for s in subscriptions_data)
    
    stats = {
        'total_users': n_users,
        'active_subscriptions': active_users,
        'churned_users': churned_users,
        'total_revenue': round(total_revenue, 2),
        'avg_engagement_score': round(sum(e['engagement_score'] for e in engagement_data) / len(engagement_data), 3) if engagement_data else 0,
        'total_support_tickets': len(support_data),
        'churn_rate': round((churned_users / n_users) * 100, 2)
    }
    
    with open('../data/raw/data_summary.json', 'w') as f:
        json.dump(stats, f, indent=4)
    
    print("\nSample Data Generated Successfully!")
    print(f"Total Users: {stats['total_users']:,}")
    print(f"Active Subscriptions: {stats['active_subscriptions']:,}")
    print(f"Churned Users: {stats['churned_users']:,}")
    print(f"Churn Rate: {stats['churn_rate']}%")
    print(f"Total Revenue: ${stats['total_revenue']:,.2f}")
    print(f"Support Tickets: {stats['total_support_tickets']:,}")
    print(f"Files saved to ../data/raw/")
    
    return stats

if __name__ == "__main__":
    generate_sample_data()