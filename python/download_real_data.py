#!/usr/bin/env python3
"""
Download Real Datasets for Advanced Subscription Analysis
Downloads and prepares real-world datasets without external dependencies
"""
import urllib.request
import json
import csv
import os
from pathlib import Path

def download_file(url, filename):
    """Download file from URL"""
    try:
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, filename)
        print(f"✅ Downloaded {filename}")
        return True
    except Exception as e:
        print(f"❌ Failed to download {filename}: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    directories = [
        "data/real",
        "data/processed", 
        "data/external"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"📁 Created directory: {directory}")

def download_real_datasets():
    """Download real-world datasets"""
    datasets = [
        {
            "name": "Telecom Customer Churn",
            "url": "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv",
            "filename": "data/real/telco_churn.csv",
            "description": "IBM Telecom customer churn dataset with 7043 customers"
        }
    ]
    
    downloaded_datasets = []
    
    for dataset in datasets:
        if download_file(dataset["url"], dataset["filename"]):
            downloaded_datasets.append(dataset)
    
    return downloaded_datasets

def process_telco_dataset():
    """Process the Telecom dataset for subscription analysis"""
    try:
        # Read CSV manually without pandas
        customers = []
        with open('data/real/telco_churn.csv', 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                customers.append(row)
        
        print(f"📊 Loaded {len(customers)} customers from Telecom dataset")
        
        # Process and create subscription-style data
        processed_customers = []
        processed_subscriptions = []
        
        for i, customer in enumerate(customers):
            # Clean and process customer data
            user_id = i + 1
            
            # Handle missing values and convert data types
            monthly_charges = float(customer.get('MonthlyCharges', '0').replace(' ', '') or '0')
            total_charges = customer.get('TotalCharges', '0').replace(' ', '')
            if total_charges == '':
                total_charges = '0'
            total_charges = float(total_charges)
            
            tenure = int(customer.get('tenure', '0'))
            
            # Calculate derived metrics
            avg_monthly_revenue = monthly_charges
            churn_status = 1 if customer.get('Churn', 'No') == 'Yes' else 0
            
            # Customer record
            processed_customer = {
                'user_id': user_id,
                'gender': customer.get('gender', 'Unknown'),
                'senior_citizen': customer.get('SeniorCitizen', '0'),
                'partner': customer.get('Partner', 'No'),
                'dependents': customer.get('Dependents', 'No'),
                'tenure_months': tenure,
                'phone_service': customer.get('PhoneService', 'No'),
                'internet_service': customer.get('InternetService', 'No'),
                'monthly_charges': monthly_charges,
                'total_charges': total_charges,
                'churn': churn_status,
                'contract': customer.get('Contract', 'Month-to-month'),
                'payment_method': customer.get('PaymentMethod', 'Electronic check')
            }
            processed_customers.append(processed_customer)
            
            # Subscription record
            subscription = {
                'subscription_id': f"telco_{user_id}",
                'user_id': user_id,
                'plan_type': customer.get('Contract', 'Month-to-month'),
                'monthly_revenue': monthly_charges,
                'tenure_months': tenure,
                'is_churned': churn_status,
                'internet_service': customer.get('InternetService', 'No'),
                'total_revenue': total_charges
            }
            processed_subscriptions.append(subscription)
        
        # Save processed data
        with open('data/processed/real_customers.csv', 'w', newline='') as file:
            if processed_customers:
                fieldnames = processed_customers[0].keys()
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(processed_customers)
        
        with open('data/processed/real_subscriptions.csv', 'w', newline='') as file:
            if processed_subscriptions:
                fieldnames = processed_subscriptions[0].keys()
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(processed_subscriptions)
        
        print("✅ Processed Telecom dataset successfully")
        
        # Generate summary statistics
        total_revenue = sum(c['total_charges'] for c in processed_customers)
        avg_monthly_revenue = sum(c['monthly_charges'] for c in processed_customers) / len(processed_customers)
        churn_rate = sum(c['churn'] for c in processed_customers) / len(processed_customers)
        avg_tenure = sum(c['tenure_months'] for c in processed_customers) / len(processed_customers)
        
        summary = {
            'dataset': 'IBM Telecom Customer Churn',
            'total_customers': len(processed_customers),
            'total_revenue': round(total_revenue, 2),
            'avg_monthly_revenue': round(avg_monthly_revenue, 2),
            'churn_rate': round(churn_rate * 100, 1),
            'avg_tenure_months': round(avg_tenure, 1),
            'processing_date': '2025-01-15'
        }
        
        return summary
        
    except Exception as e:
        print(f"❌ Error processing Telecom dataset: {e}")
        return None

def main():
    """Main execution function"""
    print("🚀 Starting Real Dataset Integration")
    print("=" * 50)
    
    # Create directories
    create_directories()
    
    # Download datasets
    print("\n📥 Downloading Real Datasets...")
    datasets = download_real_datasets()
    
    if not datasets:
        print("❌ No datasets downloaded successfully")
        return
    
    # Process datasets
    print("\n🔄 Processing Datasets...")
    telco_summary = process_telco_dataset()
    
    if telco_summary:
        print(f"\n📊 Dataset Summary:")
        print(f"   • Total Customers: {telco_summary['total_customers']:,}")
        print(f"   • Total Revenue: ${telco_summary['total_revenue']:,.2f}")
        print(f"   • Avg Monthly Revenue: ${telco_summary['avg_monthly_revenue']:.2f}")
        print(f"   • Churn Rate: {telco_summary['churn_rate']}%")
        print(f"   • Avg Tenure: {telco_summary['avg_tenure_months']} months")
    
    print("\n✅ Real Dataset Integration Complete!")

if __name__ == "__main__":
    main()