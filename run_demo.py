#!/usr/bin/env python3
"""
Subscription Churn Analysis - Complete Demo Runner
Executes the full analysis pipeline for demonstration
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def print_header(text):
    """Print a formatted header"""
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60)

def print_step(step_num, description):
    """Print a step description"""
    print(f"\n📋 Step {step_num}: {description}")
    print("-" * 50)

def run_command(command, description, directory=None):
    """Run a command and display results"""
    print(f"🔄 {description}")
    
    try:
        if directory:
            result = subprocess.run(command, cwd=directory, shell=True, 
                                  capture_output=True, text=True, timeout=300)
        else:
            result = subprocess.run(command, shell=True, 
                                  capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
            if result.stdout:
                print(f"Output: {result.stdout[:200]}...")
        else:
            print(f"❌ {description} failed")
            print(f"Error: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print(f"⏱️ {description} timed out")
        return False
    except Exception as e:
        print(f"❌ Error running {description}: {e}")
        return False
    
    return True

def check_dependencies():
    """Check if required packages are installed"""
    print_step(0, "Checking Dependencies")
    
    required_packages = [
        'pandas', 'numpy', 'matplotlib', 'seaborn', 'plotly', 
        'scikit-learn', 'streamlit', 'faker'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} found")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} missing")
    
    if missing_packages:
        print(f"\n📦 Installing missing packages: {', '.join(missing_packages)}")
        install_cmd = f"pip install {' '.join(missing_packages)}"
        if not run_command(install_cmd, "Installing dependencies"):
            print("❌ Failed to install dependencies. Please install manually.")
            return False
    
    return True

def generate_data():
    """Generate synthetic subscription data"""
    print_step(1, "Generating Synthetic Data")
    
    # Change to python directory
    python_dir = Path("python")
    if not python_dir.exists():
        print("❌ Python directory not found")
        return False
    
    # Create data directories
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    (data_dir / "raw").mkdir(exist_ok=True)
    (data_dir / "processed").mkdir(exist_ok=True)
    
    return run_command("python data_generation.py", "Generating data", python_dir)

def run_analysis():
    """Run the churn analysis"""
    print_step(2, "Running Churn Analysis")
    
    python_dir = Path("python")
    return run_command("python churn_analysis.py", "Running analysis", python_dir)

def run_scenarios():
    """Run scenario modeling"""
    print_step(3, "Running Scenario Models")
    
    python_dir = Path("python")
    return run_command("python scenario_models.py", "Running scenarios", python_dir)

def create_dashboard_info():
    """Create dashboard information"""
    print_step(4, "Dashboard Setup")
    
    print("📊 Streamlit Dashboard Ready!")
    print("\nTo launch the interactive dashboard:")
    print("1. Navigate to the dashboard directory: cd dashboard")
    print("2. Run the dashboard: streamlit run app.py")
    print("3. Open browser to: http://localhost:8501")
    
    print("\n🌐 Or run from project root:")
    print("   streamlit run dashboard/app.py")
    
    return True

def display_results():
    """Display analysis results and next steps"""
    print_step(5, "Results Summary")
    
    # Check if files were created
    files_to_check = [
        "data/raw/users.csv",
        "data/raw/subscriptions.csv",
        "data/raw/engagement.csv",
        "data/raw/support_tickets.csv",
        "presentation/executive_summary.md"
    ]
    
    print("📁 Generated Files:")
    for file_path in files_to_check:
        if Path(file_path).exists():
            size = Path(file_path).stat().st_size
            print(f"✅ {file_path} ({size:,} bytes)")
        else:
            print(f"❌ {file_path} not found")
    
    print("\n📋 Project Components:")
    print("├── 📊 Synthetic Dataset (100K+ users, 24 months)")
    print("├── 🔍 SQL Analytics Queries")  
    print("├── 🤖 Machine Learning Churn Model")
    print("├── 📈 Interactive Streamlit Dashboard")
    print("├── 🎯 What-If Scenario Models")
    print("└── 📑 Executive Summary & Presentation")
    
    print("\n🎯 Key Metrics Analyzed:")
    print("• Monthly Churn Rate & Trends")
    print("• Customer Lifetime Value (LTV)")
    print("• Cohort Retention Analysis")
    print("• Revenue Impact Projections")
    print("• Feature Importance for Churn")
    
    return True

def main():
    """Main execution function"""
    print_header("SUBSCRIPTION CHURN & REVENUE ANALYSIS")
    print("🚀 Netflix/Spotify/Amazon Prime Style Analytics Demo")
    print("\nThis demo will:")
    print("1. Generate synthetic subscription data (100K users)")
    print("2. Run comprehensive churn analysis")
    print("3. Create predictive models")
    print("4. Build interactive dashboard")
    print("5. Generate business recommendations")
    
    # Check if user wants to continue
    response = input("\n🤔 Continue with full demo? (y/n): ").lower().strip()
    if response not in ['y', 'yes']:
        print("Demo cancelled.")
        return
    
    start_time = time.time()
    
    # Run all steps
    steps = [
        (check_dependencies, "Dependencies checked"),
        (generate_data, "Data generated"),
        (run_analysis, "Analysis completed"),
        (run_scenarios, "Scenarios modeled"),
        (create_dashboard_info, "Dashboard prepared"),
        (display_results, "Results summarized")
    ]
    
    for step_func, success_msg in steps:
        if not step_func():
            print(f"\n❌ Demo stopped due to error in {step_func.__name__}")
            return
        print(f"✅ {success_msg}")
        time.sleep(1)  # Brief pause for readability
    
    # Final summary
    elapsed_time = time.time() - start_time
    print_header("DEMO COMPLETED SUCCESSFULLY! 🎉")
    
    print(f"⏱️ Total execution time: {elapsed_time:.1f} seconds")
    print("\n📋 Next Steps:")
    print("1. 📊 Launch dashboard: streamlit run dashboard/app.py")
    print("2. 📖 Read executive summary: presentation/executive_summary.md")
    print("3. 🔍 Explore SQL queries: sql/analytics/")
    print("4. 🤖 Review ML models: python/churn_analysis.py")
    print("5. 🎯 Check scenarios: python/scenario_models.py")
    
    print("\n💼 Portfolio Ready:")
    print("✅ Complete end-to-end analytics project")
    print("✅ Real-world business problem solving")
    print("✅ Advanced SQL, Python, ML, and visualization")
    print("✅ Executive-ready presentation materials")
    print("✅ Interactive dashboard for stakeholders")
    
    print("\n🌟 This project demonstrates skills for roles at:")
    print("   Netflix, Spotify, Amazon Prime, Meta, Google, etc.")
    
    print("\n📧 Ready to impress hiring managers and stakeholders!")

if __name__ == "__main__":
    main()