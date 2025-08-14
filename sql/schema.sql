-- Database Schema for Subscription Analytics
-- Tables for users, subscriptions, engagement, and support data

-- Users table - Customer demographics and acquisition info
CREATE TABLE IF NOT EXISTS users (
    user_id INTEGER PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    age INTEGER CHECK (age >= 18 AND age <= 100),
    gender CHAR(1) CHECK (gender IN ('M', 'F', 'Other')),
    country VARCHAR(2),
    city VARCHAR(100),
    signup_date DATE NOT NULL,
    acquisition_channel VARCHAR(50) NOT NULL,
    device_type VARCHAR(20) NOT NULL,
    payment_method VARCHAR(30) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Subscriptions table - Monthly billing and plan information
CREATE TABLE IF NOT EXISTS subscriptions (
    subscription_id VARCHAR(50) PRIMARY KEY,
    user_id INTEGER NOT NULL,
    plan VARCHAR(20) NOT NULL CHECK (plan IN ('Basic', 'Standard', 'Premium', 'Family')),
    price DECIMAL(6,2) NOT NULL CHECK (price > 0),
    billing_date DATE NOT NULL,
    status VARCHAR(10) NOT NULL CHECK (status IN ('active', 'churned', 'paused')),
    churn_date DATE,
    churn_reason VARCHAR(50),
    plan_change BOOLEAN DEFAULT FALSE,
    previous_plan VARCHAR(20),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);

-- Engagement table - User activity and consumption metrics
CREATE TABLE IF NOT EXISTS engagement (
    engagement_id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL,
    month DATE NOT NULL,
    login_days INTEGER CHECK (login_days >= 0 AND login_days <= 31),
    hours_watched DECIMAL(6,1) CHECK (hours_watched >= 0),
    content_items INTEGER CHECK (content_items >= 0),
    downloads INTEGER CHECK (downloads >= 0),
    shares INTEGER CHECK (shares >= 0),
    ratings_given INTEGER CHECK (ratings_given >= 0),
    engagement_score DECIMAL(4,3) CHECK (engagement_score >= 0 AND engagement_score <= 1),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(user_id),
    UNIQUE(user_id, month)
);

-- Support tickets table - Customer service interactions
CREATE TABLE IF NOT EXISTS support_tickets (
    ticket_id VARCHAR(50) PRIMARY KEY,
    user_id INTEGER NOT NULL,
    date DATE NOT NULL,
    category VARCHAR(20) NOT NULL CHECK (category IN ('Billing', 'Technical', 'Content', 'Account', 'Other')),
    priority VARCHAR(10) NOT NULL CHECK (priority IN ('Low', 'Medium', 'High')),
    resolution_time_hours DECIMAL(8,2) CHECK (resolution_time_hours >= 0),
    satisfaction_score INTEGER CHECK (satisfaction_score >= 1 AND satisfaction_score <= 5),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);

-- Indexes for performance optimization
CREATE INDEX IF NOT EXISTS idx_users_signup_date ON users(signup_date);
CREATE INDEX IF NOT EXISTS idx_users_acquisition_channel ON users(acquisition_channel);
CREATE INDEX IF NOT EXISTS idx_subscriptions_user_id ON subscriptions(user_id);
CREATE INDEX IF NOT EXISTS idx_subscriptions_billing_date ON subscriptions(billing_date);
CREATE INDEX IF NOT EXISTS idx_subscriptions_status ON subscriptions(status);
CREATE INDEX IF NOT EXISTS idx_engagement_user_id ON engagement(user_id);
CREATE INDEX IF NOT EXISTS idx_engagement_month ON engagement(month);
CREATE INDEX IF NOT EXISTS idx_tickets_user_id ON support_tickets(user_id);
CREATE INDEX IF NOT EXISTS idx_tickets_date ON support_tickets(date);

-- Views for common analytics queries
CREATE OR REPLACE VIEW monthly_metrics AS
WITH monthly_data AS (
    SELECT 
        DATE_TRUNC('month', billing_date) AS month,
        COUNT(DISTINCT user_id) AS active_users,
        SUM(price) AS mrr,
        COUNT(CASE WHEN status = 'churned' THEN 1 END) AS churned_users
    FROM subscriptions
    GROUP BY DATE_TRUNC('month', billing_date)
)
SELECT 
    month,
    active_users,
    mrr,
    churned_users,
    ROUND(100.0 * churned_users / NULLIF(active_users, 0), 2) AS churn_rate,
    ROUND(mrr / NULLIF(active_users, 0), 2) AS arpu
FROM monthly_data
ORDER BY month;

-- User summary view combining key metrics
CREATE OR REPLACE VIEW user_summary AS
SELECT 
    u.user_id,
    u.age,
    u.acquisition_channel,
    u.device_type,
    u.signup_date,
    s.total_spent,
    s.months_active,
    s.current_status,
    COALESCE(e.avg_engagement, 0) AS avg_engagement_score,
    COALESCE(t.ticket_count, 0) AS support_tickets
FROM users u
LEFT JOIN (
    SELECT 
        user_id,
        SUM(price) as total_spent,
        COUNT(DISTINCT DATE_TRUNC('month', billing_date)) as months_active,
        MAX(status) as current_status
    FROM subscriptions
    GROUP BY user_id
) s ON u.user_id = s.user_id
LEFT JOIN (
    SELECT 
        user_id,
        AVG(engagement_score) as avg_engagement
    FROM engagement
    GROUP BY user_id
) e ON u.user_id = e.user_id
LEFT JOIN (
    SELECT 
        user_id,
        COUNT(*) as ticket_count
    FROM support_tickets
    GROUP BY user_id
) t ON u.user_id = t.user_id;