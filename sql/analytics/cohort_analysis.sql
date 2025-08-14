-- Cohort Retention Analysis for Subscription Service
-- Analyzes user retention by monthly cohorts

-- 1. Monthly Cohort Creation
WITH cohort_base AS (
    SELECT 
        user_id,
        DATE_TRUNC('month', signup_date) AS cohort_month,
        signup_date
    FROM users
),

-- 2. User Activity by Month
user_activity AS (
    SELECT 
        s.user_id,
        DATE_TRUNC('month', s.billing_date) AS activity_month,
        s.status,
        c.cohort_month
    FROM subscriptions s
    JOIN cohort_base c ON s.user_id = c.user_id
),

-- 3. Cohort Size
cohort_size AS (
    SELECT 
        cohort_month,
        COUNT(DISTINCT user_id) AS cohort_users
    FROM cohort_base
    GROUP BY cohort_month
),

-- 4. Retention Calculation
retention_table AS (
    SELECT 
        c.cohort_month,
        EXTRACT(MONTH FROM AGE(u.activity_month, c.cohort_month)) AS months_since_signup,
        COUNT(DISTINCT u.user_id) AS retained_users
    FROM cohort_base c
    LEFT JOIN user_activity u ON c.user_id = u.user_id
    WHERE u.status = 'active'
    GROUP BY c.cohort_month, months_since_signup
)

-- 5. Final Retention Matrix
SELECT 
    r.cohort_month,
    cs.cohort_users,
    r.months_since_signup,
    r.retained_users,
    ROUND(100.0 * r.retained_users / cs.cohort_users, 2) AS retention_rate
FROM retention_table r
JOIN cohort_size cs ON r.cohort_month = cs.cohort_month
ORDER BY r.cohort_month, r.months_since_signup;

-- 6. Churn Rate Analysis by Cohort
WITH monthly_churn AS (
    SELECT 
        DATE_TRUNC('month', billing_date) AS month,
        COUNT(CASE WHEN status = 'churned' THEN 1 END) AS churned_users,
        COUNT(DISTINCT user_id) AS total_users,
        ROUND(100.0 * COUNT(CASE WHEN status = 'churned' THEN 1 END) / 
              NULLIF(COUNT(DISTINCT user_id), 0), 2) AS churn_rate
    FROM subscriptions
    GROUP BY month
)
SELECT 
    month,
    churned_users,
    total_users,
    churn_rate,
    AVG(churn_rate) OVER (ORDER BY month ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS rolling_3mo_avg_churn
FROM monthly_churn
ORDER BY month;

-- 7. Revenue Cohort Analysis
WITH revenue_cohorts AS (
    SELECT 
        DATE_TRUNC('month', u.signup_date) AS cohort_month,
        DATE_TRUNC('month', s.billing_date) AS revenue_month,
        SUM(s.price) AS cohort_revenue,
        COUNT(DISTINCT s.user_id) AS paying_users
    FROM users u
    JOIN subscriptions s ON u.user_id = s.user_id
    WHERE s.status = 'active'
    GROUP BY cohort_month, revenue_month
)
SELECT 
    cohort_month,
    revenue_month,
    EXTRACT(MONTH FROM AGE(revenue_month, cohort_month)) AS months_since_acquisition,
    cohort_revenue,
    paying_users,
    ROUND(cohort_revenue / NULLIF(paying_users, 0), 2) AS arpu
FROM revenue_cohorts
ORDER BY cohort_month, revenue_month;

-- 8. Customer Lifetime Value (CLV/LTV) Calculation
WITH user_revenue AS (
    SELECT 
        u.user_id,
        u.signup_date,
        DATE_TRUNC('month', u.signup_date) AS cohort_month,
        SUM(s.price) AS total_revenue,
        COUNT(DISTINCT DATE_TRUNC('month', s.billing_date)) AS months_active,
        MAX(s.billing_date) AS last_payment_date,
        CASE 
            WHEN MAX(s.status) = 'churned' THEN 1 
            ELSE 0 
        END AS is_churned
    FROM users u
    LEFT JOIN subscriptions s ON u.user_id = s.user_id
    GROUP BY u.user_id, u.signup_date
),
cohort_ltv AS (
    SELECT 
        cohort_month,
        AVG(total_revenue) AS avg_ltv,
        AVG(CASE WHEN is_churned = 1 THEN total_revenue END) AS churned_ltv,
        AVG(CASE WHEN is_churned = 0 THEN total_revenue END) AS active_ltv,
        AVG(months_active) AS avg_months_active,
        COUNT(DISTINCT user_id) AS cohort_size
    FROM user_revenue
    GROUP BY cohort_month
)
SELECT 
    cohort_month,
    cohort_size,
    ROUND(avg_ltv, 2) AS avg_customer_ltv,
    ROUND(churned_ltv, 2) AS churned_customer_ltv,
    ROUND(active_ltv, 2) AS active_customer_ltv,
    ROUND(avg_months_active, 1) AS avg_customer_lifetime_months
FROM cohort_ltv
ORDER BY cohort_month;

-- 9. Churn Prediction Features
WITH user_features AS (
    SELECT 
        u.user_id,
        u.age,
        u.acquisition_channel,
        u.device_type,
        u.payment_method,
        s.plan,
        s.price,
        COALESCE(e.avg_engagement_score, 0) AS avg_engagement_score,
        COALESCE(e.total_hours_watched, 0) AS total_hours_watched,
        COALESCE(e.total_login_days, 0) AS total_login_days,
        COALESCE(t.ticket_count, 0) AS support_tickets,
        COALESCE(t.avg_satisfaction, 0) AS avg_support_satisfaction,
        CASE WHEN MAX(s.status) = 'churned' THEN 1 ELSE 0 END AS churned
    FROM users u
    LEFT JOIN subscriptions s ON u.user_id = s.user_id
    LEFT JOIN (
        SELECT 
            user_id,
            AVG(engagement_score) AS avg_engagement_score,
            SUM(hours_watched) AS total_hours_watched,
            SUM(login_days) AS total_login_days
        FROM engagement
        GROUP BY user_id
    ) e ON u.user_id = e.user_id
    LEFT JOIN (
        SELECT 
            user_id,
            COUNT(*) AS ticket_count,
            AVG(satisfaction_score) AS avg_satisfaction
        FROM support_tickets
        GROUP BY user_id
    ) t ON u.user_id = t.user_id
    GROUP BY u.user_id, u.age, u.acquisition_channel, u.device_type, 
             u.payment_method, s.plan, s.price, e.avg_engagement_score,
             e.total_hours_watched, e.total_login_days, t.ticket_count, t.avg_satisfaction
)
SELECT 
    churned,
    COUNT(*) AS user_count,
    AVG(age) AS avg_age,
    AVG(price) AS avg_price,
    AVG(avg_engagement_score) AS avg_engagement,
    AVG(total_hours_watched) AS avg_hours_watched,
    AVG(support_tickets) AS avg_support_tickets,
    AVG(avg_support_satisfaction) AS avg_satisfaction
FROM user_features
GROUP BY churned;

-- 10. Monthly Recurring Revenue (MRR) Trend
WITH mrr_calculation AS (
    SELECT 
        DATE_TRUNC('month', billing_date) AS month,
        SUM(price) AS mrr,
        COUNT(DISTINCT user_id) AS active_subscribers,
        SUM(price) / NULLIF(COUNT(DISTINCT user_id), 0) AS arpu
    FROM subscriptions
    WHERE status = 'active'
    GROUP BY month
)
SELECT 
    month,
    mrr,
    active_subscribers,
    ROUND(arpu, 2) AS arpu,
    LAG(mrr) OVER (ORDER BY month) AS previous_mrr,
    ROUND(100.0 * (mrr - LAG(mrr) OVER (ORDER BY month)) / 
          NULLIF(LAG(mrr) OVER (ORDER BY month), 0), 2) AS mrr_growth_rate
FROM mrr_calculation
ORDER BY month;