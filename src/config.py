"""
Central configuration. Change thresholds HERE - nothing else in the project hard-codes them.
"""

from pathlib import Path



ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw" / "customer_churn_dataset.csv"
DATA_CLEAN = ROOT / "data" / "processed" / "customer_churn_clean.csv"
DATA_FEATURES = ROOT / "data" / "processed" / "customer_features.csv"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

RANDOM_STATE = 42
TEST_SIZE = 0.20
TARGET = "Churn"


BLUE = "#2a78d6" 
ORANGE = "#eb6834" 
INK = "#0b0b0b" 
MUTED = "#52514e" 
GRID = "#e4e3df"
AQUA = "#1baf7a"
YELLOW = "#eda100"
WHITE = "#f1f0ed"


# ---------------- Risk categories ----------------
# probability < RISK_LOW_MAX                -> Low
# RISK_LOW_MAX <= probability < HIGH_MIN    -> Medium
# probability >= RISK_HIGH_MIN              -> High
RISK_LOW_MAX = 0.30
RISK_HIGH_MIN = 0.70


# ---------------- Customer value tiers ----------------
# Tiers are cut on the value column using these quantiles (computed on training data).
VALUE_COLUMN = "Annual_Revenue_At_Risk"   # = Monthly_Charges x 12: what we lose if the customer leaves.

# (Customer_Lifetime_Value = Monthly x Tenure looks BACKWARD; new customers churn most yet look
#  low-value on it, so it would hide exactly the customers worth saving. Switch here to compare.)
VALUE_LOW_QUANTILE = 0.33     # below  -> Low value  (bottom third)
VALUE_HIGH_QUANTILE = 0.67    # above  -> High value (top third)


# ---------------- Behaviour rules ----------------
INACTIVITY_DAYS = 14          # Last_Login_Days > this      -> inactive
LOW_USAGE_HOURS = 12          # Usage_Hours < this          -> low usage
HIGH_COMPLAINTS = 2           # Complaints >= this          -> many complaints
HIGH_TICKETS = 3              # Support_Tickets >= this     -> many tickets
LOW_SATISFACTION = 2          # Satisfaction <= this        -> unhappy
HIGH_CHARGES = 850            # Monthly_Charges >= this     -> top-tier price (Premium plan range)

RULE_HIGH_MIN_SIGNALS = 3     # >= this many warning signals -> High (rule-based)
RULE_MEDIUM_MIN_SIGNALS = 1   # >= this many                -> Medium, else Low


# Engagement score caps (fixed business caps, NOT data min/max, so no leakage from the test set)
LOGIN_CAP = 30                # logins/month treated as "fully engaged"
USAGE_CAP = 60                # usage hours/month treated as "fully engaged"
RECENCY_CAP_DAYS = 30         # 30+ days since login = zero recency credit
ENGAGEMENT_WEIGHTS = {"logins": 0.4, "usage": 0.4, "recency": 0.2}


# Support risk = tickets + COMPLAINT_WEIGHT * complaints
COMPLAINT_WEIGHT = 2


# ---------------- Feature groups ----------------
NUMERIC_FEATURES = [
    "Age", 
    "Tenure_Months", 
    "Monthly_Charges", 
    "Login_Frequency", 
    "Support_Tickets",
    "Complaints", 
    "Discount_Received", 
    "Last_Login_Days", 
    "Usage_Hours",
    "Customer_Satisfaction", 
    "Customer_Lifetime_Value", 
    "Engagement_Score", 
    "Support_Risk",
]

# Ordered categories -> ordinal ("label") encoding keeps the order information
ORDINAL_FEATURES = {
    "Contract_Type": [
        "Month-to-Month", 
        "One Year", 
        "Two Year"
    ],
    "Subscription_Type": [
        "Basic", 
        "Standard", 
        "Premium"
    ]
}

# Unordered categories -> one-hot encoding
NOMINAL_FEATURES = ["Gender", "Location", "Payment_Method"]
BINARY_FEATURES = ["Inactivity_Flag"]

# Total_Charges is excluded: it is ~Monthly_Charges x Tenure (corr 0.75 with tenure),
# which Customer_Lifetime_Value already captures.
ALL_FEATURES = NUMERIC_FEATURES + list(ORDINAL_FEATURES) + NOMINAL_FEATURES + BINARY_FEATURES


# ---------------- Model selection ----------------
MIN_RECALL_FOR_SELECTION = 0.60   # business rule: a useful model must catch >= 60% of churners
