import joblib
import numpy as np
import pandas as pd

# Load the model
model = joblib.load("models/productivity_model.pkl")

# Define all expected columns in the right order
columns = [
    'age', 'daily_social_media_time', 'number_of_notifications', 'work_hours_per_day',
    'perceived_productivity_score', 'sleep_hours', 'screen_time_before_sleep',
    'breaks_during_work', 'coffee_consumption_per_day', 'days_feeling_burnout_per_month',
    'weekly_offline_hours', 'job_satisfaction_score',
    'gender_Male', 'gender_Other',
    'job_type_Finance', 'job_type_Health', 'job_type_IT', 'job_type_Student', 'job_type_Unemployed',
    'social_platform_preference_Instagram', 'social_platform_preference_Telegram',
    'social_platform_preference_TikTok', 'social_platform_preference_Twitter',
    'uses_focus_apps_True', 'has_digital_wellbeing_enabled_True'
]

# Basic inputs
print("Please enter the following info:")
age = float(input("Age: "))
daily_time = float(input("Daily Social Media Time (in hours): "))
notifications = int(input("Number of Notifications per day: "))
work_hours = float(input("Work Hours per Day: "))
perceived_score = float(input("Perceived Productivity Score (1-10): "))
sleep = float(input("Sleep Hours: "))
screen_time = float(input("Screen Time before Sleep (hours): "))
breaks = int(input("Breaks during Work: "))
coffee = int(input("Coffee Consumption per Day: "))
burnout_days = int(input("Days Feeling Burnout per Month: "))
offline_hours = float(input("Weekly Offline Hours: "))
job_sat = float(input("Job Satisfaction Score (1-10): "))

# One-hot categorical inputs (simplified)
gender = input("Gender (Male/Female/Other): ").strip().lower()
job = input("Job Type (Finance/Health/IT/Student/Unemployed): ").strip().lower()
platform = input("Preferred Social Platform (Instagram/Telegram/TikTok/Twitter): ").strip().lower()
uses_focus = input("Uses Focus Apps? (yes/no): ").strip().lower() == "yes"
wellbeing = input("Digital Wellbeing Enabled? (yes/no): ").strip().lower() == "yes"

# Create input data row
row = [age, daily_time, notifications, work_hours, perceived_score, sleep, screen_time,
       breaks, coffee, burnout_days, offline_hours, job_sat]

# Add boolean columns
row += [
    gender == "male", gender == "other",
    job == "finance", job == "health", job == "it", job == "student", job == "unemployed",
    platform == "instagram", platform == "telegram", platform == "tiktok", platform == "twitter",
    uses_focus, wellbeing
]

# Convert to DataFrame
X_input = pd.DataFrame([row], columns=columns)

# Predict
prediction = model.predict(X_input)[0]
print(f"\n🎯 Predicted Actual Productivity Score: {prediction:.2f}")
