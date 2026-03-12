# prediction.py

import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
import json

import matplotlib
matplotlib.use('TkAgg')  # or 'TkAgg' on Mac
plt.style.use("seaborn-v0_8-whitegrid")

# ---------- Feature Labels ----------
FEATURE_LABELS = {
    # IDs / demographics
    'birth_year': 'Birth year',
    'gender': 'Gender',
    'category': 'Competition category',
    'age': 'Current age',

    # Baseline physicals
    'weight': 'Weight (lbs)',
    'height': 'Height (in)',
    'pro_agility_l': 'Pro agility (L)',
    'pro_agility_r': 'Pro agility (R)',
    'broad_jump': 'Broad jump (in)',
    'vertical_jump': 'Vertical jump (in)',
    'dash_10yd': '10-yard dash (s)',
    'shuttle_100yd': '100-yard shuttle (s)',
    'shuttle_150yd': '150-yard shuttle (s)',
    'shuttle_300yd': '300-yard shuttle (s)',
    'flexed_arm_hang': 'Flexed arm hang (s)',
    'pull_ups': 'Pull-ups',
    'ue_watt': 'Upper body power (W)',
    'le_watt': 'Lower body power (W)',

    # Demographics & access
    'demo_edu': 'Parent education',
    'demo_income': 'Household income',
    'demo_fin_support': 'Financial support for hockey',
    'demo_sports_access': 'Access to sports facilities',
    'demo_years_org_sport': 'Years in organized sport',
    'demo_race_asian': 'Race: Asian',
    'demo_race_black': 'Race: Black',
    'demo_race_hispanic': 'Race: Hispanic',
    'demo_race_native': 'Race: Native',
    'demo_race_pacific': 'Race: Pacific Islander',
    'demo_race_white': 'Race: White',

    # Psych scales
    'mtq_score': 'Mental toughness score',
    'grt_score': 'Grit score',
    'res_score': 'Resilience score',
    'se_score': 'Self-efficacy score',
    'sms_core': 'Sport motivation core',
    'twc_score': 'Teamwork/connection score',

    # Hockey history & training
    'hst_age_first_league': 'Age first league play',
    'hst_age_regular_hky': 'Age regular hockey',
    'hst_age_supervised': 'Age first supervised training',
    'hst_age_main_sport': 'Age hockey became main sport',
    'hst_practice_hrs_band': 'Weekly practice hours (band)',
    'hst_game_hrs_band': 'Weekly game hours (band)',
    'hst_hky_8plus_months': 'Hockey ≥8 months/yr',
    'hst_multi_sport_level': 'Multi-sport involvement',
    'hst_effort': 'Self-rated effort in training',
    'hst_focus': 'Training focus',
    'hst_enjoyment': 'Enjoyment of hockey',
    'hst_camps_freq': 'Hockey camps frequency',
    'hst_relocated': 'Relocated for hockey',
    'hst_injury_count_12_16': 'Injuries age 12–16',
    'hst_days_missed_2y': 'Days missed last 2 yrs',
    'hst_comp_level': 'Highest competition level',
    'hst_age_first_at_level': 'Age at that level',

    # Cognition / sleep / nutrition
    'cog_focus_score': 'Cognitive focus score',
    'sleep_hrs_band': 'Sleep hours (band)',
    'nut_balanced_meals': 'Balanced meals per day',
    'nut_score': 'Nutrition score',
    'nut_limiting_factors': 'Nutrition limiting factors',
    'nut_energy_mornings': 'Morning energy',
    'cog_decision_intelligence_score': 'Decision intelligence',
    'cog_pregame_anxiety': 'Pre-game anxiety (reversed)',

    # Coping skills
    'cop_breathing_exercises': 'Uses breathing exercises',
    'cop_self_talk': 'Uses self-talk',
    'cop_visualization': 'Uses visualization',
    'cop_talking_to_teammate': 'Talks to teammate',
    'cop_other': 'Other coping strategies',

    # Coach
    'coach_score': 'Coach understanding score',

    # Improvement metrics (examples; adjust text how you like)
    'weight_abs_impr': 'Weight change (abs)',
    'weight_rel_impr_pct': 'Weight change (%)',
    'weight_impr_per_year': 'Weight change per year',
    'height_abs_impr': 'Height change (abs)',
    'height_rel_impr_pct': 'Height change (%)',
    'height_impr_per_year': 'Height change per year',
    'pro_agility_l_abs_impr': 'Pro agility L change (abs)',
    'pro_agility_l_rel_impr_pct': 'Pro agility L change (%)',
    'pro_agility_l_impr_per_year': 'Pro agility L change/yr',
    'pro_agility_r_abs_impr': 'Pro agility R change (abs)',
    'pro_agility_r_rel_impr_pct': 'Pro agility R change (%)',
    'pro_agility_r_impr_per_year': 'Pro agility R change/yr',
    'broad_jump_abs_impr': 'Broad jump change (abs)',
    'broad_jump_rel_impr_pct': 'Broad jump change (%)',
    'broad_jump_impr_per_year': 'Broad jump change/yr',
    'vertical_jump_abs_impr': 'Vertical jump change (abs)',
    'vertical_jump_rel_impr_pct': 'Vertical jump change (%)',
    'vertical_jump_impr_per_year': 'Vertical jump change/yr',
    'dash_10yd_abs_impr': '10-yard dash change (abs)',
    'dash_10yd_rel_impr_pct': '10-yard dash change (%)',
    'dash_10yd_impr_per_year': '10-yard dash change/yr',
    'shuttle_100yd_abs_impr': '100-yard shuttle change (abs)',
    'shuttle_100yd_rel_impr_pct': '100-yard shuttle change (%)',
    'shuttle_100yd_impr_per_year': '100-yard shuttle change/yr',
    'shuttle_150yd_abs_impr': '150-yard shuttle change (abs)',
    'shuttle_150yd_rel_impr_pct': '150-yard shuttle change (%)',
    'shuttle_150yd_impr_per_year': '150-yard shuttle change/yr',
    'shuttle_300yd_abs_impr': '300-yard shuttle change (abs)',
    'shuttle_300yd_rel_impr_pct': '300-yard shuttle change (%)',
    'shuttle_300yd_impr_per_year': '300-yard shuttle change/yr',
    'flexed_arm_hang_abs_impr': 'Flexed arm hang change (abs)',
    'flexed_arm_hang_rel_impr_pct': 'Flexed arm hang change (%)',
    'flexed_arm_hang_impr_per_year': 'Flexed arm hang change/yr',
    'pull_ups_abs_impr': 'Pull-ups change (abs)',
    'pull_ups_rel_impr_pct': 'Pull-ups change (%)',
    'pull_ups_impr_per_year': 'Pull-ups change/yr',
    'ue_watt_abs_impr': 'Upper body power change (abs)',
    'ue_watt_rel_impr_pct': 'Upper body power change (%)',
    'ue_watt_impr_per_year': 'Upper body power change/yr',
    'le_watt_abs_impr': 'Lower body power change (abs)',
    'le_watt_rel_impr_pct': 'Lower body power change (%)',
    'le_watt_impr_per_year': 'Lower body power change/yr',

    # Longitudinal / meta
    'category_progression': 'Category progression over time',
    'num_tests': 'Number of tests',
    'ever_promoted_between_tests': 'Ever promoted between tests',
}

# ---------- 1) LOAD TRAINED OBJECTS ----------
print("Loading model and preprocessing objects...", end="", flush=True)
start = time.time()

pipe = joblib.load("pro_success_pipeline.joblib")  # your saved dict

selector = pipe['selector']
scaler = pipe['scaler']
model = pipe['model']
selected_features = pipe['selected_features']
num_imputer = pipe['num_imputer']
num_features_full = pipe['num_features_full']
X_made_it_scaled = pipe['X_made_it_scaled']
made_it_ids = pipe['made_it_ids']

print(f" done in {time.time() - start:.2f}s\n")

# ---------- EXTRACT COEFFICIENTS FOR FEATURE IMPORTANCE ----------
print("Extracting model coefficients...", end="", flush=True)
coef_df = pd.DataFrame({
    'feature': selected_features,
    'coefficient': model.coef_[0]
})
coef_df['abs_coef'] = coef_df['coefficient'].abs()
coef_df = coef_df.sort_values('abs_coef', ascending=False)
print(" done.\n")

# Print top coefficients
print("Top 15 features driving pro-success prediction:")
print(coef_df.head(15)[['feature', 'coefficient']].round(4).to_string(index=False))

coef_df['feature_label'] = coef_df['feature'].map(
    lambda f: FEATURE_LABELS.get(f, f)
)


# ---------- 2) BUILD SINGLE-PLAYER DATAFRAME (RAW) ----------
athlete_112 = {
   'birth_year': 2004,
   'gender': 1,
   'category': 6,
   # baseline physicals
   'weight': 140.0,
   'height': 65.50,
   'pro_agility_l': 4.18,
   'pro_agility_r': 4.30,
   'broad_jump': 94.00,
   'vertical_jump': 25.50,
   'dash_10yd': 1.65,
   'shuttle_100yd':None,
   'shuttle_150yd': 34,
   'shuttle_300yd': None,
   'flexed_arm_hang': 17,
   'pull_ups':None,
   'ue_watt':None,
   'le_watt':None,
   # demographics / history
   'age': 21,
   'demo_edu': 3,
   'demo_income': 3,
   'demo_fin_support': 0,
   'demo_sports_access': 1,
   'demo_years_org_sport': 5,
   'demo_race_asian': 0,
   'demo_race_black': 0,
   'demo_race_hispanic': 0,
   'demo_race_native': 0,
   'demo_race_pacific': 0,
   'demo_race_white': 1,
   # psych scales
   'mtq_score': 35,
   'grt_score': 26,
   'res_score': 42,
   'se_score': 30,
   'sms_core': 82,
   'twc_score': 23,
   # hockey history & training load
   'hst_age_first_league': 6,
   'hst_age_regular_hky': 6,
   'hst_age_supervised': 6,
   'hst_age_main_sport': 6,
   'hst_practice_hrs_band': 1,
   'hst_game_hrs_band': 1,
   'hst_hky_8plus_months': 0,
   'hst_multi_sport_level': 0,
   'hst_effort': 1,
   'hst_focus': 1,
   'hst_enjoyment': 4,
   'hst_camps_freq': 3,
   'hst_relocated': 1,
   # injuries / comp level (dropped in model but kept for completeness)
   'hst_injury_count_12_16': 0,
   'hst_days_missed_2y': 1,
   'hst_comp_level': 0,
   'hst_age_first_at_level': 0,
   # cognition / sleep / nutrition
   'cog_focus_score': 10,
   'sleep_hrs_band': 3,
   'nut_balanced_meals': 1,
   'nut_score': 11,
   'nut_limiting_factors': 0,
   'nut_energy_mornings': 5,
   'cog_decision_intelligence_score': 15,
   'cog_pregame_anxiety': 5,
   # coping skills
   'cop_breathing_exercises': 0,
   'cop_self_talk': 0,
   'cop_visualization': 1,
   'cop_talking_to_teammate': 0,
   'cop_other': 0,
   # coach
   'coach_score': 10.0,
   # improvements – all zero for this profile
   'weight_abs_impr': 0.00, 'weight_rel_impr_pct': 0.00, 'weight_impr_per_year': 0.00,
   'height_abs_impr': 0.00, 'height_rel_impr_pct': 0.00, 'height_impr_per_year': 0.00,
   'pro_agility_l_abs_impr': 0.00, 'pro_agility_l_rel_impr_pct': 0.00, 'pro_agility_l_impr_per_year': 0.00,
   'pro_agility_r_abs_impr': 0.00, 'pro_agility_r_rel_impr_pct': 0.00, 'pro_agility_r_impr_per_year': 0.00,
   'broad_jump_abs_impr': 0.00, 'broad_jump_rel_impr_pct': 0.00, 'broad_jump_impr_per_year': 0.00,
   'vertical_jump_abs_impr': 0.00, 'vertical_jump_rel_impr_pct': 0.00, 'vertical_jump_impr_per_year': 0.00,
   'dash_10yd_abs_impr': 0.00, 'dash_10yd_rel_impr_pct': 0.00, 'dash_10yd_impr_per_year': 0.00,
   'shuttle_100yd_abs_impr': 0.00, 'shuttle_100yd_rel_impr_pct': 0.00, 'shuttle_100yd_impr_per_year': 0.00,
   'shuttle_150yd_abs_impr': 0.00, 'shuttle_150yd_rel_impr_pct': 0.00, 'shuttle_150yd_impr_per_year': 0.00,
   'shuttle_300yd_abs_impr': 0.00, 'shuttle_300yd_rel_impr_pct': 0.00, 'shuttle_300yd_impr_per_year': 0.00,
   'flexed_arm_hang_abs_impr': 0.00, 'flexed_arm_hang_rel_impr_pct': 0.00, 'flexed_arm_hang_impr_per_year': 0.00,
   'pull_ups_abs_impr': 0.00, 'pull_ups_rel_impr_pct': 0.00, 'pull_ups_impr_per_year': 0.00,
   'ue_watt_abs_impr': 0.00, 'ue_watt_rel_impr_pct': 0.00, 'ue_watt_impr_per_year': 0.00,
   'le_watt_abs_impr': 0.00, 'le_watt_rel_impr_pct': 0.00, 'le_watt_impr_per_year': 0.00,
   'category_progression': 0,
   'num_tests': 1,
   'ever_promoted_between_tests': 0,
}

new_player = pd.DataFrame([athlete_112])
# ensure numeric types for all numeric features used in training
for col in num_features_full:
    if col in new_player.columns:
        new_player[col] = pd.to_numeric(new_player[col], errors='coerce')

# ---------- 3) APPLY SELECTOR + SCALER + MODEL ----------
print("Computing prediction...", end="", flush=True)
time.sleep(0.3)  # tiny pause for effect

# Create measured flags expected by the model
heavy_missing_cols = [
    'shuttle_100yd', 'shuttle_150yd', 'shuttle_300yd',
    'flexed_arm_hang', 'pull_ups', 'ue_watt', 'le_watt'
]
for col in heavy_missing_cols:
    if col in new_player.columns:
        new_player[col + '_measured'] = (~new_player[col].isna()).astype(int)

# apply same median imputer used in training
cols_to_impute = [c for c in num_features_full if c in new_player.columns]
new_player[cols_to_impute] = num_imputer.transform(new_player[cols_to_impute])

# Align to selected features only (missing ones will raise, so intersect)
common_cols = [c for c in selected_features if c in new_player.columns]
X_new = new_player[common_cols]

# Scale
X_new_scaled = scaler.transform(X_new)

# ---------- 4) PREDICT PROBABILITY ----------
proba = model.predict_proba(X_new_scaled)[0, 1]
proba_pct = proba * 100

# ---------- 5) SIMILARITY TO SUCCESSFUL ATHLETES ----------
# X_made_it_scaled and made_it_ids were loaded from the joblib file
sims = cosine_similarity(X_new_scaled, X_made_it_scaled)[0]  # shape (N_pos,)

top_k = 15
top_idx = np.argsort(sims)[-top_k:]
similarity_mean_top_k = float(sims[top_idx].mean())
similarity_max_top_k = float(sims[top_idx].max())

nearest_ids = (
    [made_it_ids[i] for i in top_idx] if made_it_ids is not None else None
)

print(" done.\n")

print(f"Predicted pro-success probability: {proba_pct:.1f}%")
print(f"Mean cosine similarity to top-{top_k} successful athletes: {similarity_mean_top_k:.3f}")
print(f"Max cosine similarity to a successful athlete: {similarity_max_top_k:.3f}")
if nearest_ids is not None:
    print("Nearest successful athlete IDs:", nearest_ids)

# ---------- 6) STRUCTURED RESULTS ----------
result = {
    "athlete_id": "112",
    "pro_probability_pct": round(proba_pct, 1),
    "similarity_mean_top15": round(similarity_mean_top_k, 3),
    "similarity_max": round(similarity_max_top_k, 3),
    "top_matches": nearest_ids[-5:] if nearest_ids else []
}
print("\n=== JSON OUTPUT ===")
print(json.dumps(result, indent=2))

# ---------- 8A) FIGURE 1: SCORES (wide + short) ----------
plt.figure(figsize=(14, 2.5))   # 12 inches wide, 3 inches tall

ax1 = plt.subplot(1, 2, 1)
ax1.barh(['Pro Success'], [proba_pct], color='#2ca02c')
ax1.barh(['Pro Success'], [100 - proba_pct], left=[proba_pct], color='#e0e0e0')
ax1.set_xlim(0, 100)
ax1.text(proba_pct / 2, 0, f"{proba_pct:.1f}%", ha='center', va='center',
         fontweight='bold', color='white', fontsize=12)
ax1.set_title("Pro Success Probability", fontsize=11)

ax2 = plt.subplot(1, 2, 2)
ax2.barh(['Profile Similarity'], [similarity_mean_top_k * 100], color='#1f77b4')
ax2.set_xlim(0, 100)
ax2.text(similarity_mean_top_k * 50, 0, f"{similarity_mean_top_k:.3f}",
         ha='center', va='center', fontweight='bold', color='white', fontsize=12)
ax2.set_xlabel("Cosine Similarity (%)", fontsize=9)
ax2.set_title("Similarity to Known Pros", fontsize=11)

plt.tight_layout()
plt.show()


# ---------- 8B) FIGURE 2: TOP FEATURES ----------
plt.figure(figsize=(8, 6))

top_n = 15
sns.barplot(
    data=coef_df.head(top_n),
    x='coefficient',
    y='feature_label',     # <-- use human-readable labels
    palette='coolwarm'
)
plt.axvline(0, color='black', linewidth=1.2)
plt.title('Top Feature Impacts (Log-odds of pro success)', fontsize=12)
plt.xlabel('Coefficient', fontsize=10)
plt.ylabel('Feature', fontsize=10)
plt.tick_params(axis='y', labelsize=9)

plt.tight_layout()
plt.savefig('penguins_features.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
