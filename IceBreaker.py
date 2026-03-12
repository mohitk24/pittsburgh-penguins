# IceBreaker.py
from penguins_predictor import predict_and_visualize

# Example athlete; later you can load from CSV/UI instead of hardcoding

# ---------- 2) BUILD SINGLE-PLAYER DATAFRAME (RAW) ----------
athlete_112 = {
   'id': 112, 
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

if __name__ == "__main__":
    predict_and_visualize(athlete_112)
