import calendar
from datetime import date, timedelta
from typing import Dict, List

import pandas as pd
import streamlit as st

from diet_plan import (
    ActivityLevel,
    DietType,
    Gender,
    Goal,
    Region,
    render_diet_plan_ui,
)
from food import render_nutrition_ui
import streamlit as st
import openai
import pydantic

st.write("✅ Environment check:")
st.write("Streamlit:", st.__version__)
st.write("OpenAI:", openai.__version__)
st.write("Pydantic:", pydantic.__version__)


def _get_saved_profile() -> Dict:
    return st.session_state.get("diet_profile", {})


def _render_profile_settings() -> None:
    st.subheader("👤 Personal Profile")
    default_profile = st.session_state.get(
        "diet_profile",
        {
            "name": "",
            "age": 30,
            "gender": "Male",
            "height_cm": 175,
            "current_weight_kg": 70.0,
            "goal": "Weight Loss",
            "target_weight_kg": 65.0,
            "activity_level": "Moderate",
            "region": "South India",
            "cuisine_preference": "Vegetarian",
            "health_conditions": [],
            "allergies": [],
        },
    )

    goal_options = [g.value for g in Goal]
    activity_options = [a.value for a in ActivityLevel]
    region_options = [r.value for r in Region]
    diet_options = [d.value for d in DietType]
    health_options = ["Diabetes", "High BP", "PCOD", "Thyroid", "High Cholesterol"]

    with st.form("profile_settings_form"):
        col_name, col_age, col_gender = st.columns([2, 1, 1])
        name = col_name.text_input("Name", value=default_profile.get("name", ""), placeholder="e.g., Asha")
        age = col_age.number_input("Age", min_value=11, max_value=119, value=int(default_profile.get("age", 30) or 30))
        gender = col_gender.selectbox("Gender", [g.value for g in Gender], index=[g.value for g in Gender].index(default_profile.get("gender", Gender.MALE.value)))

        col_height, col_weight = st.columns(2)
        height_cm = col_height.number_input("Height (cm)", min_value=51, max_value=299, value=int(default_profile.get("height_cm", 175) or 175))
        current_weight_kg = col_weight.number_input("Current weight (kg)", min_value=21.0, max_value=299.0, value=float(default_profile.get("current_weight_kg", 70.0) or 70.0))

        col_goal, col_target, col_activity = st.columns(3)
        goal = col_goal.selectbox("Goal", goal_options, index=goal_options.index(default_profile.get("goal", Goal.WEIGHT_LOSS.value)))

        target_weight_kg = None
        if goal != Goal.MAINTENANCE.value:
            target_weight_kg = col_target.number_input(
                "Target weight (kg)",
                min_value=21.0,
                max_value=299.0,
                value=float(default_profile.get("target_weight_kg", max(current_weight_kg - 5, 21.0)) or max(current_weight_kg - 5, 21.0)),
            )
        else:
            col_target.markdown("Target weight not required for maintenance.")

        activity_level = col_activity.selectbox(
            "Activity level",
            activity_options,
            index=activity_options.index(default_profile.get("activity_level", ActivityLevel.MODERATE.value)),
        )

        col_region, col_diet = st.columns(2)
        region = col_region.selectbox("Preferred region", region_options, index=region_options.index(default_profile.get("region", Region.SOUTH_INDIA.value)))
        cuisine_preference = col_diet.selectbox("Diet type", diet_options, index=diet_options.index(default_profile.get("cuisine_preference", DietType.VEGETARIAN.value)))

        health_conditions = st.multiselect(
            "Health conditions",
            health_options,
            default=[condition for condition in default_profile.get("health_conditions", []) if condition in health_options],
        )
        allergies_text = st.text_input("Allergies (comma separated)", value=", ".join(default_profile.get("allergies", [])), placeholder="e.g., peanuts, soy")
        submitted = st.form_submit_button("Save profile", type="primary", use_container_width=True)

    if submitted:
        st.session_state["diet_profile"] = {
            "name": name.strip(),
            "age": age,
            "gender": gender,
            "height_cm": height_cm,
            "current_weight_kg": current_weight_kg,
            "goal": goal,
            "target_weight_kg": target_weight_kg,
            "activity_level": activity_level,
            "region": region,
            "cuisine_preference": cuisine_preference,
            "health_conditions": health_conditions,
            "allergies": [a.strip() for a in allergies_text.split(",") if a.strip()],
        }
        st.success("Profile updated. Use the Diet Planner tab to generate a plan.")

    profile = st.session_state.get("diet_profile")
    if profile:
        st.markdown("#### Current profile")
        col_left, col_right = st.columns(2)
        with col_left:
            st.write(f"- **Name:** {profile.get('name') or '—'}")
            st.write(f"- **Age:** {profile.get('age')} years")
            st.write(f"- **Gender:** {profile.get('gender')}")
            st.write(f"- **Height:** {profile.get('height_cm')} cm")
            st.write(f"- **Current weight:** {profile.get('current_weight_kg')} kg")
        with col_right:
            st.write(f"- **Goal:** {profile.get('goal')}")
            if profile.get("target_weight_kg"):
                st.write(f"- **Target weight:** {profile.get('target_weight_kg')} kg")
            st.write(f"- **Activity:** {profile.get('activity_level')}")
            st.write(f"- **Region:** {profile.get('region')}")
            st.write(f"- **Diet type:** {profile.get('cuisine_preference')}")
        if profile.get("health_conditions"):
            st.caption("Health conditions: " + ", ".join(profile["health_conditions"]))
        if profile.get("allergies"):
            st.caption("Allergies: " + ", ".join(profile["allergies"]))
def _render_dashboard() -> None:
    today = date.today()
    profile = _get_saved_profile()
    plan_result = st.session_state.get("diet_plan_result")
    hydration_log = st.session_state.get("hydration_log", {})
    foods_state = st.session_state.get("nutrition_foods_list", [])

    today_iso = today.isoformat()
    todays_plan = plan_result["schedule"].get(today_iso) if plan_result else None
    todays_water = hydration_log.get(today_iso)

    st.subheader("Wellness Snapshot")
    st.caption("Keep your nutrition plan, meal checks, and hydration cues aligned each day.")

    col_metrics = st.columns(4)
    plan_status = "Ready" if plan_result else "Not generated"
    col_metrics[0].metric("Plan status", plan_status, help="Generate via Diet Planner")
    col_metrics[1].metric("Today's goal", f"{todays_plan['total_calories']} cal" if todays_plan else "—")
    col_metrics[2].metric("Water logged", f"{todays_water:.1f} L" if todays_water is not None else "Log pending")
    col_metrics[3].metric("Meals queued", len(foods_state or []))

    st.markdown("### Today at a glance")
    col_plan, col_meals = st.columns((3, 2))

    with col_plan:
        if todays_plan:
            macros_summary = {
                "Calories": todays_plan["total_calories"],
                "Protein (g)": round(todays_plan["total_protein_g"], 1),
                "Carbs (g)": round(todays_plan["total_carbs_g"], 1),
                "Fat (g)": round(todays_plan["total_fat_g"], 1),
            }
            macro_rows = [{"Macro": key, "Target": value} for key, value in macros_summary.items()]
            st.table(macro_rows)

            meals = todays_plan.get("meals", {})
            preview = [
                {
                    "Meal": meal_name.replace("_", " ").title(),
                    "Calories": data.get("meal_total_calories", 0),
                    "Signature dish": data.get("items", [{}])[0].get("food", "—"),
                }
                for meal_name, data in meals.items()
            ]
            if preview:
                st.markdown("#### Today's menu")
                st.dataframe(preview, use_container_width=True, hide_index=True)
        else:
            st.info("Generate a 7-day plan to see today's targets and menu.")

    with col_meals:
        if foods_state:
            st.markdown("#### Meal analyzer queue")
            for idx, entry in enumerate(foods_state[:5], start=1):
                st.markdown(f"{idx}. {entry['item']}")
            if len(foods_state) > 5:
                st.caption(f"...and {len(foods_state) - 5} more")
        else:
            st.caption("Use the Nutrition Analyzer to stage a meal for macro review.")

        if todays_water is not None and plan_result:
            st.metric("Hydration logged", f"{todays_water:.1f} L", delta=None)

    hydration_points = []
    for iso, amount in hydration_log.items():
        entry_date = date.fromisoformat(iso)
        if entry_date >= today - timedelta(days=6):
            hydration_points.append({"Date": entry_date, "Water (L)": amount})
    if hydration_points:
        df_hydration = pd.DataFrame(hydration_points).sort_values("Date").set_index("Date")
        st.markdown("#### Hydration trend (past 7 days)")
        st.bar_chart(df_hydration)


def _render_hydration_coach() -> None:
    st.subheader("💧 Hydration Coach")
    profile = _get_saved_profile()
    default_weight = float(profile.get("current_weight_kg") or 70.0)
    hydration_log = st.session_state.setdefault("hydration_log", {})

    col_weight, col_activity, col_climate = st.columns(3)
    weight = col_weight.number_input("Body weight (kg)", min_value=30.0, max_value=200.0, value=default_weight, step=0.5)
    activity_minutes = col_activity.slider("Active minutes today", min_value=0, max_value=180, value=30, step=10)
    climate = col_climate.selectbox("Climate", ["Temperate", "Warm & humid", "Hot & dry"])

    base_water_ml = weight * 35
    activity_bonus = activity_minutes * 12
    climate_multiplier = {"Temperate": 1.0, "Warm & humid": 1.1, "Hot & dry": 1.2}[climate]
    total_ml = (base_water_ml + activity_bonus) * climate_multiplier
    total_l = total_ml / 1000

    col_summary = st.columns(3)
    col_summary[0].metric("Daily target", f"{total_l:.2f} L")
    col_summary[1].metric("Base need", f"{base_water_ml/1000:.2f} L")
    col_summary[2].metric("Activity bonus", f"{activity_bonus/1000:.2f} L")

    st.markdown("### Log intake")
    selected_date = st.date_input("Select date", value=date.today(), key="hydration_log_date")
    selected_iso = selected_date.isoformat()
    existing = float(hydration_log.get(selected_iso, round(total_l, 1)))
    consumed = st.number_input(
        "Water consumed (L)",
        min_value=0.0,
        max_value=10.0,
        value=existing,
        step=0.1,
        key=f"hydration_consumed_{selected_iso}",
    )
    if st.button("Save hydration log", use_container_width=True):
        hydration_log[selected_iso] = round(consumed, 2)
        st.success(f"Logged {consumed:.2f} L for {selected_date.strftime('%b %d')}.")

    st.caption("Hydration entries feed directly into the calendar view for tracking.")


def render_wellness_calendar() -> None:
    st.subheader("🗓️ Wellness Calendar")
    st.caption("Track plan adherence and hydration without leaving the hub.")

    today = date.today()
    plan_result = st.session_state.get("diet_plan_result", {})
    plan_schedule: Dict[str, Dict] = plan_result.get("schedule", {}) if plan_result else {}
    hydration_log = st.session_state.setdefault("hydration_log", {})
    plan_adherence = st.session_state.setdefault("plan_adherence", {})

    calendar_events: Dict[str, List[Dict[str, str]]] = {}
    for date_iso, day_plan in plan_schedule.items():
        calendar_events.setdefault(date_iso, []).append(
            {
                "type": "Diet Plan",
                "notes": f"{day_plan['total_calories']} cal target",
                "achieved": plan_adherence.get(date_iso, False),
            }
        )
    for date_iso, liters in hydration_log.items():
        calendar_events.setdefault(date_iso, []).append(
            {"type": "Hydration", "notes": f"{liters:.1f} L logged"}
        )

    focus_options = sorted({entry["type"] for entries in calendar_events.values() for entry in entries}) or ["Diet Plan", "Hydration"]
    col_filters, col_summary = st.columns([2, 1])
    with col_filters:
        selected_focus = st.multiselect(
            "Focus filter",
            focus_options,
            default=focus_options,
            help="Toggle categories to tailor the view.",
        )
    with col_summary:
        total_items = sum(len(entries) for entries in calendar_events.values())
        st.metric("Items tracked", total_items)

    filtered_events: Dict[str, List[Dict[str, str]]] = {}
    for key, entry_list in calendar_events.items():
        kept = [entry for entry in entry_list if not selected_focus or entry["type"] in selected_focus]
        if kept:
            filtered_events[key] = kept

    col_calendar, col_schedule = st.columns([3, 2])

    with col_calendar:
        col_month, col_year = st.columns([2, 1])
        month = col_month.selectbox(
            "Month",
            list(range(1, 13)),
            index=today.month - 1,
            format_func=lambda m: calendar.month_name[m],
        )
        year = col_year.number_input("Year", min_value=2020, max_value=2100, value=today.year)

        month_calendar = calendar.monthcalendar(year, month)
        weekdays = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

        table_html = "<table style='width:100%; text-align:center; border-collapse:collapse;'>"
        table_html += "<tr>" + "".join(f"<th style='padding:6px; border-bottom:1px solid #ddd;'>{day}</th>" for day in weekdays) + "</tr>"
        for week in month_calendar:
            table_html += "<tr>"
            for day in week:
                if day == 0:
                    table_html += "<td style='padding:12px; color:#bbb;'>—</td>"
                    continue
                day_date = date(year, month, day)
                key = day_date.isoformat()
                markers = ""
                if key in filtered_events:
                    icons = []
                    for entry in filtered_events[key]:
                        if entry["type"] == "Diet Plan":
                            icons.append("✅" if entry.get("achieved") else "🎯")
                        elif entry["type"] == "Hydration":
                            icons.append("💧")
                    markers = "<div style='margin-top:4px; font-size:12px;'>" + " ".join(icons[:3]) + "</div>"
                achieved_today = any(
                    entry["type"] == "Diet Plan" and entry.get("achieved")
                    for entry in filtered_events.get(key, [])
                )
                background = "background-color:#e6f4ea;" if achieved_today else ""
                table_html += f"<td style='padding:12px; border-bottom:1px solid #f0f0f0; border-right:1px solid #f0f0f0; {background}'>{day}{markers}</td>"
            table_html += "</tr>"
        table_html += "</table>"
        st.markdown(table_html, unsafe_allow_html=True)

    with col_schedule:
        st.markdown("#### Plan & hydration log")
        event_keys = sorted(filtered_events.keys()) or [today.isoformat()]
        default_index = 0
        if today.isoformat() in event_keys:
            default_index = event_keys.index(today.isoformat())
        selected_iso = st.selectbox(
            "Select date",
            options=event_keys,
            index=default_index if default_index < len(event_keys) else 0,
            format_func=lambda iso: date.fromisoformat(iso).strftime("%a, %d %b"),
            key="calendar_selected_iso",
        )
        selected_date = date.fromisoformat(selected_iso)

        if selected_iso in plan_schedule:
            day_plan = plan_schedule[selected_iso]
            st.metric("Plan target", f"{day_plan['total_calories']} cal")
            st.metric("Protein focus", f"{round(day_plan['total_protein_g'], 1)} g")

            achieved = plan_adherence.get(selected_iso, False)
            if achieved:
                st.success("Plan marked as achieved.")
                if st.button("Mark as not achieved", key=f"plan_unmark_{selected_iso}"):
                    plan_adherence[selected_iso] = False
                    st.session_state["plan_adherence"] = plan_adherence
            else:
                st.warning("Plan not marked as achieved.")
                if st.button("Mark as achieved", key=f"plan_mark_{selected_iso}"):
                    plan_adherence[selected_iso] = True
                    st.session_state["plan_adherence"] = plan_adherence
        else:
            st.caption("No plan scheduled for this date.")

        hydration_value = hydration_log.get(selected_iso)
        if hydration_value is not None:
            st.metric("Water logged", f"{hydration_value:.1f} L")
        else:
            st.caption("No hydration entry logged. Record intake via the Hydration Coach tab.")

def main() -> None:
    st.set_page_config(
        page_title="FitFuel Wellness Hub",
        page_icon="🍽️",
        layout="wide",
    )

    st.title("FitFuel Wellness Hub")
    st.caption("An integrated fitness companion for diet planning, meal analysis, hydration, and training insights.")

    tabs = st.tabs([
        "🏠 Dashboard",
        "👤 Profile",
        "🥗 Diet Planner",
        "🍽️ Nutrition Analyzer",
        "💧 Hydration Coach",
        "🗓️ Wellness Calendar",
    ])

    dashboard_tab, profile_tab, diet_tab, nutrition_tab, hydration_tab, calendar_tab = tabs

    with dashboard_tab:
        _render_dashboard()

    with profile_tab:
        _render_profile_settings()

    with diet_tab:
        render_diet_plan_ui(show_title=False)

    with nutrition_tab:
        render_nutrition_ui()

    with hydration_tab:
        _render_hydration_coach()

    with calendar_tab:
        render_wellness_calendar()


if __name__ == "__main__":
    main()
