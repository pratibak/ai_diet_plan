import csv
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import openai
import streamlit as st

DEFAULT_DB_PATH = Path(__file__).resolve().parent / "archive" / "ifct2017_compositions.csv"
DB_SAMPLE_LIMIT = 350

DB_FIELDS = (
    ("energy_kcal", "Calories", "kcal", 0),
    ("protein_g", "Protein", "g", 1),
    ("fat_g", "Fat", "g", 1),
    ("carbs_g", "Carbs", "g", 1),
    ("fiber_g", "Fiber", "g", 1),
    ("calcium_mg", "Calcium", "mg", 1),
    ("iron_mg", "Iron", "mg", 2),
)

_DB_RECORDS_KEY = "nutrition_db_records"
_DB_TEXT_KEY = "nutrition_db_text"
_DB_NAME_KEY = "nutrition_db_name"
_FOODS_LIST_KEY = "nutrition_foods_list"


def _safe_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _format_number(value: Optional[float], decimals: int) -> Optional[str]:
    if value is None:
        return None
    formatted = f"{value:.{decimals}f}"
    if "." in formatted:
        formatted = formatted.rstrip("0").rstrip(".")
    return formatted


def _load_default_database() -> Tuple[List[Dict[str, Optional[float]]], str]:
    if not DEFAULT_DB_PATH.exists():
        raise FileNotFoundError(f"Nutrition database not found at {DEFAULT_DB_PATH}")

    records: List[Dict[str, Optional[float]]] = []
    with DEFAULT_DB_PATH.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            record = {
                "code": (row.get("code") or "").strip(),
                "food_name": (row.get("name") or "").strip(),
                "scientific_name": (row.get("scie") or "").strip(),
                "region": (row.get("regn") or "").strip(),
                "energy_kcal": _safe_float(row.get("enerc")),
                "protein_g": _safe_float(row.get("protcnt")),
                "fat_g": _safe_float(row.get("fatce")),
                "carbs_g": _safe_float(row.get("choavldf")),
                "fiber_g": _safe_float(row.get("fibtg")),
                "calcium_mg": _safe_float(row.get("ca")),
                "iron_mg": _safe_float(row.get("fe")),
            }
            records.append(record)

    database_text = _build_database_text(records)
    return records, database_text


def _build_database_text(records: List[Dict[str, Optional[float]]]) -> str:
    lines: List[str] = []
    for record in records[:DB_SAMPLE_LIMIT]:
        name = record.get("food_name") or "Unknown food"
        scientific = record.get("scientific_name")
        region = record.get("region")

        header = name
        if scientific:
            header = f"{header} ({scientific})"
        if region:
            header = f"{header} — Region code {region}"

        details = ["Per 100g edible portion"]
        for field, label, unit, decimals in DB_FIELDS:
            value_str = _format_number(record.get(field), decimals)
            if value_str is not None:
                details.append(f"{label}: {value_str} {unit}")

        lines.append(f"{header} | " + " | ".join(details))

    return "\n".join(lines)


def _store_database(records: List[Dict[str, Optional[float]]], text: str, name: str) -> None:
    st.session_state[_DB_RECORDS_KEY] = records
    st.session_state[_DB_TEXT_KEY] = text
    st.session_state[_DB_NAME_KEY] = name


def _get_database() -> Tuple[List[Dict[str, Optional[float]]], str, str]:
    records = st.session_state.get(_DB_RECORDS_KEY, [])
    text = st.session_state.get(_DB_TEXT_KEY, "")
    name = st.session_state.get(_DB_NAME_KEY, DEFAULT_DB_PATH.name)
    return records, text, name


def _ensure_database_loaded() -> bool:
    if _DB_RECORDS_KEY in st.session_state:
        return True

    try:
        records, text = _load_default_database()
    except FileNotFoundError as exc:
        st.error(str(exc))
        return False
    except Exception as exc:
        st.error(f"Unable to load nutrition database: {exc}")
        return False

    _store_database(records, text, DEFAULT_DB_PATH.name)
    return True


def _ensure_api_key() -> str:
    key = st.secrets.get("OPENAI_API_KEY") if hasattr(st, "secrets") else None
    key = key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise ValueError("OpenAI API key not configured. Set OPENAI_API_KEY via Streamlit secrets or environment.")
    return key


def _build_prompts(foods: List[Dict[str, str]], database_text: str) -> Tuple[str, str]:
    system_prompt = f"""You are a certified nutritionist specialising in Indian foods.
Use the database excerpt to match foods and calculate nutrient values per meal.
All nutrition numbers in the database represent 100g edible portions.

DATABASE EXCERPT (first {DB_SAMPLE_LIMIT} foods):
{database_text}

Return ONLY valid JSON matching this schema:
{{
  "parsed_foods": [
    {{
      "original_input": "string",
      "food_name": "standardised name",
      "matched_database_food": "exact match from database",
      "quantity": "parsed quantity string",
      "quantity_grams": number,
      "confidence": 0.0-1.0,
      "notes": "assumptions"
    }}
  ],
  "nutrition_breakdown": [
    {{
      "item": "original food",
      "matched_food": "database food",
      "quantity": "quantity summary",
      "quantity_grams": number,
      "calories": number,
      "protein_g": number,
      "carbs_g": number,
      "fat_g": number,
      "fiber_g": number or null,
      "calcium_mg": number or null,
      "iron_mg": number or null,
      "confidence": 0.0-1.0,
      "notes": "calculation notes"
    }}
  ],
  "total_nutrition": {{
    "total_calories": number,
    "total_protein_g": number,
    "total_carbs_g": number,
    "total_fat_g": number,
    "total_fiber_g": number or null,
    "total_calcium_mg": number or null,
    "total_iron_mg": number or null
  }},
  "warnings": ["important caveats or assumptions"]
}}
"""

    food_lines = []
    for idx, food in enumerate(foods, start=1):
        item = food.get("item", "").strip()
        quantity = food.get("quantity", "")
        if quantity:
            food_lines.append(f"{idx}. {item} – {quantity}")
        else:
            food_lines.append(f"{idx}. {item}")

    user_prompt = (
        "Analyze the following foods using the database excerpt. "
        "Estimate realistic edible weights when quantities are ambiguous, note assumptions, and respond with JSON only.\n\n"
        + "\n".join(food_lines)
    )

    return system_prompt, user_prompt


def _call_openai_for_nutrition(foods: List[Dict[str, str]], database_text: str) -> Dict:
    key = _ensure_api_key()
    client = openai.OpenAI(api_key=key, timeout=90.0)
    system_prompt, user_prompt = _build_prompts(foods, database_text)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        response_format={"type": "json_object"},
        temperature=0.2,
        max_tokens=4000,
    )

    if not response.choices:
        raise ValueError("No response from OpenAI.")

    content = response.choices[0].message.content.strip()
    if "```json" in content:
        content = content.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in content:
        content = content.split("```", 1)[1].split("```", 1)[0].strip()

    return json.loads(content)


def _coerce_number(value: Optional[object]) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _format_cell(value: Optional[object], decimals: int = 1) -> str:
    number = _coerce_number(value)
    if number is not None:
        formatted = f"{number:.{decimals}f}"
        if "." in formatted:
            formatted = formatted.rstrip("0").rstrip(".")
        return formatted
    if value in (None, "", [], {}):
        return "-"
    return str(value)


def _render_breakdown_table(rows: List[Dict[str, object]]) -> None:
    if not rows:
        st.info("No detailed breakdown available yet.")
        return

    table_rows: List[Dict[str, str]] = []
    for row in rows:
        confidence = _coerce_number(row.get("confidence"))
        table_rows.append(
            {
                "Input": _format_cell(row.get("item"), 3),
                "Matched Food": _format_cell(row.get("matched_food"), 3),
                "Quantity": _format_cell(row.get("quantity"), 3),
                "Grams": _format_cell(row.get("quantity_grams"), 1),
                "Calories": _format_cell(row.get("calories"), 1),
                "Protein (g)": _format_cell(row.get("protein_g"), 1),
                "Carbs (g)": _format_cell(row.get("carbs_g"), 1),
                "Fat (g)": _format_cell(row.get("fat_g"), 1),
                "Fiber (g)": _format_cell(row.get("fiber_g"), 1),
                "Calcium (mg)": _format_cell(row.get("calcium_mg"), 2),
                "Iron (mg)": _format_cell(row.get("iron_mg"), 2),
                "Confidence (%)": _format_cell(confidence * 100 if confidence is not None else None, 1),
                "Notes": _format_cell(row.get("notes"), 3),
            }
        )

    st.dataframe(
        table_rows,
        use_container_width=True,
        hide_index=True,
    )


def _display_analysis(result: Dict, elapsed: float) -> None:
    st.success(f"✅ Analysis complete in {elapsed:.1f} seconds!")

    meal_data = result.get("meal_nutrition", {})
    metadata = result.get("metadata", {})

    warnings = metadata.get("warnings") or result.get("warnings") or []
    if warnings:
        with st.expander("⚠️ Warnings & Notes", expanded=True):
            for warning in warnings:
                st.warning(warning)

    summary_tab, breakdown_tab = st.tabs(["📊 Summary", "🍲 Breakdown"])

    with summary_tab:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Calories", _format_cell(meal_data.get("total_calories"), decimals=0), help="Overall caloric content")
        col2.metric("Protein", f"{_format_cell(meal_data.get('total_protein_g'))} g", help="Total protein intake")
        col3.metric("Carbs", f"{_format_cell(meal_data.get('total_carbs_g'))} g", help="Total carbohydrate intake")
        col4.metric("Fat", f"{_format_cell(meal_data.get('total_fat_g'))} g", help="Total fat intake")

        extras: List[Tuple[str, Optional[object], int]] = [
            ("Fiber", meal_data.get("total_fiber_g"), 1),
            ("Calcium", meal_data.get("total_calcium_mg"), 0),
            ("Iron", meal_data.get("total_iron_mg"), 2),
        ]
        available_extras = [(label, value, decimals) for label, value, decimals in extras if value not in (None, "")]
        if available_extras:
            extra_columns = st.columns(len(available_extras))
            for column, (label, value, decimals) in zip(extra_columns, available_extras):
                suffix = " g" if "Fiber" in label else " mg"
                column.metric(label, f"{_format_cell(value, decimals=decimals)}{suffix}")

        meta_messages = []
        if metadata.get("database_name"):
            meta_messages.append(f"Source: `{metadata['database_name']}` (values per 100g).")
        if metadata.get("analyzed_at"):
            meta_messages.append(f"Generated at {metadata['analyzed_at']}.")
        meta_messages.append(f"Foods analysed: {metadata.get('database_size', 'N/A')} in database.")
        st.caption(" ".join(meta_messages))

    with breakdown_tab:
        breakdown = meal_data.get("breakdown") or []
        _render_breakdown_table(breakdown)

        parsed_foods = result.get("parsed_foods") or []
        if parsed_foods:
            with st.expander("AI parsing details", expanded=False):
                for parsed in parsed_foods:
                    original = parsed.get("original_input", "Unknown input")
                    matched = parsed.get("matched_database_food") or parsed.get("food_name", "Unknown match")
                    quantity = parsed.get("quantity") or f"{parsed.get('quantity_grams', 'n/a')} g"
                    confidence = parsed.get("confidence")
                    confidence_pct = f"{confidence * 100:.0f}%" if isinstance(confidence, (int, float)) else "n/a"
                    notes = parsed.get("notes", "")
                    st.markdown(
                        f"- **{original}** → {matched} ({quantity}, confidence {confidence_pct})"
                        + (f" — {notes}" if notes else "")
                    )

    st.download_button(
        "⬇️ Download Results (JSON)",
        data=json.dumps(result, indent=2),
        file_name="nutrition_analysis.json",
        mime="application/json",
        use_container_width=True,
    )


def _run_analysis(foods: List[Dict[str, str]], database_text: str) -> None:
    if not foods:
        st.warning("Add at least one food item before analyzing.")
        return

    with st.spinner("🤖 AI is analyzing your meal... This may take 5–15 seconds"):
        start = time.time()
        analysis = _call_openai_for_nutrition(foods, database_text)
        elapsed = time.time() - start

    records, _, db_name = _get_database()
    result = {
        "meal_nutrition": {
            **(analysis.get("total_nutrition") or {}),
            "breakdown": analysis.get("nutrition_breakdown", []),
        },
        "processing_time": elapsed,
        "metadata": {
            "database_size": len(records),
            "database_name": db_name,
            "model": "gpt-4o-mini",
            "warnings": analysis.get("warnings", []),
            "analyzed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "parsed_foods": analysis.get("parsed_foods", []),
        "warnings": analysis.get("warnings", []),
    }

    _display_analysis(result, elapsed)


def render_nutrition_ui() -> None:
    if not _ensure_database_loaded():
        return

    records, database_text, _ = _get_database()
    if not database_text:
        st.error("Nutrition database text representation is empty.")
        return

    st.session_state.setdefault(_FOODS_LIST_KEY, [])
    st.session_state.setdefault("nutrition_quick_input", "")
    st.session_state.setdefault("nutrition_quick_pick", [])

    foods_state: List[Dict[str, str]] = st.session_state[_FOODS_LIST_KEY]
    existing_items = {food["item"].lower() for food in foods_state}

    high_protein_candidates = sorted(
        (
            (record.get("food_name"), record.get("protein_g"))
            for record in records
            if record.get("food_name") and record.get("protein_g") not in (None, "")
        ),
        key=lambda item: (_safe_float(str(item[1])) or 0),
        reverse=True,
    )
    seen_names = set()
    common_food_options: List[str] = []
    for name, _protein in high_protein_candidates:
        if name not in seen_names:
            common_food_options.append(name)
            seen_names.add(name)
        if len(common_food_options) >= 20:
            break
    if not common_food_options:
        common_food_options = ["Paneer", "Chicken Breast", "Greek Yogurt", "Sprouted Moong Salad"]

    st.header("🍽️ AI Food Nutrition Analyzer")

    col_overview, col_index = st.columns([2, 1])
    with col_overview:
        st.markdown(f"Craft a meal snapshot; the assistant handles macros. **{len(foods_state)}** item(s) selected.")
    # with col_index:
    #     st.metric("Database size", len(records))

    def _set_status(level: str, message: str) -> None:
        st.session_state["nutrition_status"] = (level, message)

    def _add_food_callback():
        text = st.session_state.get("nutrition_quick_input", "")
        quick_picks = st.session_state.get("nutrition_quick_pick", [])
        items_to_add: List[str] = []
        if text and text.strip():
            items_to_add.append(text.strip())
        items_to_add.extend([item for item in quick_picks if item])

        if not items_to_add:
            _set_status("warning", "Add a description or select quick picks before submitting.")
            return

        added_any = False
        for item in dict.fromkeys(items_to_add):
            if item.lower() not in existing_items:
                foods_state.append({"item": item})
                existing_items.add(item.lower())
                added_any = True
            else:
                _set_status("info", f"'{item}' is already listed.")

        if added_any:
            _set_status("success", "Meal updated.")

    def _make_remove_callback(index: int):
        def _remove():
            if 0 <= index < len(foods_state):
                foods_state.pop(index)
                _set_status("info", "Removed an item from the meal.")
        return _remove

    def _clear_all_callback():
        st.session_state[_FOODS_LIST_KEY] = []
        _set_status("info", "Meal reset.")

    tab_quick, tab_bulk = st.tabs(["🍴 Quick Entry", "📝 Bulk Paste"])

    with tab_quick:
        status = st.session_state.get("nutrition_status")
        if status:
            level, message = status
            if level == "success":
                st.success(message)
            elif level == "warning":
                st.warning(message)
            elif level == "error":
                st.error(message)
            else:
                st.info(message)
            st.session_state["nutrition_status"] = None

        with st.form("quick_entry_form"):
            col_input, col_pick = st.columns([2, 1])
            col_input.text_input(
                "Describe a dish",
                placeholder="e.g., Paneer bhurji, 150 g",
                key="nutrition_quick_input",
            )
            col_pick.multiselect(
                "High-protein quick picks",
                options=common_food_options,
                key="nutrition_quick_pick",
                help="Selections submit with the button below",
            )
            submitted = st.form_submit_button("Add to meal", use_container_width=True)
            if submitted:
                _add_food_callback()

        if foods_state:
            st.markdown(f"### Current Meal ({len(foods_state)} items)")
            for idx, food in enumerate(foods_state):
                col_desc, col_remove = st.columns([6, 1])
                col_desc.markdown(f"**{idx + 1}.** {food['item']}")
                col_remove.button("Remove", key=f"remove_food_{idx}", on_click=_make_remove_callback(idx))

            col_analyze, col_clear = st.columns(2)
            if col_analyze.button("Analyze meal", type="primary", use_container_width=True, disabled=not foods_state):
                try:
                    _run_analysis(foods_state, database_text)
                except ValueError as exc:
                    st.error(str(exc))
                except Exception as exc:
                    st.error(f"Unable to analyze meal: {exc}")
            col_clear.button("Reset meal", use_container_width=True, on_click=_clear_all_callback, disabled=not foods_state)
        else:
            st.caption("No dishes yet. Add a description or choose quick picks to begin.")

    with tab_bulk:
        st.caption("Paste multiple lines from a tracker or menu to analyse instantly.")
        with st.form("bulk_entry_form"):
            bulk_input = st.text_area(
                "One food per line",
                placeholder="Chapathi, 4 pieces\nPotato kurma, 200 g\nIdli with sambar, 3 pieces\nDal tadka, 1 bowl",
                height=220,
                key="nutrition_bulk_input",
            )
            col_bulk_add, col_bulk_analyze = st.columns(2)
            add_bulk = col_bulk_add.form_submit_button("Add lines", use_container_width=True)
            analyze_bulk = col_bulk_analyze.form_submit_button("Analyze pasted lines", use_container_width=True, type="primary")

        def _parse_bulk(text: str) -> List[str]:
            return [line.strip() for line in (text or "").splitlines() if line.strip()]

        if add_bulk or analyze_bulk:
            parsed_lines = _parse_bulk(bulk_input)
            if not parsed_lines:
                st.warning("Please enter at least one food item before continuing.")
            else:
                if add_bulk:
                    for item in parsed_lines:
                        if item.lower() not in existing_items:
                            foods_state.append({"item": item})
                            existing_items.add(item.lower())
                    _set_status("success", "Bulk items added to the meal list.")
                if analyze_bulk:
                    try:
                        _run_analysis([{"item": item} for item in parsed_lines], database_text)
                    except ValueError as exc:
                        st.error(str(exc))
                    except Exception as exc:
                        st.error(f"Unable to analyze meal: {exc}")


if __name__ == "__main__":
    st.set_page_config(page_title="Nutrition Analyzer", page_icon="🍽️", layout="wide")
    render_nutrition_ui()
