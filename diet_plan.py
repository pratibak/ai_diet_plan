import streamlit as st
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime, date, timedelta
import openai
import json
import os
from openai import OpenAI
import time
from enum import Enum

# ============================================================================
# ENUMS & CONSTANTS
# ============================================================================

class Gender(str, Enum):
    MALE = "Male"
    FEMALE = "Female"
    OTHER = "Other"

class Goal(str, Enum):
    WEIGHT_LOSS = "Weight Loss"
    MUSCLE_GAIN = "Muscle Gain"
    MAINTENANCE = "Maintenance"

class ActivityLevel(str, Enum):
    SEDENTARY = "Sedentary"
    LIGHT = "Light"
    MODERATE = "Moderate"
    ACTIVE = "Active"
    VERY_ACTIVE = "Very Active"

class Region(str, Enum):
    SOUTH_INDIA = "South India"
    NORTH_INDIA = "North India"
    WEST_INDIA = "West India"
    EAST_INDIA = "East India"

class DietType(str, Enum):
    VEGETARIAN = "Vegetarian"
    NON_VEGETARIAN = "Non-Vegetarian"
    VEGAN = "Vegan"
    EGGETARIAN = "Eggetarian"

# Activity multipliers for TDEE calculation
ACTIVITY_MULTIPLIERS = {
    ActivityLevel.SEDENTARY: 1.2,
    ActivityLevel.LIGHT: 1.375,
    ActivityLevel.MODERATE: 1.55,
    ActivityLevel.ACTIVE: 1.725,
    ActivityLevel.VERY_ACTIVE: 1.9
}

# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class DietPlanRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    age: int = Field(..., gt=10, lt=120, description="Age in years")
    gender: Gender
    goal: Goal
    height_cm: float = Field(..., gt=50, lt=300, description="Height in centimeters")
    current_weight_kg: float = Field(..., gt=20, lt=300, description="Current weight in kg")
    target_weight_kg: Optional[float] = Field(None, description="Target weight (required for weight goals)")
    activity_level: ActivityLevel = ActivityLevel.MODERATE
    health_conditions: List[str] = Field(default=[], description="e.g., ['Diabetes', 'High BP']")
    region: Region = Region.SOUTH_INDIA
    cuisine_preference: DietType = DietType.VEGETARIAN
    allergies: List[str] = Field(default=[], description="Food allergies")
    
    @validator('target_weight_kg')
    def validate_target_weight(cls, v, values):
        if 'goal' in values and values['goal'] != Goal.MAINTENANCE:
            if v is None:
                raise ValueError("Target weight is required for weight loss/gain goals")
            if 'current_weight_kg' in values:
                current = values['current_weight_kg']
                if values['goal'] == Goal.WEIGHT_LOSS and v >= current:
                    raise ValueError("Target weight must be less than current weight for weight loss")
                if values['goal'] == Goal.MUSCLE_GAIN and v <= current:
                    raise ValueError("Target weight must be greater than current weight for muscle gain")
        return v

class MealItem(BaseModel):
    food: str
    quantity: str
    calories: int

class Meal(BaseModel):
    items: List[MealItem]
    meal_total_calories: int
    meal_protein_g: float
    meal_carbs_g: float
    meal_fat_g: float

class DayPlan(BaseModel):
    day: int
    day_name: str
    total_calories: int
    total_protein_g: float
    total_carbs_g: float
    total_fat_g: float
    meals: Dict[str, Meal]

class UserProfile(BaseModel):
    name: str
    age: int
    gender: str
    bmr: float
    tdee: float
    target_calories: int
    target_protein_g: float
    target_carbs_g: float
    target_fat_g: float
    goal: str

class DietPlanResponse(BaseModel):
    user_profile: UserProfile
    seven_day_plan: List[DayPlan]
    notes: str
    warnings: List[str]
    generation_metadata: Dict[str, Any]

# ============================================================================
# NUTRITION CALCULATION FUNCTIONS
# ============================================================================

def calculate_bmr(weight_kg: float, height_cm: float, age: int, gender: Gender) -> float:
    """Calculate Basal Metabolic Rate using Mifflin-St Jeor Equation"""
    if gender == Gender.MALE:
        bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age + 5
    else:  # Female or Other
        bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age - 161
    return round(bmr, 2)

def calculate_tdee(bmr: float, activity_level: ActivityLevel) -> float:
    """Calculate Total Daily Energy Expenditure"""
    multiplier = ACTIVITY_MULTIPLIERS[activity_level]
    return round(bmr * multiplier, 2)

def calculate_target_calories(tdee: float, goal: Goal, current_weight: float, target_weight: float) -> int:
    """Calculate target daily calories based on goal"""
    if goal == Goal.WEIGHT_LOSS:
        # Safe deficit: 500 cal/day = ~0.5kg/week
        target = tdee - 500
        # Ensure minimum calories
        min_calories = 1200 if current_weight < 70 else 1500
        target = max(target, min_calories)
    elif goal == Goal.MUSCLE_GAIN:
        # Surplus: 300-500 cal/day
        target = tdee + 400
    else:  # Maintenance
        target = tdee
    
    return int(target)

def calculate_macros(target_calories: int, goal: Goal) -> Dict[str, float]:
    """Calculate macro targets (protein, carbs, fat in grams)"""
    if goal == Goal.WEIGHT_LOSS:
        protein_pct = 0.30
        carbs_pct = 0.40
        fat_pct = 0.30
    elif goal == Goal.MUSCLE_GAIN:
        protein_pct = 0.30
        carbs_pct = 0.45
        fat_pct = 0.25
    else:  # Maintenance
        protein_pct = 0.25
        carbs_pct = 0.45
        fat_pct = 0.30
    
    protein_g = round((target_calories * protein_pct) / 4, 1)  # 4 cal/g
    carbs_g = round((target_calories * carbs_pct) / 4, 1)      # 4 cal/g
    fat_g = round((target_calories * fat_pct) / 9, 1)          # 9 cal/g
    
    return {
        "protein_g": protein_g,
        "carbs_g": carbs_g,
        "fat_g": fat_g
    }

def detect_conflicts_and_warnings(request: DietPlanRequest) -> List[str]:
    """Detect potential conflicts in user requirements"""
    warnings = []
    
    # Check for conflicting diet + health conditions
    if request.cuisine_preference == DietType.VEGAN:
        if "Soy" in request.allergies or "Tofu" in request.allergies:
            warnings.append("Vegan diet with soy allergy may limit protein sources. Consider protein supplements.")
    
    # Check aggressive weight loss
    if request.goal == Goal.WEIGHT_LOSS and request.target_weight_kg:
        weight_diff = request.current_weight_kg - request.target_weight_kg
        if weight_diff > request.current_weight_kg * 0.15:  # More than 15% loss
            warnings.append("Target weight loss is aggressive. Recommend breaking it into smaller goals.")
    
    # Check multiple health conditions
    if len(request.health_conditions) > 2:
        warnings.append("Multiple health conditions detected. Please consult a healthcare provider before starting this plan.")
    
    # Check age-related warnings
    if request.age > 60:
        warnings.append("For seniors, ensure adequate calcium and vitamin D intake. Consider consulting a doctor.")
    
    if request.age < 18:
        warnings.append("Growing individuals need special nutrition. This plan should be reviewed by a pediatrician.")
    
    return warnings

# ============================================================================
# LLM PROMPT CONSTRUCTION
# ============================================================================

def build_system_prompt() -> str:
    """Build the system prompt for GPT"""
    return """You are an expert nutritionist specializing in Indian cuisine. Create personalized 7-day meal plans.

REQUIREMENTS:
- EXACTLY 7 days: breakfast, morning_snack, lunch, evening_snack, dinner
- Return ONLY valid JSON
- Accurate calories and macros
- Regional Indian foods with realistic portions
- Respect dietary restrictions and allergies

CALORIE GUIDELINES (per piece/serving):
- Chapati: 80 cal, Idli: 40 cal, Dosa: 120 cal
- Rice (1 cup cooked): 200 cal, Dal: 180 cal
- Chicken (100g): 165 cal, Fish: 120 cal, Egg: 70 cal, Paneer (100g): 265 cal

JSON FORMAT:
{
  "seven_day_plan": [
    {
      "day": 1,
      "day_name": "Monday",
      "total_calories": <sum>,
      "total_protein_g": <sum>,
      "total_carbs_g": <sum>,
      "total_fat_g": <sum>,
      "meals": {
        "breakfast": {
          "items": [{"food": "Idli", "quantity": "3 pieces", "calories": 120}],
          "meal_total_calories": 120,
          "meal_protein_g": 4.5,
          "meal_carbs_g": 24.0,
          "meal_fat_g": 0.5
        },
        "morning_snack": {...},
        "lunch": {...},
        "evening_snack": {...},
        "dinner": {...}
      }
    }
  ],
  "notes": "Brief tips"
}

Return ONLY the JSON object."""

def build_user_prompt(request: DietPlanRequest, nutrition_targets: Dict) -> str:
    """Build the user-specific prompt"""
    
    prompt = f"""Generate a 7-day meal plan:

USER: {request.name}, {request.age} years, {request.gender.value}, Goal: {request.goal.value}
Weight: {request.current_weight_kg} kg → {request.target_weight_kg} kg | Activity: {request.activity_level.value}

TARGETS: {nutrition_targets['target_calories']} cal/day | P: {nutrition_targets['protein_g']}g | C: {nutrition_targets['carbs_g']}g | F: {nutrition_targets['fat_g']}g

DISTRIBUTION: Breakfast 25% | Morning Snack 10% | Lunch 35% | Evening Snack 10% | Dinner 20%

REGION: {request.region.value} - Use authentic regional dishes
DIET: {request.cuisine_preference.value} - Follow dietary restrictions
"""

    if request.health_conditions:
        prompt += f"\nHEALTH: {', '.join(request.health_conditions)} - Adapt meals accordingly\n"
    
    if request.allergies:
        prompt += f"\nALLERGIES - NEVER INCLUDE: {', '.join(request.allergies)}\n"
    
    prompt += """
VARIETY: No repeated meals within 3 days. Mix traditional and modern options.
Return ONLY the JSON following the schema."""

    return prompt

def build_few_shot_examples() -> List[Dict]:
    """Provide few-shot examples for better LLM performance"""
    example_response = {
        "seven_day_plan": [
            {
                "day": 1,
                "day_name": "Monday",
                "total_calories": 1800,
                "total_protein_g": 95.0,
                "total_carbs_g": 220.0,
                "total_fat_g": 55.0,
                "meals": {
                    "breakfast": {
                        "items": [
                            {"food": "Idli", "quantity": "3 pieces", "calories": 120},
                            {"food": "Sambar", "quantity": "1 bowl", "calories": 100},
                            {"food": "Coconut chutney", "quantity": "2 tbsp", "calories": 50}
                        ],
                        "meal_total_calories": 270,
                        "meal_protein_g": 8.0,
                        "meal_carbs_g": 48.0,
                        "meal_fat_g": 4.0
                    },
                    "morning_snack": {
                        "items": [
                            {"food": "Banana", "quantity": "1 medium", "calories": 105},
                            {"food": "Almonds", "quantity": "10 pieces", "calories": 70}
                        ],
                        "meal_total_calories": 175,
                        "meal_protein_g": 4.0,
                        "meal_carbs_g": 25.0,
                        "meal_fat_g": 6.0
                    },
                    "lunch": {
                        "items": [
                            {"food": "Brown rice", "quantity": "1 cup", "calories": 215},
                            {"food": "Mixed vegetable sambar", "quantity": "1 bowl", "calories": 120},
                            {"food": "Beetroot poriyal", "quantity": "1 cup", "calories": 80},
                            {"food": "Curd", "quantity": "1 small bowl", "calories": 60}
                        ],
                        "meal_total_calories": 475,
                        "meal_protein_g": 18.0,
                        "meal_carbs_g": 85.0,
                        "meal_fat_g": 8.0
                    },
                    "evening_snack": {
                        "items": [
                            {"food": "Sundal", "quantity": "1 cup", "calories": 150},
                            {"food": "Green tea", "quantity": "1 cup", "calories": 2}
                        ],
                        "meal_total_calories": 152,
                        "meal_protein_g": 8.0,
                        "meal_carbs_g": 28.0,
                        "meal_fat_g": 2.0
                    },
                    "dinner": {
                        "items": [
                            {"food": "Chapati", "quantity": "2 pieces", "calories": 160},
                            {"food": "Palak paneer", "quantity": "1 cup", "calories": 250},
                            {"food": "Cucumber salad", "quantity": "1 bowl", "calories": 30}
                        ],
                        "meal_total_calories": 440,
                        "meal_protein_g": 22.0,
                        "meal_carbs_g": 45.0,
                        "meal_fat_g": 18.0
                    }
                }
            }
        ],
        "notes": "This meal plan provides balanced nutrition with emphasis on whole grains, lean proteins, and plenty of vegetables."
    }
    
    return [
        {
            "role": "user",
            "content": "Generate a 1-day example meal plan for a vegetarian person needing 1800 calories."
        },
        {
            "role": "assistant",
            "content": json.dumps(example_response)
        }
    ]

# ============================================================================
# OPENAI INTEGRATION
# ============================================================================


def generate_diet_plan_with_llm(request: DietPlanRequest, nutrition_targets: Dict) -> Dict:
    """Call OpenAI API to generate diet plan"""
    
    total_start = time.time()
    print(f"[DEBUG] Starting diet plan generation at {time.strftime('%H:%M:%S')}")
    
    # Initialize OpenAI client
    # Try to get API key from Streamlit secrets first, then from environment variable
    print(f"[DEBUG] Getting API key...")
    api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        raise ValueError("OpenAI API key not configured. Please set OPENAI_API_KEY in Streamlit secrets or environment variable.")
    
    print(f"[DEBUG] API key retrieved: {'Yes' if api_key else 'No'}")
    
    # Set the API key for OpenAI client - STREAMLIT CLOUD COMPATIBLE VERSION
    print(f"[DEBUG] Creating OpenAI client...")
    
    # Clear any proxy-related environment variables that might interfere
    proxy_vars = ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy', 'ALL_PROXY', 'all_proxy']
    original_proxy_values = {}
    for var in proxy_vars:
        if var in os.environ:
            original_proxy_values[var] = os.environ[var]
            del os.environ[var]
    
    try:
        # Create client with minimal parameters
        client = OpenAI(api_key=api_key)
        print(f"[DEBUG] OpenAI client created successfully")
    finally:
        # Restore proxy settings if they existed
        for var, value in original_proxy_values.items():
            os.environ[var] = value

# Build messages
    print(f"[DEBUG] Building messages...")
    
    # Build messages
    print(f"[DEBUG] Building messages...")
    prompt_start = time.time()
    print(f"[DEBUG] Building system prompt...")
    system_prompt = build_system_prompt()
    print(f"[DEBUG] System prompt built: {len(system_prompt)} chars")
    
    messages = [
        {"role": "system", "content": system_prompt}
    ]
    
    # Add few-shot examples
    # messages.extend(build_few_shot_examples())  # Uncomment for better results
    
    # Add user request
    print(f"[DEBUG] Building user prompt...")
    user_prompt = build_user_prompt(request, nutrition_targets)
    print(f"[DEBUG] User prompt built: {len(user_prompt)} chars")
    
    messages.append({
        "role": "user",
        "content": user_prompt
    })
    
    print(f"[DEBUG] Messages list contains {len(messages)} messages")
    prompt_time = time.time() - prompt_start
    print(f"[DEBUG] Prompt building took: {prompt_time:.2f}s")
    
    # Call OpenAI API with retries
    max_retries = 1  # Reduce retries for faster failure detection
    content = None
    for attempt in range(max_retries):
        try:
            print(f"[DEBUG] Attempt {attempt + 1}/{max_retries}")
            api_start = time.time()
            print(f"[DEBUG] About to call OpenAI API...")
            print(f"[DEBUG] Model: gpt-4o-mini")
            print(f"[DEBUG] Number of messages: {len(messages)}")
            print(f"[DEBUG] Temperature: 0.1")  # Lower for consistency
            print(f"[DEBUG] Max tokens: 2500")  # Reduced for faster generation
            print(f"[DEBUG] Calling client.chat.completions.create() now...")
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",  # Fast and cost-effective
                messages=messages,
                response_format={"type": "json_object"},  # Force JSON output
                temperature=0.1,  # Lower for faster, more consistent output
                max_tokens=10000,  # Reduced for faster generation while still complete
                top_p=0.95  # Faster than default
            )
            
            print(f"[DEBUG] API call returned!")
            api_time = time.time() - api_start
            print(f"[DEBUG] API call took: {api_time:.2f}s")
            
            print(f"[DEBUG] Checking response structure...")
            print(f"[DEBUG] Response type: {type(response)}")
            print(f"[DEBUG] Response has choices: {hasattr(response, 'choices')}")
            
            if hasattr(response, 'choices') and response.choices:
                print(f"[DEBUG] Number of choices: {len(response.choices)}")
                print(f"[DEBUG] Response tokens: {response.usage.total_tokens} (prompt: {response.usage.prompt_tokens}, completion: {response.usage.completion_tokens})")
                
                # Check if response was complete
                finish_reason = response.choices[0].finish_reason
                print(f"[DEBUG] Finish reason: {finish_reason}")
                if finish_reason != "stop":
                    raise ValueError(f"Response incomplete. Finish reason: {finish_reason}")
                
                # Parse response
                parse_start = time.time()
                print(f"[DEBUG] Extracting content from response...")
                content = response.choices[0].message.content
                print(f"[DEBUG] Raw content length: {len(content) if content else 0} characters")
                
                # Strip content
                content = content.strip() if content else ""
                print(f"[DEBUG] Response content length after strip: {len(content)} characters")
                
                # Try to extract JSON from markdown code blocks if present
                if "```json" in content:
                    print(f"[DEBUG] Found markdown code block with json")
                    content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    print(f"[DEBUG] Found markdown code block")
                    content = content.split("```")[1].split("```")[0].strip()
                
                # Remove any leading/trailing whitespace that might cause issues
                content = content.strip()
                print(f"[DEBUG] Final content length: {len(content)} characters")
                print(f"[DEBUG] Content preview: {content[:100]}...")
                
                # Additional fallback: Try to fix common JSON issues
                # (Though with response_format="json_object", this shouldn't be needed)
                print(f"[DEBUG] Attempting JSON parse...")
                plan_data = json.loads(content)
                print(f"[DEBUG] JSON parse successful!")
                
                parse_time = time.time() - parse_start
                print(f"[DEBUG] JSON parsing took: {parse_time:.2f}s")
                
                # Validate structure
                print(f"[DEBUG] Validating plan structure...")
                if "seven_day_plan" not in plan_data:
                    raise ValueError("Response missing 'seven_day_plan' key")
                if len(plan_data["seven_day_plan"]) != 7:
                    raise ValueError(f"Expected 7 days, got {len(plan_data['seven_day_plan'])}")
                return plan_data
            else:
                raise ValueError("No choices in response")
            
        except json.JSONDecodeError as e:
            if attempt == max_retries - 1:
                # Log the problematic content for debugging
                error_msg = f"Failed to parse LLM response: {str(e)}"
                if content:
                    # Show first 500 chars of content for debugging
                    if len(content) > 500:
                        error_msg += f"\nContent preview: {content[:500]}..."
                    else:
                        error_msg += f"\nContent: {content}"
                raise ValueError(error_msg)
            continue
            
        except openai.APIError as e:
            if attempt == max_retries - 1:
                raise ValueError(f"OpenAI API error: {str(e)}")
            continue
        
        except Exception as e:
            if attempt == max_retries - 1:
                raise ValueError(f"Unexpected error: {str(e)}")
            continue
    
    raise ValueError("Failed to generate diet plan after retries")

# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

def validate_diet_plan(plan_data: Dict, nutrition_targets: Dict, request: DietPlanRequest) -> tuple[bool, List[str]]:
    """Validate the generated diet plan"""
    errors = []
    warnings = []
    
    seven_day_plan = plan_data.get("seven_day_plan", [])
    
    # Check number of days
    if len(seven_day_plan) != 7:
        errors.append(f"Expected 7 days, got {len(seven_day_plan)}")
        return False, errors
    
    # Validate each day
    for day_plan in seven_day_plan:
        day_num = day_plan.get("day", "unknown")
        
        # Check required meals
        meals = day_plan.get("meals", {})
        required_meals = ["breakfast", "morning_snack", "lunch", "evening_snack", "dinner"]
        for meal_name in required_meals:
            if meal_name not in meals:
                errors.append(f"Day {day_num}: Missing {meal_name}")
        
        # Check calorie totals
        total_calories = day_plan.get("total_calories", 0)
        target_calories = nutrition_targets["target_calories"]
        
        tolerance = 150  # Allow 150 cal variance
        if abs(total_calories - target_calories) > tolerance:
            warnings.append(f"Day {day_num}: Calories {total_calories} vs target {target_calories}")
        
        # Check for allergens
        for meal_name, meal_data in meals.items():
            items = meal_data.get("items", [])
            for item in items:
                food_name = item.get("food", "").lower()
                for allergen in request.allergies:
                    if allergen.lower() in food_name:
                        errors.append(f"Day {day_num}, {meal_name}: Contains allergen '{allergen}' in '{item.get('food')}'")
    
    # If critical errors, fail validation
    if errors:
        return False, errors
    
    # Warnings are okay, just informational
    return True, warnings

# ============================================================================
# STREAMLIT UI
# ============================================================================

def render_diet_plan_ui(show_title: bool = True):
    if show_title:
        st.title("🥗 AI Diet Plan Generator")
        st.markdown("Fill out your profile once, then update goals anytime for a refreshed 7-day meal plan.")
    else:
        st.header("🥗 Personalized Diet Plan")
        st.caption("Update your fitness profile and generate a fresh 7-day plan in seconds.")

    profile = st.session_state.get("diet_profile")
    if not profile:
        st.info("Set up your personal profile in the Profile tab before generating a plan.")
        return

    st.caption("Profile is locked in. Adjust it via the Profile tab if anything changes before generating a new plan.")

    generate_plan = st.button("Generate 7-Day Plan", type="primary", use_container_width=True)

    # Main content area
    if generate_plan:
        name = profile.get("name", "").strip()
        if not name:
            st.error("Please add your name in the Profile tab before generating a plan.")
            return

        age = int(profile.get("age", 30))
        gender = profile.get("gender", Gender.MALE.value)
        height_cm = float(profile.get("height_cm", 175))
        current_weight_kg = float(profile.get("current_weight_kg", 70.0))
        goal = profile.get("goal", Goal.WEIGHT_LOSS.value)
        target_weight_kg = profile.get("target_weight_kg") if goal != Goal.MAINTENANCE.value else None
        activity_level = profile.get("activity_level", ActivityLevel.MODERATE.value)
        region = profile.get("region", Region.SOUTH_INDIA.value)
        cuisine_preference = profile.get("cuisine_preference", DietType.VEGETARIAN.value)
        health_conditions = profile.get("health_conditions", [])
        allergies = profile.get("allergies", [])
        
        # Validate target weight
        try:
            request = DietPlanRequest(
                name=name,
                age=age,
                gender=Gender(gender),
                goal=Goal(goal),
                height_cm=height_cm,
                current_weight_kg=current_weight_kg,
                target_weight_kg=target_weight_kg,
                activity_level=ActivityLevel(activity_level),
                health_conditions=health_conditions,
                region=Region(region),
                cuisine_preference=DietType(cuisine_preference),
                allergies=allergies
            )
        except ValueError as e:
            st.error(f"Validation Error: {str(e)}")
            return
        
        # Calculate nutrition targets
        with st.spinner("Calculating your nutrition targets..."):
            bmr = calculate_bmr(current_weight_kg, height_cm, age, Gender(gender))
            tdee = calculate_tdee(bmr, ActivityLevel(activity_level))
            target_calories = calculate_target_calories(
                tdee, Goal(goal), current_weight_kg, 
                target_weight_kg if target_weight_kg else current_weight_kg
            )
            macros = calculate_macros(target_calories, Goal(goal))
            
            nutrition_targets = {
                "target_calories": target_calories,
                "protein_g": macros["protein_g"],
                "carbs_g": macros["carbs_g"],
                "fat_g": macros["fat_g"]
            }
        
        # Detect warnings
        warnings = detect_conflicts_and_warnings(request)
        
        # Generate plan
        with st.spinner("🤖 AI is creating your personalized 7-day meal plan... This may take 30-60 seconds"):
            gen_start = time.time()
            try:
                plan_data = generate_diet_plan_with_llm(request, nutrition_targets)
                gen_time = time.time() - gen_start
                
                # Validate
                validation_passed, validation_warnings = validate_diet_plan(plan_data, nutrition_targets, request)
                
                if not validation_passed:
                    st.error(f"Validation failed: {', '.join(validation_warnings)}")
                    return
                
                warnings.extend(validation_warnings)
                
                plan_start_date = date.today()
                plan_schedule = {}
                for idx, day_info in enumerate(plan_data.get("seven_day_plan", [])):
                    plan_date = plan_start_date + timedelta(days=idx)
                    plan_schedule[plan_date.isoformat()] = day_info
                st.session_state["diet_plan_result"] = {
                    "generated_at": plan_start_date.isoformat(),
                    "schedule": plan_schedule,
                    "plan": plan_data,
                    "targets": nutrition_targets,
                }
                st.session_state["plan_adherence"] = {}

                # Display warnings
                if warnings:
                    with st.expander("⚠️ Warnings", expanded=True):
                        for warning in warnings:
                            st.warning(warning)
                
                # Display user profile
                st.header("Your Nutrition Profile")
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("BMR", f"{bmr:.0f} cal")
                with col2:
                    st.metric("TDEE", f"{tdee:.0f} cal")
                with col3:
                    st.metric("Target Calories", f"{target_calories} cal")
                with col4:
                    st.metric("Protein", f"{macros['protein_g']:.1f}g")
                with col5:
                    st.metric("Carbs", f"{macros['carbs_g']:.1f}g")
                
                # Display diet plan
                st.header("7-Day Meal Plan")
                
                for day_data in plan_data["seven_day_plan"]:
                    with st.expander(f"Day {day_data['day']}: {day_data['day_name']}", expanded=(day_data['day'] == 1)):
                        # Daily summary
                        st.markdown(f"**Daily Total:** {day_data['total_calories']} cal | "
                                  f"Protein: {day_data['total_protein_g']:.1f}g | "
                                  f"Carbs: {day_data['total_carbs_g']:.1f}g | "
                                  f"Fat: {day_data['total_fat_g']:.1f}g")
                        
                        # Meals
                        for meal_name, meal_data in day_data["meals"].items():
                            st.markdown(f"### {meal_name.replace('_', ' ').title()}")
                            st.markdown(f"*{meal_data['meal_total_calories']} cal | "
                                      f"P: {meal_data['meal_protein_g']:.1f}g | "
                                      f"C: {meal_data['meal_carbs_g']:.1f}g | "
                                      f"F: {meal_data['meal_fat_g']:.1f}g*")
                            
                            for item in meal_data["items"]:
                                st.markdown(f"- **{item['food']}** ({item['quantity']}) - {item['calories']} cal")
                        
                        st.divider()
                
                # Notes
                if plan_data.get("notes"):
                    st.info(f"💡 {plan_data['notes']}")
                
            except Exception as e:
                gen_time = time.time() - gen_start if 'gen_start' in locals() else 0
                st.error(f"Error generating diet plan: {str(e)}")
                if gen_time > 0:
                    st.info(f"⏱️ Time before error: {gen_time:.1f} seconds")
    
def main():
    st.set_page_config(
        page_title="AI Diet Plan Generator",
        page_icon="🥗",
        layout="wide"
    )
    render_diet_plan_ui(show_title=True)


if __name__ == "__main__":
    main()
