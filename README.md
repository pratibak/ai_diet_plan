# FitFuel Wellness Hub

FitFuel is a Streamlit application that combines an AI diet planner, a meal-level nutrition analyzer, hydration tracking, and a wellness calendar into a single workspace. It is purpose-built for Indian cuisine and uses the IFCT 2017 nutrient composition data plus OpenAI models for tailored meal planning.

## Key Features

- 🧬 **Profile-driven planning** – Create a single personal profile (goals, dietary preferences, allergies) and reuse it across all tools.
- 🍱 **AI Diet Planner** – Generate validated 7-day Indian meal plans with macro targets, allergen checks, and actionable warnings.
- 📊 **Nutrition Analyzer** – Stage meals ad‑hoc and see macros/micros using the IFCT 2017 food database; ideal before or after eating.
- 💧 **Hydration Coach** – Compute recommended intake for the day and log actual consumption with a few clicks.
- 🗓️ **Wellness Calendar** – View plan adherence and hydration logs on an interactive calendar; achieved plan days glow green.
- 📈 **Dashboard overview** – Glance at today’s targets, queued meals, and hydration trends without digging into multiple tabs.

## Prerequisites

- Python 3.8+ (for local development)
- Docker & Docker Compose (optional)
- An OpenAI API key with access to GPT-4o-mini or similar model

## Installation

### Option 1: Docker (Recommended)

```bash
git clone <repository-url> diet
cd diet
echo "OPENAI_API_KEY=your-api-key-here" > .env
docker-compose up --build
```

Or using Docker directly:

```bash
docker build -t fitfuel .
docker run -p 8501:8501 --env-file .env fitfuel
```

### Option 2: Local Development

```bash
git clone <repository-url> diet
cd diet
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY="your-api-key-here"
streamlit run app.py
```

The app is available at http://localhost:8501.

## Application Tour

| Tab | Purpose |
| --- | --- |
| **Dashboard** | Shows today’s plan targets, staged meals, hydration trend, and quick status cards. |
| **Profile** | Single source of truth for personal details, goals, allergies, and preferences. |
| **Diet Planner** | Uses the saved profile to generate a 7-day plan, complete with macros, warnings, and meal breakdowns. |
| **Nutrition Analyzer** | Evaluate any meal using quick picks or bulk paste; compare against IFCT macros. |
| **Hydration Coach** | Calculates daily target (based on weight, activity, climate) and logs actual intake. |
| **Wellness Calendar** | Interactive month view showing plan adherence (green cells) and hydration logs. |

### Diet Planner Workflow

1. Save or update personal details in the **Profile** tab.
2. Switch to **Diet Planner** and click **Generate 7-Day Plan**.
3. Review BMR/TDEE, per-day macros, warnings, and meal breakdowns.
4. Generated plans automatically populate the dashboard and calendar.

### Nutrition Analyzer Workflow

1. Go to **Nutrition Analyzer**.
2. Use quick high-protein picks or paste multiple foods in bulk.
3. Run **Analyze meal** to view macros, breakdown table, and AI parsing notes.
4. Download results as JSON for further tracking if needed.

### Hydration & Calendar

- Log water intake in **Hydration Coach**; logs sync to the calendar.
- Mark diet-plan days as achieved inside the calendar; completed days display with a green background.

## Data & Integrations

- **Diet planning** – OpenAI GPT-4o-mini (or compatible) via the official SDK.
- **Food composition** – IFCT 2017 dataset embedded locally (`archive/ifct2017_compositions.csv`).
- **State management** – Streamlit session state retains profile, plan, hydration, and analyzer queues during a session.

## Project Structure

```
diet/
├── app.py                # Streamlit entry point and tab routing
├── diet_plan.py          # Diet planning logic, validation, and UI sections
├── food.py               # Nutrition analyzer logic and IFCT integrations
├── archive/              # IFCT datasets and derived references
├── docs/                 # Supplemental documentation (generated PDF flows)
├── requirements.txt      # Python dependencies
├── Dockerfile
├── docker-compose.yml
├── env.example
└── README.md
```

## Running Tests / Linting

The project is Streamlit-first; there are no automated tests yet. To validate changes manually:

```bash
streamlit run app.py   # ensure UI loads without errors
python3 -m compileall app.py diet_plan.py food.py   # quick syntax check
```

## Generating Documentation

This repository includes a generated PDF (`docs/fitfuel_logic.pdf`) summarizing the application logic, tab flow, and data sources. Regenerate it with:

```bash
python scripts/generate_flow_pdf.py  # if you create a helper script
```

## License

MIT License. See `LICENSE` (add one if required).

## Prerequisites

- Python 3.8 or higher (for local development)
- Docker and Docker Compose (for containerized deployment)
- OpenAI API key

## Installation

### Option 1: Docker (Recommended)

1. Clone the repository:
```bash
cd /Users/pratibakr/Documents/diet
```

2. Create a `.env` file with your OpenAI API key:
```bash
echo "OPENAI_API_KEY=your-api-key-here" > .env
```

3. Build and run with Docker Compose:
```bash
docker-compose up --build
```

Or using Docker directly:
```bash
docker build -t diet-plan-generator .
docker run -p 8501:8501 -e OPENAI_API_KEY=your-api-key-here diet-plan-generator
```

### Option 2: Local Development

1. Clone the repository:
```bash
cd /Users/pratibakr/Documents/diet
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your OpenAI API key:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

## Running the Application

### Using Docker
```bash
docker-compose up
```

### Using Docker directly
```bash
docker run -p 8501:8501 --env-file .env diet-plan-generator
```

### Local Development
Start the Streamlit app:

```bash
streamlit run app.py
```

The application will be available at:
- Web UI: http://localhost:8501

## How to Use

### Diet Plan Generator

1. **Fill in Personal Information**: Enter your name, age, and gender in the sidebar
2. **Set Physical Details**: Input your height and current weight
3. **Choose Your Goal**: Select Weight Loss, Muscle Gain, or Maintenance
4. **Set Target Weight** (if applicable): Enter your target weight for weight-based goals
5. **Select Activity Level**: Choose from Sedentary, Light, Moderate, Active, or Very Active
6. **Choose Preferences**: Select your regional preference and diet type
7. **Add Health Info**: Select any health conditions and enter allergies if any
8. **Generate Plan**: Click the "Generate Diet Plan" button
9. **View Results**: Browse your personalized 7-day meal plan with detailed nutritional information

### Nutrition Analyzer

1. Switch the sidebar toggle to **Nutrition Analyzer**.
2. The IFCT 2017 food composition database loads automatically; no upload required.
3. Search the database from the sidebar or review the sample foods list.
4. Add foods one by one or paste multiple lines in bulk.
5. Click **Analyze** to generate calorie and macro totals, micronutrients, warnings, and a downloadable JSON report.

## Example User Inputs

### Personal Information
- Name: Arjun
- Age: 30
- Gender: Male

### Physical Details
- Height: 175 cm
- Current Weight: 78 kg

### Goals
- Goal: Weight Loss
- Target Weight: 70 kg
- Activity Level: Moderate

### Preferences
- Region: South India
- Diet Type: Vegetarian

### Health & Allergies
- Health Conditions: Diabetes
- Allergies: Peanuts

## Supported Options

### Goal
- Weight Loss
- Muscle Gain
- Maintenance

### Gender
- Male
- Female
- Other

### Activity Level
- Sedentary
- Light
- Moderate
- Active
- Very Active

### Region
- South India
- North India
- West India
- East India

### Diet Type
- Vegetarian
- Non-Vegetarian
- Vegan
- Eggetarian

### Health Conditions
- Diabetes
- High BP
- PCOD
- Thyroid
- High Cholesterol

## Output Details

The generated diet plan includes:

1. **User Profile**: BMR, TDEE, target calories, and macro targets (protein, carbs, fat)
2. **7-Day Meal Plan**: Complete meal schedule with:
   - Breakfast, Morning Snack, Lunch, Evening Snack, Dinner
   - Detailed nutritional information for each meal
   - Individual food items with quantities and calories
3. **Warnings**: Health-related warnings and recommendations
4. **Notes**: General tips and considerations for the meal plan

## Project Structure

```
diet/
├── diet_plan.py         # Main Streamlit application
├── requirements.txt     # Python dependencies
├── Dockerfile          # Docker container configuration
├── docker-compose.yml  # Docker Compose configuration
├── .dockerignore       # Docker ignore file
├── env.example         # Environment variables example
└── README.md          # This file
```

## Docker Deployment

### Build the Docker Image
```bash
docker build -t diet-plan-generator .
```

### Run the Container
```bash
docker run -d -p 8501:8501 --name diet-plan --env-file .env diet-plan-generator
```

### Using Docker Compose
```bash
# Start the service
docker-compose up -d

# View logs
docker-compose logs -f

# Stop the service
docker-compose down

# Rebuild and restart
docker-compose up --build -d
```

Access the application at: http://localhost:8501

## Features Overview

### Nutrition Calculation
- **BMR**: Mifflin-St Jeor Equation
- **TDEE**: Based on activity level multipliers
- **Target Calories**: Deficit for weight loss, surplus for muscle gain
- **Macros**: Protein, Carbs, Fat distribution by goal

### Health Constraints
- Automatic food restrictions based on health conditions
- Preferred foods recommendations
- Special notes for each condition

### Regional Cuisine
- Authentic regional staples and dishes
- Breakfast, lunch, dinner items per region
- Cultural food preferences

### Validation
- Calorie target matching
- Macro distribution validation
- Allergen checking
- Meal completeness verification

## Error Handling

The application provides clear error messages for:
- Validation errors (e.g., target weight conflicts with goal)
- Missing required fields
- OpenAI API errors
- Plan validation failures

## Notes

- The application uses OpenAI's GPT-4o-mini model for cost-effectiveness
- Meal plans are generated fresh for each request
- Allergen checking is strict - allergens are never included
- Health warnings are provided for guidance, not medical advice
- Consult healthcare providers for medical conditions

## License

This project is for educational and personal use.

## Author

AI Diet Plan Generator - Personalized nutrition planning with Indian cuisine
# ai_diet_plan
