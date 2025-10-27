# AI Diet Plan Generator

A Streamlit-based AI diet plan generator that creates personalized 7-day meal plans based on Indian cuisine preferences, health conditions, and dietary restrictions.

## Features

- 🎯 **Personalized Nutrition**: Calculate BMR, TDEE, and macro targets based on age, gender, weight, height, and activity level
- 🍛 **Regional Indian Cuisine**: Support for South, North, West, and East Indian regional preferences
- 🥗 **Dietary Preferences**: Vegetarian, Non-Vegetarian, Vegan, and Eggetarian options
- 🏥 **Health Condition Support**: Special considerations for Diabetes, High BP, PCOD, Thyroid, High Cholesterol
- 🤖 **AI-Powered**: Uses OpenAI GPT-4 to generate diverse and authentic meal plans
- ⚠️ **Smart Warnings**: Detects potential conflicts and provides health warnings
- 🚫 **Allergy Safety**: Strict allergen checking in meal plans
- 🎨 **User-Friendly Interface**: Interactive Streamlit UI with sidebar inputs and detailed meal plan display

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

1. **Fill in Personal Information**: Enter your name, age, and gender in the sidebar
2. **Set Physical Details**: Input your height and current weight
3. **Choose Your Goal**: Select Weight Loss, Muscle Gain, or Maintenance
4. **Set Target Weight** (if applicable): Enter your target weight for weight-based goals
5. **Select Activity Level**: Choose from Sedentary, Light, Moderate, Active, or Very Active
6. **Choose Preferences**: Select your regional preference and diet type
7. **Add Health Info**: Select any health conditions and enter allergies if any
8. **Generate Plan**: Click the "Generate Diet Plan" button
9. **View Results**: Browse your personalized 7-day meal plan with detailed nutritional information

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
