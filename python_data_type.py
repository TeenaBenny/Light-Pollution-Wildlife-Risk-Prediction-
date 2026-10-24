# Light Pollution Wildlife Risk Prediction
# Step 1: Python Data Types
# Small Dummy Dataset

# ------------------------------------------------
# 1. LIST
# ------------------------------------------------

light_pollution = [20, 40, 70, 90, 60]

wildlife_count = [50, 45, 30, 15, 35]

night_activity = [80, 65, 40, 20, 50]

print("Light Pollution:", light_pollution)
print("Wildlife Count:", wildlife_count)
print("Night Activity:", night_activity)

print()


# ------------------------------------------------
# 2. TUPLE
# ------------------------------------------------

locations = (
    "Forest A",
    "Forest B",
    "Forest C",
    "Forest D",
    "Forest E"
)

print("Locations:", locations)
print("First Location:", locations[0])

print()


# ------------------------------------------------
# 3. DICTIONARY
# ------------------------------------------------

forest = {
    "location": "Forest A",
    "light_pollution": 20,
    "wildlife_count": 50,
    "night_activity": 80,
    "risk": "Low"
}

print("Forest Data:")
print(forest)

print("Location:", forest["location"])
print("Light Pollution:", forest["light_pollution"])
print("Wildlife Count:", forest["wildlife_count"])
print("Night Activity:", forest["night_activity"])
print("Risk:", forest["risk"])

print()


# ------------------------------------------------
# 4. LIST OF DICTIONARIES
# ------------------------------------------------

data = [
    {
        "location": "Forest A",
        "light_pollution": 20,
        "wildlife_count": 50,
        "night_activity": 80,
        "risk": "Low"
    },

    {
        "location": "Forest B",
        "light_pollution": 40,
        "wildlife_count": 45,
        "night_activity": 65,
        "risk": "Low"
    },

    {
        "location": "Forest C",
        "light_pollution": 70,
        "wildlife_count": 30,
        "night_activity": 40,
        "risk": "High"
    },

    {
        "location": "Forest D",
        "light_pollution": 90,
        "wildlife_count": 15,
        "night_activity": 20,
        "risk": "High"
    },

    {
        "location": "Forest E",
        "light_pollution": 60,
        "wildlife_count": 35,
        "night_activity": 50,
        "risk": "Medium"
    }
]


# ------------------------------------------------
# 5. PRINT ALL DATA
# ------------------------------------------------

print("Wildlife Dataset:")

for forest in data:
    print(forest)

print()


# ------------------------------------------------
# 6. PRINT ONLY LOCATION AND RISK
# ------------------------------------------------

print("Location and Risk:")

for forest in data:
    print(
        forest["location"],
        "->",
        forest["risk"]
    )

print()


# ------------------------------------------------
# 7. PRINT LIGHT POLLUTION
# ------------------------------------------------

print("Light Pollution Values:")

for forest in data:
    print(
        forest["location"],
        "->",
        forest["light_pollution"]
    )

print()


# ------------------------------------------------
# 8. SIMPLE CALCULATION
# ------------------------------------------------

light_values = [20, 40, 70, 90, 60]

total = sum(light_values)

average = total / len(light_values)

print("Total Light Pollution:", total)
print("Average Light Pollution:", average)