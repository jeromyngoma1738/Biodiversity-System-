import pandas as pd
import numpy as np
import joblib

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

from models import Observation


# ============================================================
# CBU NATURE PARK ECOSYSTEM DEFINITION
# TROPHIC ROLES
# ============================================================

TROPHIC_ROLE = {

    # =========================
    # PRODUCERS
    # =========================
    "Tree": "producer",
    "Grass": "producer",
    "Flowering Plant": "producer",
    "Acacia": "producer",
    "Miombo Tree": "producer",
    "Mopane": "producer",
    "Grass Species": "producer",
    "Plant": "producer",
    "Brachystegia": "producer",
    "Julbernardia": "producer",
    "Isoberlinia": "producer",
    "Faidherbia Albida": "producer",
    "Khasi Pine": "producer",
    "Apple-ring Acacia": "producer",
    "Ana Tree": "producer",
    "wild loquate": "producer",
    "Wild custard apple": "producer",
    "Msasa": "producer",

    # =========================
    # HERBIVORES
    # =========================
    "Grasshopper": "herbivore",
    "Caterpillar": "herbivore",
    "Snail": "herbivore",
    "Rabbit": "herbivore",

    # Antelopes
    "Impala": "herbivore",
    "Waterbuck": "herbivore",
    "Puku": "herbivore",
    "Kudu": "herbivore",
    "Sable Antelope": "herbivore",
    "Bushbuck": "herbivore",
    "Duiker": "herbivore",

    # Other herbivores
    "Warthog": "herbivore",
    "Zebra": "herbivore",
    "Buffalo": "herbivore",
    "waterbucks":"herbivore",

    # =========================
    # OMNIVORES
    # =========================
    "Mouse": "omnivore",
    "Mice": "omnivore",
    "Rat": "omnivore",
    "Rats": "omnivore",
    "Rodent": "omnivore",
    "Crow": "omnivore",
    "Dove": "omnivore",
    "pigeon": "omnivore",
    "African Grey Hornbill": "omnivore",
    "Southern Yellow-billed Hornbill": "omnivore",
    "Crested Barbet": "omnivore",
    "Civet": "omnivore",

    # =========================
    # SMALL PREDATORS / INSECTIVORES
    # =========================
    "Spider": "small_predator",
    "Praying Mantis": "small_predator",
    "Frog": "small_predator",
    "Frogs": "small_predator",
    "Lizard": "small_predator",

    # Insect-eating birds
    "Insectivorous Bird": "small_predator",
    "African Hoopoe": "small_predator",
    "Fork-tailed Drongo": "small_predator",
    "Common Fiscal": "small_predator",
    "African Paradise Flycatcher": "small_predator",

    # PREDATORS
    "Snake": "predator",
    "Snakes": "predator",
    "Mongoose": "predator",
    "Cat": "predator",
    
    
    # HERBIVOROUS / SEED-EATING BIRDS
 
    "Helmeted Guineafowl": "herbivore",
    "Cape Turtle Dove": "herbivore",
    "Laughing Dove": "herbivore",
    "Dove": "herbivore",
    "Village Weaver": "herbivore",
    "Southern Red Bishop": "herbivore",
    "Speckled Mousebird": "herbivore",

    # LARGE PREDATORS
    "Monitor Lizard": "large_predator",
    "Eagle": "large_predator",
    "Crocodile": "large_predator",

    # DECOMPOSERS
    "Fungi": "decomposer",
    "Bacteria": "decomposer",
    "Termite": "decomposer",
    "Earthworm": "decomposer",
}

FOOD_WEB = {

    "producer": {
        "feeds": ["herbivore", "omnivore"],
        "fed_by": ["decomposer"]
    },

    "herbivore": {
        "feeds": [""],
        "fed_by": ["producer"]
    },

    "small_predator": {
        "feeds": ["predator", "large_predator"],
        "fed_by": [ "omnivore"]
    },

    "predator": {
        "feeds": ["large_predator"],
        "fed_by": [ "omnivore"]
    },

    "omnivore": {
        "feeds": ["producer","small_predator","predator" ],
        "fed_by": ["predator", "large_predator"]
    },

    "large_predator": {
        "feeds": [],
        "fed_by": [
            "herbivore",
            "small_predator",
            "predator",
            "omnivore"
        ]
    },

    "decomposer": {
        "feeds": ["producer"],
        "fed_by": [""]
    }
}


# ============================================================
# SPECIES-SPECIFIC DIET
#
# FIX: this dict previously covered fewer than a quarter of the
# species defined in TROPHIC_ROLE, and several entries pointed
# at generic, non-species placeholders ("Bird", "Insects",
# "Fruit", "Seeds", "Leaves", "Eggs", "Small Animals") that never
# match an actual observed species. Every entry below now points
# only at real species that also appear in TROPHIC_ROLE, and
# every herbivore/omnivore/small_predator/predator/large_predator
# species has an entry.
# ============================================================

SPECIES_DIET = {

    # -------------------------
    # Small predators
    # -------------------------
    "Frog": ["Grasshopper", "Caterpillar", "Spider", "Snail"],
    "Frogs": ["Grasshopper", "Caterpillar", "Spider", "Snail"],
    "Lizard": ["Grasshopper", "Caterpillar", "Spider"],
    "Spider": ["Grasshopper", "Caterpillar"],
    "Praying Mantis": ["Grasshopper", "Caterpillar"],
    "Insectivorous Bird": ["Grasshopper", "Caterpillar", "Spider"],
    "African Hoopoe": ["Grasshopper", "Caterpillar"],
    "Fork-tailed Drongo": ["Grasshopper", "Caterpillar", "Spider"],
    "Common Fiscal": ["Grasshopper", "Caterpillar", "Lizard"],
    "African Paradise Flycatcher": ["Grasshopper", "Caterpillar"],

    # -------------------------
    # Predators
    # -------------------------
    "Snake": ["Frog", "Mouse", "Rat","Mice" ,"Lizard", "Helmeted Guineafowl"],
    "Snakes": ["Frog", "Mouse", "Rat", "Mice","Lizard", "Helmeted Guineafowl"],
    "Mongoose": ["Snake", "Mouse", "Rat", "Frog", "Grasshopper"],

    # -------------------------
    # Large predators
    # -------------------------
    "Monitor Lizard": [
        "Frog", "Snake", "Lizard", "Mouse",
        "Rat", "Helmeted Guineafowl", "Grasshopper"
    ],
    "Eagle": ["Snake", "Mouse", "Rat", "Lizard", "Dove"],
    "Crocodile": ["Impala", "Waterbuck", "Puku"],

    # -------------------------
    # Herbivores
    # -------------------------
    "Impala": ["Grass", "Grass Species", "Miombo Tree", "Acacia"],
    "Zebra": ["Grass", "Grass Species"],
    "Puku": ["Grass", "Plant"],
    "Kudu": ["Acacia", "Miombo Tree", "Brachystegia"],
    "Waterbuck": ["Grass", "Grass Species"],
    "Buffalo": ["Grass", "Grass Species"],
    "Sable Antelope": ["Grass", "Grass Species", "Miombo Tree"],
    "Bushbuck": ["Acacia", "Miombo Tree"],
    "Duiker": ["Grass", "Grass Species"],
    "Warthog": ["Grass", "Grass Species", "Plant"],
    "Rabbit": ["Grass", "Grass Species"],
    "Snail": ["Grass", "Flowering Plant"],
    "Grasshopper": ["Grass", "Grass Species"],
    "Caterpillar": ["Flowering Plant", "Grass"],
    "Helmeted Guineafowl": ["Grass Species"],
    "Cape Turtle Dove": ["Grass Species"],
    "Laughing Dove": ["Grass Species"],
    "Dove": ["Grass Species"],
    "Village Weaver": ["Grass Species"],
    "Southern Red Bishop": ["Grass Species"],
    "Speckled Mousebird": ["Flowering Plant"],

    # -------------------------
    # Omnivores
    # -------------------------
    "Mouse": ["Grass", "Grass Species", "Grasshopper"],
    "Rat": ["Grass", "Grass Species", "Grasshopper"],
    "Mice" :["Grass", "Grass Species", "Grasshopper"],
    
    "Rats": ["Grass", "Grass Species", "Grasshopper"],
    "Rodent": ["Grass", "Grasshopper"],
    "African Pied Crow": ["Grasshopper", "Flowering Plant"],
    "Hadada Ibis": ["Earthworm", "Grasshopper"],
    "African Grey Hornbill": ["Grasshopper", "Flowering Plant"],
    "Southern Yellow-billed Hornbill": ["Grasshopper", "Flowering Plant"],
    "Crested Barbet": ["Grasshopper", "Flowering Plant"],
    "Civet": ["Mouse", "Rat", "Grasshopper", "Flowering Plant"],
}


# ============================================================
# IMPACT LEVELS
# ============================================================

IMPACT_LABELS = {
    0: "Stable Ecosystem",
    1: "Slight Disturbance",
    2: "Moderate Risk",
    3: "High Risk",
    4: "Critical"
}


# ============================================================
# GET TROPHIC ROLE
# ============================================================

def get_role(species_name):

    if not species_name:
        return "unknown"

    species_name = species_name.strip().lower()

    for species, role in TROPHIC_ROLE.items():

        if species.lower() == species_name:
            return role

    return "unknown"


# ============================================================
# DATA EXTRACTION
# ============================================================

def get_observation_data():

    observations = (Observation.query .filter_by(status="Approved") .all())
    data = []

    for obs in observations:

        if not obs.species:
            continue

        data.append({
            "species": obs.species.specie_Common_Name,
            "habitat": obs.species.specie_Habitat,
            "location": obs.species.location,
            "population": obs.population_count,
            "date": obs.observation_date
        })

    return pd.DataFrame(data)


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_features(df):

    df = df.copy()

    if df.empty:
        return df

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    df = df.dropna(
        subset=[
            "population",
            "habitat",
            "location",
            "species",
            "date"
        ]
    )

    df["role"] = df["species"].apply(
        get_role
    )

    return df


# ============================================================
# SHANNON BIODIVERSITY INDEX
# ============================================================

def compute_biodiversity_index(df):

    if df.empty:
        return pd.DataFrame(columns=["location","date", "biodiversity_index"])

    results = []

    grouped = df.groupby(["location", "date"])

    for (location, date), group in grouped:
        total_population = (group["population"].sum())

        if total_population <= 0:
            biodiversity_index = 0

        else:
            proportions = (  group["population"] / total_population )
            
            proportions = proportions[ proportions > 0 ]

            biodiversity_index = -(
                proportions
                * np.log(proportions)
            ).sum()

        results.append({
            "location": location,
            "date": date,
            "biodiversity_index": biodiversity_index
        })

    return pd.DataFrame(results)


# ============================================================
# MACHINE LEARNING MODEL
# ============================================================

def train_species_impact_model():

    df = prepare_features(
        get_observation_data()
    )

    if df.empty:

        raise ValueError(
            "No approved observations available."
        )

    biodiversity = compute_biodiversity_index(
        df
    )

    pivot = (
        df.pivot_table(
            index=["location", "date"],
            columns="species",
            values="population",
            aggfunc="sum",
            fill_value=0
        )
        .reset_index()
    )

    merged = pivot.merge(
        biodiversity,
        on=["location", "date"]
    )

    species_cols = [
        column
        for column in pivot.columns
        if column not in ["location", "date"]
    ]

    if not species_cols:

        raise ValueError(
            "No species available for training."
        )

    X = merged[species_cols]

    y = merged["biodiversity_index"]

    # Not enough observations for train/test split
    if len(merged) < 5:
        model = LinearRegression()
        model.fit(X, y)
        impact = pd.Series( model.coef_, index=species_cols).sort_values()
        joblib.dump(  model,  "impact_model.pkl")
        return model, impact

    # Train/test split
    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )
    )

    # Train model
    model = LinearRegression()

    model.fit( X_train, y_train)

    # Prediction
    predictions = model.predict(
        X_test
    )

    # Model evaluation
    mae = mean_absolute_error(y_test, predictions)

    r2 = r2_score( y_test,predictions )

    print(f"AI Model MAE: {mae:.3f}")

    print( f"AI Model R²: {r2:.3f}")

    # Species coefficients
    impact = pd.Series(model.coef_,index=species_cols).sort_values()

    # Save model
    joblib.dump( model, "impact_model.pkl")

    return model, impact


# ============================================================
# ECOLOGICAL IMPACT CLASSIFICATION
# ============================================================

def classify_impact(role, pct_change):

    abs_change = abs(pct_change)

    # ========================================================
    # PRODUCER
    # ========================================================

    if role == "producer":
        if pct_change <= -50:
            return ( 4, "Severe vegetation loss. "  "This may reduce food and shelter "
                    "for herbivores and insects and might " "have a bad effect on the ecosystem.")

        elif pct_change <= -20:

            return (
                3,
                "Vegetation decline detected. "
                "Reduced plant availability may "
                "affect herbivore populations."
            )

        elif abs_change < 20:

            return ( 0, "Plant population is relatively " "stable and the ecosystem base " "appears healthy.")

        elif pct_change >= 200:
            return ( 2, "Very large increase in plant population " "detected. This may indicate unusual "
                    "vegetation growth or changes in grazing pressure. Continued monitoring is recommended." )

        elif pct_change >= 100:
            return ( 1, "Large increase in vegetation detected. This may indicate strong plant regeneration or favorable environmental conditions." )

        else:
            return (  1,  "Plant population is increasing. " "This is generally beneficial, although unusual increases should be monitored.")
        
    # HERBIVORE
    elif role == "herbivore" :
        if pct_change >= 150:
            return ( 4,  "Sharp herbivore population increase. This may cause excessive grazing pressure reduce vegetation, and increase competition for food.")

        elif pct_change <= -80:
            return ( 4, "Severe herbivore population decline. " "This may sharply reduce food availability for predators and signal a serious ecosystem disruption.")

        elif pct_change >= 100:

            return (3, "Herbivore population has increased " "significantly. Increased grazing pressure " "may reduce vegetation availability."
 )

        elif pct_change <= -60:
            return (2, "Herbivore population has declined ")

        elif abs_change < 40:
            return (0,"Herbivore population is relatively stable.")

        else:
            return (1, "Moderate herbivore population change " "detected. Continued monitoring is " "recommended.")

    # ========================================================
    # SMALL PREDATOR
    # ========================================================

    elif role == "small_predator":

        if pct_change <= -50:

            return ( 4,"Sharp decline in small predator population. This may increase the population of insects and other prey." )

        elif pct_change <= -20:
            return (3,"Small predator population is declining. " "This may reduce natural control of ""insect and small prey populations.")

        elif pct_change >= 50:

            return (3,"Small predator population has increased " "significantly. This may increase pressure " "on insect and small prey populations." )

        elif abs_change < 20:

            return (0,"Small predator population is relatively ""stable.")

        else:

            return ( 1,"Moderate change in small predator population detected. Continued monitoring is recommended.")

    # ========================================================
    # PREDATOR
    # ========================================================

    elif role == "predator":
        if pct_change <= -50:
            return (4,"Sharp predator decline. Reduced ""predator numbers may allow prey ""populations to increase." )

        elif pct_change <= -20:
            return ( 3, "Predator population is declining. " "Natural control of prey populations " "may be reduced.")

        elif pct_change >= 50:

            return ( 4, "Predator population has increased sharply. ""Excessive predator numbers may put strong " "pressure on prey populations and disrupt "
                "ecosystem balance.")

        elif pct_change >= 20:
            return ( 2, "Predator population is increasing. " "Higher predator numbers may reduce " "prey populations.")

        elif abs_change < 20:
            return (0,"Predator population is relatively stable ""and natural population control is maintained.")

        else:
            return ( 1, "Predator population is increasing. " "This may support ecosystem balance.")

    # ========================================================
    # LARGE PREDATOR
    # ========================================================

    elif role == "large_predator":

        if pct_change <= -50:

            return (
                4,
                "Sharp decline in large predator population. "
                "This may affect natural control of prey "
                "populations within the ecosystem."
            )

        elif pct_change <= -20:

            return (
                3,
                "Large predator population is declining. "
                "This may alter relationships between "
                "predators and their prey."
            )

        elif pct_change >= 50:

            return (
                3,
                "Large predator population has increased "
                "significantly. This may increase pressure "
                "on available prey."
            )

        elif abs_change < 20:

            return (
                0,
                "Large predator population is relatively stable."
            )

        else:

            return (
                1,
                "Moderate change in large predator population "
                "detected. Continued monitoring is recommended."
            )

    # ========================================================
    # OMNIVORE
    # ========================================================

    elif role == "omnivore":

        if pct_change <= -50:

            return (
                4,
                "Sharp omnivore decline. This may affect "
                "both plant consumption and interactions "
                "with smaller animal populations."
            )

        elif pct_change <= -20:

            return (
                3,
                "Omnivore population is declining. "
                "This may alter both plant and animal "
                "feeding relationships."
            )

        elif pct_change >= 50:

            return (
                3,
                "Omnivore population has increased "
                "significantly. This may increase pressure "
                "on both plant and animal food resources."
            )

        elif abs_change < 20:

            return (
                0,
                "Omnivore population is relatively stable."
            )

        else:

            return (
                1,
                "Moderate omnivore population change "
                "detected. Continued monitoring is recommended."
            )

    # ========================================================
    # DECOMPOSER
    # ========================================================

    elif role == "decomposer":

        if pct_change <= -20:

            return (
                2,
                "Decomposer population has declined. "
                "This may reduce nutrient recycling "
                "and affect soil fertility."
            )

        elif abs_change < 30:

            return (
                0,
                "Decomposer population is relatively "
                "stable and nutrient cycling appears normal."
            )

        else:

            return (
                0,
                "Increasing decomposer activity may "
                "support nutrient recycling and soil health."
            )

    # ========================================================
    # UNKNOWN
    # ========================================================

    else:

        return (
            1,
            "Unknown trophic role. Add this species "
            "to TROPHIC_ROLE for a more reliable "
            "ecological assessment."
        )

def get_affected_species(species_name, df):

    if df.empty:
        return []

    available_species = set(df["species"].unique())
    role = get_role(species_name)

    affected_species = set()

    # --- 1. Species that eat this species (its consumers) ---
    for predator, food_items in SPECIES_DIET.items():
        if species_name in food_items and predator in available_species:
            affected_species.add(predator)

    consumer_roles = FOOD_WEB.get(role, {}).get("feeds", [])
    for other_species in available_species:
        if other_species != species_name and get_role(other_species) in consumer_roles:
            affected_species.add(other_species)

    # --- 2. Species that this species eats (its prey) ---
    for prey in SPECIES_DIET.get(species_name, []):
        if prey in available_species:
            affected_species.add(prey)

    prey_roles = FOOD_WEB.get(role, {}).get("fed_by", [])
    for other_species in available_species:
        if other_species != species_name and get_role(other_species) in prey_roles:
            affected_species.add(other_species)

    affected_species.discard(species_name)

    return sorted(affected_species)


# ============================================================
# FOOD WEB EFFECT
# ============================================================

def propagate_effects(role, direction):
    web = FOOD_WEB.get(  role, {})
    affected = web.get( direction, [])

    if not affected:
        return ( "No further downstream effect mapped." )

    return ("Likely to affect: "+ ", ".join(affected)+ ".")


def generate_food_web_effect(species, pct_change, df):
    affected_species = get_affected_species(species, df)

    if not affected_species:
        return "No direct food-web dependency recorded."

    names = ", ".join(affected_species)
    role = get_role(species)

    if pct_change == 0:
        return (f"{species} is relatively stable, so no "
                f"major direct food-web change is expected.")

    direction = "decline" if pct_change < 0 else "increase"

    if role == "producer":
        if pct_change < 0:
            return (f"The decline in {species} may reduce food and habitat "
                    f"availability for species that depend on it, "
                    f"including: {names}.")
        else:
            return (f"The increase in {species} may improve food and "
                    f"habitat availability for species that depend on "
                    f"it, including: {names}.")

    elif role == "decomposer":
        if pct_change < 0:
            return (f"The decline in {species} may slow nutrient "
                    f"recycling, potentially reducing soil fertility "
                    f"that supports: {names}.")
        else:
            return (f"The increase in {species} may speed up nutrient "
                    f"recycling, benefiting: {names}.")

    else:  # herbivore, omnivore, predator, small_predator, large_predator
        if pct_change < 0:
            return (f"The decline in {species} may reduce food "
                    f"availability for, and ease predation pressure "
                    f"on: {names}.")
        else:
            return (f"The increase in {species} may raise food "
                    f"availability for, and increase predation "
                    f"pressure on: {names}.")

# MAIN MODEL SPECIES EFFECT REPORT

def generate_species_effect_report():
    df = prepare_features(get_observation_data())

    if df.empty:
        return pd.DataFrame()

    # BASELINE POPULATION
    baselines = (df.groupby("species")["population"].mean())

    # LATEST POPULATION
    latest = (df.sort_values("date").groupby("species").last()["population"])
    
    # TRAIN AI MODEL
    try:
        _, statistical_impact = (train_species_impact_model())

    except (ValueError,KeyError, ZeroDivisionError):
        statistical_impact = (pd.Series(dtype=float))

    # GENERATE SPECIES REPORT

    report = []

    for species in baselines.index:
        current = latest.get( species, 0 )
        baseline = baselines[species]

        # PERCENTAGE CHANGE

        if baseline > 0:
            pct_change = ( (current - baseline) / baseline ) * 100

        else:
            pct_change = 0

        # SPECIES ROLE
        role = get_role( species)

        # ECOLOGICAL IMPACT

        impact_level, reason = (classify_impact( role,pct_change))

        # AFFECTED SPECIES
        affected_species = (get_affected_species(species,df))

        # Remove current species
        affected_species = [
            s
            for s in affected_species
            if s != species
        ]

        # FOOD WEB EFFECT
        downstream = (generate_food_web_effect(species,pct_change,df))

        # STATISTICAL EFFECT

        statistical_effect = (statistical_impact.get(species,0) )

        # ADD RESULT
        report.append({
            "species": species,
            "role": role,
            "current_population": current,
            "baseline_population": round( baseline,1),
            "percent_change": round( pct_change, 1),
            "impact_level": impact_level,
            "impact_label": IMPACT_LABELS[impact_level],

            "reason": reason,
            "downstream_effect": downstream,
            "affected_species": affected_species,
            "statistical_effect_on_biodiversity":
                round(statistical_effect,4)
        })

    # SORT BY RISK
    return (pd.DataFrame(report) .sort_values("impact_level", ascending=False ) .reset_index(drop=True))

# TEST AI MODEL
if __name__ == "__main__":
    report = (generate_species_effect_report())

    if report.empty:
        print("No approved observations " "available for analysis.")

    else:
        print( report.to_string(index=False))