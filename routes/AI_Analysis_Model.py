
import pandas as pd
import numpy as np
import joblib

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from models import Observation

# CBU NATURE PARK ECOSYSTEM DEFINITION

# Trophic roles:
TROPHIC_ROLE = {
    # PRODUCERS
    "Tree": "producer",
    "Grass": "producer",
    "Flowering Plant": "producer",
    "Acacia": "producer",
    "Miombo Tree": "producer",
    "Mopane": "producer",
    "Grass Species": "producer",
    "Plant": "producer",
    "Miombo Tree": "producer",
    "Brachystegia": "producer",
    "Julbernardia": "producer",
    "Isoberlinia": "producer",
    "Faidherbia Albid": "producer",
    "khasi pine": "producer",
    "apple-ring acacia": "producer",
    "Ana Tree": "producer",
    

    # HERBIVORES
    # Small herbivores
    "Grasshopper": "herbivore",
    "Caterpillar": "herbivore",
    "Snail": "herbivore",
    "Rodent": "herbivore",
    "Small Mammal": "herbivore",

    # Antelopes
    "Impala": "herbivore",
    "Puku": "herbivore",
    "Kudu": "herbivore",
    "Sable Antelope": "herbivore",
    "Bushbuck": "herbivore",
    "Duiker": "herbivore",

    # Other herbivores
    "Warthog": "herbivore",
    "Zebra": "herbivore",
    "Buffalo": "herbivore",
    "Rabbit": "herbivore",

    # PREDATORS / INSECTIVORES
    "Spider": "small_predator",
    "Snakes":"predator",
    "Praying Mantis": "small_predator",
    "Frogs": "small_predator",
    "Lizard": "small_predator",
    "Snake": "predator",
    "Bird": "predator",
    "Insectivorous Bird": "small_predator",
    "Mongoose": "predator",
    "Civet": "predator",
    
    # larger Predator
    "Monitor Lizard":"large_predator",
    "Eagle":"large_predator",
    "Crocodile":"large_predator",
    "Snake":"large_predator",
    
    # DECOMPOSERS
    "Fungi": "decomposer",
    "Bacteria": "decomposer",
    "Termite": "decomposer",
    "Earthworm": "decomposer",
}

# FOOD WEB
FOOD_WEB = {
    "producer": { "feeds": ["herbivore"], "fed_by": ["decomposer"]},
    "herbivore": {"feeds": ["predator","large_predator"],"fed_by": ["producer"] },

    "predator": {
        "feeds": ["large_predator","predator"],
        "fed_by": ["herbivore"]
    },

    "decomposer": {
        "feeds": ["producer"],
        "fed_by": ["producer","herbivore","predator"]
    }
}

# IMPACT LEVELS
IMPACT_LABELS = {
    0: "Stable Ecosystem",
    1: "Slight Disturbance",
    2: "Moderate Risk",
    3: "High Risk",
    4: "Critical"
}
# GET TROPHIC ROLE
def get_role(species_name):
    if not species_name:
        return "unknown"
    species_name = species_name.strip().lower()
    for species, role in TROPHIC_ROLE.items():
        if species.lower() == species_name:
            return role
    return "unknown"

# DATA EXTRACTION

def get_observation_data():
    observations = (Observation.query.filter_by(status="Approved").all())
    data = []
    for obs in observations:
        if not obs.species:
            continue
        data.append({ "species": obs.species.specie_Common_Name,
            "habitat":obs.species.specie_Habitat,
            "location":obs.species.location,
            "population":obs.population_count,
            "date":obs.observation_date})
    return pd.DataFrame(data)

# PREPARE DATA
def prepare_features(df):
    df = df.copy()
    if df.empty:
        return df
    df["date"] = pd.to_datetime(  df["date"], errors="coerce")
    df = df.dropna( subset=[ "population", "habitat", "location", "species","date"])
    df["role"] = df["species"].apply( get_role)
    return df

# SHANNON BIODIVERSITY INDEX
def compute_biodiversity_index(df):
    if df.empty:
        return pd.DataFrame(columns=["location","date","biodiversity_index"] )
    results = []
    grouped = df.groupby( ["location", "date"])

    for (location, date), group in grouped:
        total_population = (group["population"].sum())
        if total_population <= 0:
            biodiversity_index = 0
        else:
            proportions = (group["population"] / total_population )
            proportions = proportions[ proportions > 0 ]
            biodiversity_index = -( proportions *np.log(proportions)).sum()
            
        results.append({"location": location,"date": date, "biodiversity_index": biodiversity_index })
    return pd.DataFrame(results)

# MACHINE LEARNING MODEL
def train_species_impact_model():
    df = prepare_features(get_observation_data())
    if df.empty:
        raise ValueError("No approved observations available." )
    
    biodiversity = ( compute_biodiversity_index(df))
    
    pivot = (df.pivot_table(index=["location","date"],columns="species", values="population",aggfunc="sum", fill_value=0).reset_index())

    merged = pivot.merge(biodiversity, on=["location","date"])

    species_cols = [column      
        for column in pivot.columns
        if column not in ["location", "date"]]

    if not species_cols:
        raise ValueError("No species available for training.")
    
    X = merged[species_cols]
    y = merged["biodiversity_index"]

    # Not enough observations for train/test split
    if len(merged) < 5:
        model = LinearRegression()
        model.fit(X, y)
        impact = pd.Series( model.coef_,index=species_cols ).sort_values()
        joblib.dump( model, "impact_model.pkl")
        return model, impact

    # Train/test split
    X_train, X_test, y_train, y_test = (train_test_split(X, y, test_size=0.2, random_state=42))

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Prediction
    predictions = model.predict( X_test)

    # Model evaluation
    mae = mean_absolute_error(y_test,predictions)
    r2 = r2_score(y_test, predictions)
    print(f"AI Model MAE: {mae:.3f}")
    print( f"AI Model R²: {r2:.3f}")
    
    # Species coefficients
    impact = pd.Series( model.coef_, index=species_cols).sort_values()
    # Save model
    joblib.dump( model, "impact_model.pkl")
    return model, impact

# ECOLOGICAL IMPACT CLASSIFICATION

def classify_impact(role, pct_change):
    abs_change = abs( pct_change)
    # PRODUCER
    if role == "producer":
        if pct_change <= -50:
            return (4, "Severe vegetation loss. ""This may reduce food and shelter " "for herbivores and insects and cause cascading biodiversity effects.")
        elif pct_change <= -20:
            return (3, "Vegetation decline detected. " "Reduced plant availability may ""affect herbivore populations.")
        elif abs_change < 20:
            return (0, "Plant population is relatively " "stable and the ecosystem base " "appears healthy.")
        elif pct_change >= 200:
            return ( 2, "Very large increase in plant population detected. " "This may indicate unusual vegetation growth or " "changes in grazing pressure. Continued monitoring "
                    "is recommended.")
        elif pct_change >= 100:
            return (1, "Large increase in vegetation detected. This may " "indicate strong plant regeneration or favorable "
            "environmental conditions.")
        
        else:
            return ( 1, "Plant population is increasing. " "This is generally beneficial, " "although unusual increases " "should be monitored.")

    # HERBIVORE
    elif role == "herbivore":
        if pct_change >= 150:
            return (4, "Sharp herbivore population increase. This may cause "
                    "excessive grazing pressure, reduce vegetation, and "
                    "increase competition for food.")
        elif pct_change <= -80:
            return (4, "Severe herbivore population decline. This may sharply ""reduce food availability for predators and signal a "
                    "serious ecosystem disruption.")
        elif pct_change >= 100:
            return (3, "Herbivore population has increased significantly. "
                    "Increased grazing pressure may reduce vegetation ""availability.")
        elif pct_change <= -60:
            return ( 2, "Herbivore population has declined ""significantly. This may reduce food ""availability for predators.")
        elif abs_change < 40:
            return (0,"Herbivore population is relatively " "stable." )
        else:
            return (1,"Moderate herbivore population change ""detected. Continued monitoring " "is recommended.")
        
    # PREDATOR
    elif role == "large_predator":
        if pct_change <= -50:
            return (4,"Sharp predator decline. Reduced " "predator numbers may allow prey " "populations to increase.")
        elif pct_change <= -20:
            return (3,"Predator population is declining. " "Natural control of prey populations " "may be reduced.")
        elif pct_change >= 50:
            return ( 4, "Predator population has increased sharply. " "Excessive predator numbers may put strong pressure " "on prey populations and disrupt ecosystem balance." )
        elif pct_change >= 20:
            return ( 2, "Predator population is increasing. Higher predator " "numbers may reduce prey populations." )
        elif abs_change < 20:
            return ( 0, "Predator population is relatively "  "stable and natural population ""control is maintained." )
        else:
            return (1,"Predator population is increasing. " "This may support ecosystem balance.")
        
    if role == "small_predator":
        if pct_change <= -50:
            return (4, "Severe decline in small predator population. "
                    "This may cause an increase in prey species such as "
                    "insects, frogs, rodents, or lizards and may disrupt "
                    "the food web.")
        elif pct_change <= -20:
            return (3, "Small predator population is declining. " "Reduced predation may allow prey populations to increase "
                    "and may affect ecosystem balance.")
        elif abs_change < 20:
            return (0, "Small predator population is relatively stable. ""Predation pressure appears balanced and the ecosystem "
                    "food web appears relatively stable.")
        elif pct_change >= 200:
            return (2, "Very large increase in small predator population detected. " "This may increase predation pressure on prey species "
                    "and potentially reduce their populations. Continued " "monitoring is recommended.")
        elif pct_change >= 100:
            return (1, "Large increase in small predator population detected. " "This may increase predation on prey species and change " "the balance of the food web.")

    # DECOMPOSER
    elif role == "decomposer":
        if pct_change <= -20:
            return (2,"Decomposer population has declined. " "This may reduce nutrient recycling ""and affect soil fertility.")
        elif abs_change < 30:
            return (0, "Decomposer population is relatively " "stable and nutrient cycling appears " "normal.")
        else:
            return (0,"Increasing decomposer activity may " "support nutrient recycling and " "soil health.")
    # UNKNOWN
    else:
        return ( 1, "Unknown trophic role. Add this species ""to TROPHIC_ROLE for a more reliable ""ecological assessment.")

# FIND AFFECTED SPECIES

def get_affected_species( df,role):
    affected_roles = (FOOD_WEB.get(role, {}).get("feeds", []))
    if not affected_roles:
        return []

    affected_species = (df[df["role"].isin(affected_roles)]["species"].unique().tolist())
    return affected_species

# FOOD WEB EFFECT
def propagate_effects(role,direction):
    web = FOOD_WEB.get( role,{} )
    affected = web.get(direction, [] )
    
    if not affected:
        return ("No further downstream ""effect mapped.")
    return ("Likely to affect: "+ ", ".join(affected) + ".")


# MAIN MODEL SPECIES EFFECT REPORT

def generate_species_effect_report():
    df = prepare_features(get_observation_data())

    if df.empty:
        return pd.DataFrame()


    # BASELINE POPULATION
    baselines = ( df.groupby("species")[  "population" ].mean() )

    # LATEST POPULATION
    latest = (df.sort_values("date").groupby("species").last()["population"])

    # TRAIN AI MODEL
    try:
        _, statistical_impact = ( train_species_impact_model())

    except (ValueError, KeyError, ZeroDivisionError):
        statistical_impact = (pd.Series(dtype=float) )

    # GENERATE SPECIES REPORT
    report = []
    for species in baselines.index:
        current = latest.get( species,0)
        baseline = baselines[species]
        
        # Percentage change
        if baseline > 0:
            pct_change = (( current - baseline )/ baseline) * 100

        else:
            pct_change = 0

        
        # Species role
        role = get_role(species)

        # Ecological impact
        impact_level, reason = (classify_impact( role, pct_change))
        
        # Affected species
        affected_species = ( get_affected_species( df, role ) )

        # Remove current species
        affected_species = [
            s
            for s in affected_species
            if s != species
        ]

        # Food web effect
        downstream = (propagate_effects( role, "feeds"))

        # Statistical effect
        statistical_effect = (statistical_impact.get( species, 0))
        # Add result
        report.append({
            "species":species,
            "role": role,
            "current_population": current,
            "baseline_population": round( baseline, 1 ),
            "percent_change": round( pct_change, 1),
            "impact_level": impact_level,
            "impact_label": IMPACT_LABELS[ impact_level ],
            "reason":reason,
            "downstream_effect": downstream,
            "affected_species":affected_species,
            "statistical_effect_on_biodiversity": round( statistical_effect,  4 )
        })
    # SORT BY RISK
    return (pd.DataFrame(report).sort_values( "impact_level", ascending=False).reset_index(drop=True))

# TEST AI MODEL
if __name__ == "__main__":
    report = (generate_species_effect_report())
    if report.empty:
        print("No approved observations " "available for analysis." )
    else:
        print(report.to_string( index=False ))