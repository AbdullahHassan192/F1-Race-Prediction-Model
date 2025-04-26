#config
TEAM_PERFORMANCE = {
    # quali_adjust --> time adjustment for qualifying (in seoncds)
    # race_score   --> race performance multiplier (1 is perfect)
    'Red Bull': {
        'quali_adjust': -0.20,  # 0.20s faster than base
        'race_score': 0.80      # 20% slower than perfect lap
    },
    'Ferrari': {
        'quali_adjust': -0.10,
        'race_score': 0.75
    },
    'McLaren': {
        'quali_adjust': -0.25,
        'race_score': 0.90
    },
    'Mercedes': {
        'quali_adjust': -0.15,
        'race_score': 0.77
    },
    'Aston Martin': {
        'quali_adjust': +0.15,
        'race_score': 0.62,
    },
    'RB': {
        'quali_adjust': +0.15, # Note: We have no RB drivers as of now since they're both rookies
        'race_score': 0.59,
    },
    'Williams': {
        'quali_adjust': +0.10,
        'race_score': 0.69
    },
    'Alpine': {
        'quali_adjust': +0.20,
        'race_score': 0.65
    },
    'Kick Sauber': {
        'quali_adjust': +0.30,
        'race_score': 0.55
    },
    'Haas': {
        'quali_adjust': +0.40,
        'race_score': 0.50
    }
}

DRIVER_PERFORMANCE = {
    # quali_adjust --> individual qualifying boost
    # race_skill   --> race pace multiplier
    # wet_skill    --> wet weather race pace multiplier
    'Max Verstappen': {
        'quali_adjust': -0.20, # 0.2s faster than base, adds on top of team adjustment
        'race_skill': 1.04, # 4% faster than base time
        'wet_skill': 0.98 # only 2% loss in rain 
    },
    'Charles Leclerc': {
        'quali_adjust': -0.10,
        'race_skill': 1.02,
        'wet_skill': 0.96
    },
    'Carlos Sainz': {
        'quali_adjust': -0.05,
        'race_skill': 1.01,
        'wet_skill': 0.97
    },
    'Lando Norris': {
        'quali_adjust': -0.13,
        'race_skill': 1.03,
        'wet_skill': 0.95
    },
    'Oscar Piastri': {
        'quali_adjust': -0.15,
        'race_skill': 1.03,
        'wet_skill': 0.94
    },
    'Lewis Hamilton': {
        'quali_adjust': -0.08,
        'race_skill': 1.02,
        'wet_skill': 0.98
    },
    'George Russell': {
        'quali_adjust': -0.12,
        'race_skill': 1.02,
        'wet_skill': 0.95
    },
    'Fernando Alonso': {
        'quali_adjust': -0.05,
        'race_skill': 1.01,
        'wet_skill': 0.99
    },
    'Lance Stroll': {
        'quali_adjust': +0.02,
        'race_skill': 0.99,
        'wet_skill': 0.92
    },
    'Yuki Tsunoda': {
        'quali_adjust': -0.01,
        'race_skill': 1.01,
        'wet_skill': 0.93
    },
    'Alexander Albon': {
        'quali_adjust': -0.03,
        'race_skill': 1.01,
        'wet_skill': 0.95
    },
    'Pierre Gasly': {
        'quali_adjust': +0.02,
        'race_skill': 1.01,
        'wet_skill': 0.96
    },
    'Esteban Ocon': {
        'quali_adjust': +0.03,
        'race_skill': 1.00,
        'wet_skill': 0.95
    },
    'Nico Hulkenberg': {
        'quali_adjust': +0.05,
        'race_skill': 1.00,
        'wet_skill': 0.97
    },
}

DRIVER_MAPPING = {
    "Oscar Piastri": "PIA", "George Russell": "RUS", "Lando Norris": "NOR", "Max Verstappen": "VER",
    "Lewis Hamilton": "HAM", "Charles Leclerc": "LEC", "Yuki Tsunoda": "TSU", "Alexander Albon": "ALB",
    "Esteban Ocon": "OCO", "Nico Hülkenberg": "HUL", "Fernando Alonso": "ALO", "Lance Stroll": "STR",
    "Carlos Sainz Jr.": "SAI", "Pierre Gasly": "GAS"
}

DRIVER_ORDER = [
    "Oscar Piastri", "George Russell", "Lando Norris", "Max Verstappen",
    "Lewis Hamilton", "Charles Leclerc", "Yuki Tsunoda", "Alexander Albon",
    "Esteban Ocon", "Nico Hülkenberg", "Fernando Alonso", "Lance Stroll",
    "Carlos Sainz Jr.", "Pierre Gasly"
]


DRIVER_TEAMS = {
    'Max Verstappen': 'Red Bull',
    'Yuki Tsunoda': 'Red Bull',
    'Charles Leclerc': 'Ferrari',
    'Lewis Hamilton': 'Ferrari',
    'Lando Norris': 'McLaren',
    'Oscar Piastri': 'McLaren',
    'George Russell': 'Mercedes',
    'Fernando Alonso': 'Aston Martin',
    'Lance Stroll': 'Aston Martin',
    'Alexander Albon': 'Williams',
    'Carlos Sainz': 'Williams',
    'Pierre Gasly': 'Alpine',
    'Nico Hulkenberg': 'Kick Sauber',
    'Esteban Ocon': 'Haas'
}

CIRCUIT_DATA = {
    'Saudi Arabia': {
        'base_quali': 87.7, # Base Q3 lap time (s)
        'base_race': 91.0, # Base race lap time (s)
    },

    # Add more tracks if you want to predict on them
    
}

DEFAULT_WEATHER = {
    'rain_prob': 0.1,  # 10% chance of rain
    'temp': 25.0,      # 25 degrees celsius 
    'humidity': 0.4    # on a 0-1 scale
}

