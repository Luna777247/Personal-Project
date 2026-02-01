/**
 * Climate Change Impact Simulation - Heavy Rain Model
 * Agent-Based Model using GAMA Platform
 *
 * This model simulates the impact of heavy rainfall on urban drainage systems.
 * Agents represent stormwater management and community responses.
 */

model HeavyRainImpactModel

global {
    // Environment parameters
    int grid_width <- 40;
    int grid_height <- 40;

    // Climate parameters
    float base_rainfall <- 0.5; // base rainfall intensity
    float storm_probability <- 0.1; // probability of heavy storm
    float storm_intensity <- 5.0; // storm rainfall multiplier
    float drainage_capacity <- 2.0; // drainage system capacity

    // Population parameters
    int initial_residents <- 80;

    // Infrastructure parameters
    int drainage_systems <- 10;

    // Simulation parameters
    int simulation_cycles <- 150;

    // Global variables
    float current_rainfall <- 0.0;
    int flooded_areas <- 0;
    int overwhelmed_drainage <- 0;
    float total_damage <- 0.0;

    init {
        // Create residents
        create resident number: initial_residents;

        // Create drainage systems
        create drainage_system number: drainage_systems;

        // Create urban cells
        create urban_cell number: grid_width * grid_height;
    }

    reflex simulate_weather {
        // Determine rainfall
        if flip(storm_probability) {
            current_rainfall <- base_rainfall * storm_intensity;
        } else {
            current_rainfall <- base_rainfall + rnd(-0.2, 0.2);
        }

        // Update urban cells
        ask urban_cell {
            water_accumulation <- water_accumulation + current_rainfall;
            do update_flood_status;
        }

        // Update drainage systems
        ask drainage_system {
            do process_water;
        }

        // Count flooded areas
        flooded_areas <- urban_cell count (each.is_flooded);
        overwhelmed_drainage <- drainage_system count (each.is_overwhelmed);

        // Calculate damage
        total_damage <- total_damage + flooded_areas * 0.1;
    }
}

species urban_cell {
    float water_accumulation <- 0.0;
    bool is_flooded <- false;
    drainage_system nearest_drainage;

    init {
        nearest_drainage <- drainage_system closest_to self;
    }

    action update_flood_status {
        if nearest_drainage != nil {
            float drainage_effect <- nearest_drainage.capacity / distance_to(self, nearest_drainage);
            water_accumulation <- max(0.0, water_accumulation - drainage_effect);
        }

        is_flooded <- water_accumulation > 3.0;
    }

    aspect default {
        draw square(1) color: is_flooded ? #blue : #gray;
    }
}

species drainage_system {
    float capacity <- drainage_capacity;
    float current_load <- 0.0;
    bool is_overwhelmed <- false;

    action process_water {
        urban_cell[] nearby_cells <- urban_cell at_distance 5.0;
        current_load <- sum(nearby_cells collect each.water_accumulation);

        is_overwhelmed <- current_load > capacity;

        if not is_overwhelmed {
            // Reduce water in nearby cells
            ask nearby_cells {
                water_accumulation <- water_accumulation * 0.7;
            }
        }
    }

    aspect default {
        draw circle(1.5) color: is_overwhelmed ? #red : #cyan;
    }
}

species resident skills: [moving] {
    urban_cell home_cell;
    bool is_affected <- false;
    float risk_perception <- rnd(0.0, 1.0);

    init {
        home_cell <- one_of(urban_cell);
        location <- home_cell.location;
    }

    reflex respond_to_weather {
        if home_cell.is_flooded {
            is_affected <- true;
            if flip(risk_perception) {
                // Move to higher ground or safe area
                do wander amplitude: 10.0;
            }
        }
    }

    aspect default {
        draw circle(0.3) color: is_affected ? #purple : #white;
    }
}

experiment HeavyRainSimulation type: gui {
    parameter "Base Rainfall" var: base_rainfall min: 0.1 max: 2.0;
    parameter "Storm Probability" var: storm_probability min: 0.0 max: 0.5;
    parameter "Storm Intensity" var: storm_intensity min: 1.0 max: 10.0;
    parameter "Drainage Capacity" var: drainage_capacity min: 0.5 max: 5.0;
    parameter "Initial Residents" var: initial_residents min: 20 max: 200;

    output {
        display "Urban Flood Map" {
            species urban_cell;
            species drainage_system;
            species resident;
        }

        display "Rain Impact Statistics" {
            chart "Heavy Rain Impact" type: series {
                data "Flooded Areas" value: flooded_areas color: #blue;
                data "Overwhelmed Drainage" value: overwhelmed_drainage color: #red;
                data "Total Damage" value: total_damage color: #orange;
                data "Current Rainfall" value: current_rainfall color: #cyan;
            }
        }
    }
}