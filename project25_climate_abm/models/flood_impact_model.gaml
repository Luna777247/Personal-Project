/**
 * Climate Change Impact Simulation - Flood Model
 * Agent-Based Model using GAMA Platform
 *
 * This model simulates the impact of flooding on an urban community.
 * Agents represent households that respond to flood events.
 */

model FloodImpactModel

global {
    // Environment parameters
    int grid_width <- 50;
    int grid_height <- 50;

    // Climate parameters
    float rainfall_intensity <- 0.1; // mm per cycle
    float flood_threshold <- 10.0; // water level threshold for flooding

    // Population parameters
    int initial_population <- 100;

    // Simulation parameters
    int simulation_cycles <- 100;

    // Global variables
    float total_water_level <- 0.0;
    int flooded_households <- 0;
    int evacuated_households <- 0;

    init {
        // Create households
        create household number: initial_population;

        // Initialize environment
        create water_cell number: grid_width * grid_height;
    }

    reflex update_environment {
        // Simulate rainfall
        total_water_level <- total_water_level + rainfall_intensity;

        // Update water cells
        ask water_cell {
            water_level <- water_level + rainfall_intensity;
            if water_level > flood_threshold {
                is_flooded <- true;
            }
        }

        // Count flooded households
        flooded_households <- household count (each.is_flooded);
        evacuated_households <- household count (each.has_evaced);
    }
}

species water_cell {
    float water_level <- 0.0;
    bool is_flooded <- false;

    aspect default {
        draw square(1) color: is_flooded ? #blue : #lightblue;
    }
}

species household skills: [moving] {
    point home_location;
    bool is_flooded <- false;
    bool has_evaced <- false;
    float risk_tolerance <- rnd(0.0, 1.0);

    init {
        home_location <- location;
    }

    reflex check_flood_risk {
        water_cell my_cell <- water_cell closest_to self;
        if my_cell != nil and my_cell.is_flooded {
            is_flooded <- true;
            if flip(risk_tolerance) {
                // Evacuate
                has_evaced <- true;
                do wander;
            }
        }
    }

    aspect default {
        draw circle(0.5) color: has_evaced ? #red : (is_flooded ? #orange : #green);
    }
}

experiment FloodSimulation type: gui {
    parameter "Rainfall Intensity" var: rainfall_intensity min: 0.01 max: 1.0;
    parameter "Flood Threshold" var: flood_threshold min: 1.0 max: 50.0;
    parameter "Initial Population" var: initial_population min: 10 max: 500;

    output {
        display "Flood Map" {
            species water_cell;
            species household;
        }

        display "Statistics" {
            chart "Flood Impact" type: series {
                data "Flooded Households" value: flooded_households color: #red;
                data "Evacuated Households" value: evacuated_households color: #orange;
            }
        }
    }
}