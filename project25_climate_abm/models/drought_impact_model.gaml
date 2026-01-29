/**
 * Climate Change Impact Simulation - Drought Model
 * Agent-Based Model using GAMA Platform
 *
 * This model simulates the impact of drought on agricultural communities.
 * Agents represent farmers who manage water resources and crops.
 */

model DroughtImpactModel

global {
    // Environment parameters
    int grid_width <- 30;
    int grid_height <- 30;

    // Climate parameters
    float rainfall_probability <- 0.3; // probability of rain per cycle
    float evaporation_rate <- 0.05; // water loss per cycle
    float drought_threshold <- 5.0; // water level below which is drought

    // Population parameters
    int initial_farmers <- 50;

    // Simulation parameters
    int simulation_cycles <- 200;

    // Global variables
    float total_water_level <- 20.0;
    int affected_farmers <- 0;
    int failed_crops <- 0;

    init {
        // Create farmers
        create farmer number: initial_farmers;

        // Create water sources
        create water_source number: 5;

        // Initialize environment
        create land_cell number: grid_width * grid_height;
    }

    reflex update_environment {
        // Simulate weather
        if flip(rainfall_probability) {
            total_water_level <- total_water_level + rnd(1.0, 3.0);
        }

        // Evaporation
        total_water_level <- max(0.0, total_water_level - evaporation_rate);

        // Update water sources
        ask water_source {
            water_level <- water_level * (1 - evaporation_rate);
            if flip(rainfall_probability) {
                water_level <- water_level + rnd(0.5, 2.0);
            }
        }

        // Update farmers
        ask farmer {
            do manage_farm;
        }

        // Count affected farmers
        affected_farmers <- farmer count (each.is_affected);
        failed_crops <- farmer count (each.crop_failed);
    }
}

species land_cell {
    float soil_moisture <- 10.0;
    bool is_dry <- false;

    aspect default {
        draw square(1) color: is_dry ? #yellow : #green;
    }
}

species water_source {
    float water_level <- 15.0;
    float max_capacity <- 20.0;

    init {
        water_level <- rnd(10.0, max_capacity);
    }

    aspect default {
        draw circle(2) color: #blue;
    }
}

species farmer skills: [moving] {
    water_source my_water_source;
    bool is_affected <- false;
    bool crop_failed <- false;
    float water_usage <- rnd(0.5, 2.0);
    float crop_health <- 1.0;

    init {
        my_water_source <- one_of(water_source);
    }

    action manage_farm {
        if my_water_source != nil {
            float available_water <- my_water_source.water_level;
            if available_water > water_usage {
                // Irrigate
                my_water_source.water_level <- available_water - water_usage;
                crop_health <- min(1.0, crop_health + 0.1);
            } else {
                // Not enough water
                is_affected <- true;
                crop_health <- max(0.0, crop_health - 0.2);
                if crop_health < 0.3 {
                    crop_failed <- true;
                }
            }
        }
    }

    reflex move_to_water {
        if my_water_source != nil {
            do goto target: my_water_source.location;
        }
    }

    aspect default {
        draw triangle(1) color: crop_failed ? #red : (is_affected ? #orange : #green);
    }
}

experiment DroughtSimulation type: gui {
    parameter "Rainfall Probability" var: rainfall_probability min: 0.0 max: 1.0;
    parameter "Evaporation Rate" var: evaporation_rate min: 0.0 max: 0.2;
    parameter "Drought Threshold" var: drought_threshold min: 0.0 max: 20.0;
    parameter "Initial Farmers" var: initial_farmers min: 10 max: 200;

    output {
        display "Farm Landscape" {
            species land_cell;
            species water_source;
            species farmer;
        }

        display "Drought Statistics" {
            chart "Drought Impact" type: series {
                data "Affected Farmers" value: affected_farmers color: #orange;
                data "Failed Crops" value: failed_crops color: #red;
                data "Total Water Level" value: total_water_level color: #blue;
            }
        }
    }
}