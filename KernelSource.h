#ifndef FINAL_PROJECT_KERNELSOURCE_H
#define FINAL_PROJECT_KERNELSOURCE_H

// GPU kernel code
const char *gpuKernelSource = R"(
        __kernel void gameOfLife(__global const int* current_species,
                                __global int* next_species,
                                const int width, const int height,
                                const int num_species) {

            int x = get_global_id(0);
            int y = get_global_id(1);

            // Return if (x, y) is outside of grid
            if (x >= width || y >= height) return;

            // Get linear cell index
            int cellIndex = y * width + x;

            // Store local copy of species info
            int current_cell_species = current_species[cellIndex];

            // Initialize next state to current state
            next_species[cellIndex] = current_cell_species;

            int xCoords[8] = {x-1, x, x+1, x-1, x+1, x-1, x, x+1};
            int yCoords[8] = {y-1, y-1, y-1, y, y, y+1, y+1, y+1};
            int neighbourXCoord, neighbourYCoord;

            // If cell is alive (speciesID = -1 for dead cell)
            if (current_cell_species != -1) {
                int count = 0;
                int target_species = current_cell_species;

                for(int i = 0; i < 8; i++) {
                    neighbourXCoord = xCoords[i];
                    neighbourYCoord = yCoords[i];
                    if(neighbourXCoord >= 0 && neighbourXCoord < width && neighbourYCoord >= 0 && neighbourYCoord < height) {
                        int neighbourIndex = neighbourYCoord * width + neighbourXCoord;
                        if (current_species[neighbourIndex] == target_species) {
                            count++;
                        }
                    }
                }
                if (count < 2 || count > 3) {
                    next_species[cellIndex] = -1;   // Death, speciesID = -1 (N/A)
                }
            } else {
                    // Cell is dead - check for birth
                    int species_count[10] = {0};
                    for(int i = 0; i < 8; i++) {
                        neighbourXCoord = xCoords[i];
                        neighbourYCoord = yCoords[i];

                        if (neighbourXCoord >= 0 && neighbourXCoord < width && neighbourYCoord >= 0 && neighbourYCoord < height) {
                            int neighbourIndex = neighbourYCoord * width + neighbourXCoord;
                            int species_id = current_species[neighbourIndex];
                            if (species_id > 0) {
                                species_count[species_id - 1]++;
                            }
                        }
                    }

                // Check which species have exactly 3 neighbors
                int reproductionConditionMet[10];
                int num_candidates = 0;

                // Store the index of all the species that satisfy the
                // condition for new life.
                for (int i = 0; i < num_species; i++) {
                    if (species_count[i] == 3) {
                        reproductionConditionMet[num_candidates++] = i + 1;
                    }
                }

                // Of the species that can create new life, choose one at random
                if (num_candidates > 0) {
                    // Using 32-bit PCG hash "random" selection
                    uint seed = cellIndex;

                    uint state = seed * 747796405u + 2891336453u;
                    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
                    uint hash = (word >> 22u) ^ word;

                    // Pick candidate
                    int selected = reproductionConditionMet[hash % num_candidates];

                    next_species[cellIndex] = selected;
                }
            }
        }
)";

#endif
