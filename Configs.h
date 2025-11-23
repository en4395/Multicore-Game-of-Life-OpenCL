#ifndef FINAL_PROJECT_CONFIGS_H
#define FINAL_PROJECT_CONFIGS_H

extern int NUMBER_OF_SPECIES;

// Grid parameters
constexpr int WIDTH = 1024;
constexpr int HEIGHT = 768;

// Timing
constexpr int FRAME_RATE = 30; // 30 FPS
constexpr int FRAME_DELAY = 1000 / FRAME_RATE; // ≈33ms

// Colours
constexpr float CELL_COLOURS[11][3] = {
        {0.21f, 0.11f, 0.032f}, // DEAD: Saddle brown rgb(53,27,8)
        {0.85f, 0.75f, 0.85f},  // SPECIES 1: Thistle rgb(216,191,216)
        {0.37f, 0.62f, 0.63f},  // SPECIES 2: Cadet blue rgb(95,158,160)
        {0.18f, 0.55f, 0.34f},  // SPECIES 3: Sea green rgb(46,139,87)
        {0.96f, 0.87f, 0.70f},  // SPECIES 4: Wheat rgb(245,222,179)
        {0.74f, 0.72f, 0.42f},  // SPECIES 5: Dark khaki rgb(189,183,107)
        {1.00f, 0.84f, 0.00f},  // SPECIES 6: Gold rgb(255,215,0)
        {1.00f, 0.27f, 0.00f},  // SPECIES 7: Orange red rgb(255,69,0)
        {0.70f, 0.13f, 0.13f},  // SPECIES 8: Firebrick rgb(178,34,34)
        {0.86f, 0.44f, 0.58f},  // SPECIES 9: Pale violet red rgb(219,112,147)
        {0.55f, 0.00f, 0.00f}   // SPECIES 10: Dark red rgb(139,0,0);
};

#endif
