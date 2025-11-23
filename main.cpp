#include <iostream>
#include <OpenCL/opencl.h>
#include <GLUT/glut.h>
#include <OpenGL/OpenGL.h>
#include <thread>

#include "Configs.h"
#include "KernelSource.h"

// ----------- OPENCL ----------- //
cl_device_id device_id;
cl_context context;
cl_command_queue commands;
cl_program program;
cl_kernel grid_update_kernel;
cl_mem old_species_ids_mem;
cl_mem species_ids_mem;
void initialiseOpenCL();
void cleanupOpenCL();

// ----------- GAME OF LIFE ----------- //
std::vector<int> oldSpeciesIDs; // Previous species IDs
std::vector<int> speciesIDs;    // Updated species IDs
GLubyte* gridData = nullptr;    // CPU pixel buffer
void initialiseGrid();
void updateGridState();

// ----------- OPENGL ----------- //
GLuint pixelBuffer;
void initialiseOpenGL(int argc, char** argv);
void displayFunc();             // Display callback
void idleFunc();                // Idle callback
void keyboardFunc(unsigned char key, int x, int y); // Keyboard callback
void setPixels();

// ----------- HOST THREAD ----------- //
std::thread hostThread;
void hostFunc();
std::atomic<bool> running{true};

int main(int argc, char** argv) {

    // Register cleanup function
    atexit(cleanupOpenCL);

    initialiseOpenGL(argc, argv);
    initialiseOpenCL();
    initialiseGrid();

    hostThread = std::thread(hostFunc);

    glutMainLoop();

    running = false;
    if(hostThread.joinable()) {
        hostThread.join();
    }

    cleanupOpenCL();

    return 0;
}

void initialiseOpenCL() {
    cl_int err;

    // Connect to a compute device
    int gpu = 1;
    err = clGetDeviceIDs(NULL, gpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU, 1, &device_id, NULL);
    if (err != CL_SUCCESS) {
        printf("Error: Failed to create a device group!\n");
        return;
    }

    CGLContextObj glContext = CGLGetCurrentContext();
    CGLShareGroupObj sharegroup = CGLGetShareGroup(glContext);

    cl_context_properties props[] = {
            CL_CONTEXT_PROPERTY_USE_CGL_SHAREGROUP_APPLE,
            (cl_context_properties)sharegroup,
            0
    };

    // Create OpenCL context with share group
    context = clCreateContext(props, 1, &device_id, nullptr, nullptr, &err);
    if (err != CL_SUCCESS) {
        printf("Error: Failed to create a compute context!\n");
        return;
    }

    // Create a command queue
    commands = clCreateCommandQueue(context, device_id, 0, &err);
    if (!commands) {
        printf("Error: Failed to create a command queue!\n");
        return;
    }

    // Create the compute program from the source character array
    program = clCreateProgramWithSource(context, 1, (const char **)&gpuKernelSource, NULL, &err);
    if (!program) {
        printf("Error: Failed to create compute program!\n");
        return;
    }

    // Build the program executable
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t len;
        char buffer[2048];

        printf("Error: Failed to build program executable!\n");
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
        return;
    }

    // Create the compute kernel
    grid_update_kernel = clCreateKernel(program, "gameOfLife", &err);
    if (!grid_update_kernel || err != CL_SUCCESS) {
        printf("Error: Failed to create compute grid_update_kernel!\n");
        return;
    }

    // Create buffers
    old_species_ids_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * WIDTH * HEIGHT, NULL, &err);
    species_ids_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * WIDTH * HEIGHT, NULL, &err);
    if (!old_species_ids_mem || !species_ids_mem) {
        printf("Error: Failed to allocate device memory!\n");
        return;
    }

    std::cout << "OpenCL initialized successfully!" << std::endl;
}

void cleanupOpenCL() {
    std::cout << "Freeing memory allocated by opencl\n";

    // Cleanup, free allocated memory
    if (old_species_ids_mem) clReleaseMemObject(old_species_ids_mem);
    if (species_ids_mem) clReleaseMemObject(species_ids_mem);
    if (program) clReleaseProgram(program);
    if (grid_update_kernel) clReleaseKernel(grid_update_kernel);
    if (commands) clReleaseCommandQueue(commands);
    if (context) clReleaseContext(context);
}

void initialiseGrid() {
    oldSpeciesIDs.resize(WIDTH * HEIGHT);
    speciesIDs.resize(WIDTH * HEIGHT);

    std::srand(static_cast<unsigned int>(std::time(nullptr)));
    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            int cellIndex = y * WIDTH + x;
            // Species ID ranges from 1 to NUMBER_OF_SPECIES
            int speciesID = std::rand() % NUMBER_OF_SPECIES + 1;
            oldSpeciesIDs[cellIndex]= speciesID;
        }
    }

    speciesIDs = oldSpeciesIDs;
}

void updateGridState() {
    // Swap grid buffers
    swap(speciesIDs, oldSpeciesIDs);

    cl_int err;

    // Copy current grid data to device
    err = clEnqueueWriteBuffer(commands, old_species_ids_mem, CL_TRUE, 0,
                               sizeof(int) * WIDTH * HEIGHT, oldSpeciesIDs.data(), 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        printf("Error: Failed to write to device memory!\n");
        return;
    }

    // Set kernel arguments
    err = 0;
    err |= clSetKernelArg(grid_update_kernel, 0, sizeof(cl_mem), &old_species_ids_mem);
    err |= clSetKernelArg(grid_update_kernel, 1, sizeof(cl_mem), &species_ids_mem);
    err |= clSetKernelArg(grid_update_kernel, 2, sizeof(int), &WIDTH);
    err |= clSetKernelArg(grid_update_kernel, 3, sizeof(int), &HEIGHT);
    err |= clSetKernelArg(grid_update_kernel, 4, sizeof(int), &NUMBER_OF_SPECIES);
    if (err != CL_SUCCESS) {
        printf("Error: Failed to set grid_update_kernel arguments! %d\n", err);
        return;
    }

    // Create a 2D array of WIDTH * HEIGHT work items
    size_t globalWorkSize[2] = {WIDTH, HEIGHT};
    // Execute the grid_update_kernel
    err = clEnqueueNDRangeKernel(commands, grid_update_kernel, 2, NULL, globalWorkSize, NULL, 0, NULL, NULL);
    if (err) {
        printf("Error: Failed to execute grid_update_kernel!\n");
        return;
    }

    // Wait for device to service commands
    clFinish(commands);

    // Read back the results from the device
    err |= clEnqueueReadBuffer(commands, species_ids_mem, CL_TRUE, 0, sizeof(int) * WIDTH * HEIGHT, speciesIDs.data(), 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        printf("Error: Failed to read output arrays! %d\n", err);
        return;
    }
}

void initialiseOpenGL(int argc, char** argv) {
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
    glutInitWindowSize(WIDTH, HEIGHT);
    glutInitWindowPosition(100, 100);   // Position of window on screen
    glutCreateWindow("Game of Life");

    // Set up pixel buffer object
    gridData = new GLubyte[WIDTH * HEIGHT * 3];
    glGenBuffers(1, &pixelBuffer);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pixelBuffer);
    glBufferData(GL_PIXEL_UNPACK_BUFFER,WIDTH * HEIGHT * 3, nullptr, GL_STREAM_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    // Set callbacks
    glutDisplayFunc(displayFunc);
    glutKeyboardFunc(keyboardFunc);
    glutIdleFunc(idleFunc);
}

void displayFunc() {
    // Clear buffer
    glClear(GL_COLOR_BUFFER_BIT);

    // Bind pixel buffer object
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pixelBuffer);

    void* ptr = glMapBuffer(GL_PIXEL_UNPACK_BUFFER, GL_WRITE_ONLY);
    if (ptr) {
        memcpy(ptr, gridData, WIDTH * HEIGHT * 3);
        glUnmapBuffer(GL_PIXEL_UNPACK_BUFFER);
    }

    // Draw pixels
    glDrawPixels(WIDTH, HEIGHT, GL_RGB, GL_UNSIGNED_BYTE, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    // Swap buffers
    glutSwapBuffers();
}

void idleFunc() {


    glutPostRedisplay();
}

void keyboardFunc(unsigned char key, int x, int y) {
    // Exit the glut window when user hits escape or q
    if (key == 27 || key == 'q') {
        exit(0);
    }
}

void setPixels() {
    for(int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            int cellIndex = y * WIDTH + x;
            int pixelIndex = cellIndex * 3; // Multiply by 3 for RGB

            switch (speciesIDs[cellIndex]) {
                case -1: // DEAD: Saddle brown
                    gridData[pixelIndex + 0] = 53;   // R
                    gridData[pixelIndex + 1] = 27;   // G
                    gridData[pixelIndex + 2] = 8;    // B
                    break;
                case 1: // SPECIES 1: Thistle
                    gridData[pixelIndex + 0] = 216;  // R
                    gridData[pixelIndex + 1] = 191;  // G
                    gridData[pixelIndex + 2] = 216;  // B
                    break;
                case 2: // SPECIES 2: Cadet blue
                    gridData[pixelIndex + 0] = 95;   // R
                    gridData[pixelIndex + 1] = 158;  // G
                    gridData[pixelIndex + 2] = 160;  // B
                    break;
                case 3: // SPECIES 3: Sea green
                    gridData[pixelIndex + 0] = 46;   // R
                    gridData[pixelIndex + 1] = 139;  // G
                    gridData[pixelIndex + 2] = 87;   // B
                    break;
                case 4: // SPECIES 4: Wheat
                    gridData[pixelIndex + 0] = 245;  // R
                    gridData[pixelIndex + 1] = 222;  // G
                    gridData[pixelIndex + 2] = 179;  // B
                    break;
                case 5: // SPECIES 5: Dark khaki
                    gridData[pixelIndex + 0] = 189;  // R
                    gridData[pixelIndex + 1] = 183;  // G
                    gridData[pixelIndex + 2] = 107;  // B
                    break;
                case 6: // SPECIES 6: Gold
                    gridData[pixelIndex + 0] = 255;  // R
                    gridData[pixelIndex + 1] = 215;  // G
                    gridData[pixelIndex + 2] = 0;    // B
                    break;
                case 7: // SPECIES 7: Orange red
                    gridData[pixelIndex + 0] = 255;  // R
                    gridData[pixelIndex + 1] = 69;   // G
                    gridData[pixelIndex + 2] = 0;    // B
                    break;
                case 8: // SPECIES 8: Firebrick
                    gridData[pixelIndex + 0] = 178;  // R
                    gridData[pixelIndex + 1] = 34;   // G
                    gridData[pixelIndex + 2] = 34;   // B
                    break;
                case 9: // SPECIES 9: Pale violet red
                    gridData[pixelIndex + 0] = 219;  // R
                    gridData[pixelIndex + 1] = 112;  // G
                    gridData[pixelIndex + 2] = 147;  // B
                    break;
                case 10: // SPECIES 10: Dark red
                    gridData[pixelIndex + 0] = 139;  // R
                    gridData[pixelIndex + 1] = 0;    // G
                    gridData[pixelIndex + 2] = 0;    // B
                    break;
                default: // ERROR: Magenta
                    gridData[pixelIndex + 0] = 255;  // R
                    gridData[pixelIndex + 1] = 0;    // G
                    gridData[pixelIndex + 2] = 255;  // B
            }
        }
    }
}

void hostFunc() {
    while(true) {
        updateGridState();
        setPixels();
        std::this_thread::sleep_for(std::chrono::milliseconds (30));
    }
}