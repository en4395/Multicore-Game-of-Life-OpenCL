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
cl_program gpu_program;
cl_program cpu_program;
cl_kernel grid_update_kernel;
cl_kernel pixels_update_kernel;
cl_mem old_species_ids_mem;
cl_mem species_ids_mem;
cl_mem pixel_buffer_mem;
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
void setPixelsNoCL();

int main(int argc, char** argv) {

    // Register cleanup function
    atexit(cleanupOpenCL);

    initialiseOpenGL(argc, argv);
    initialiseOpenCL();
    initialiseGrid();

    glutMainLoop();

    cleanupOpenCL();

    return 0;
}

void initialiseOpenCL() {
    cl_int err;

    // ----------- SHARED FOR NOW ----------- //
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

    // ----------- GPU KERNEL ----------- //

    // Create the compute program from the source character array
    gpu_program = clCreateProgramWithSource(context, 1, (const char **)&gpuKernelSource, NULL, &err);
    if (!gpu_program) {
        printf("Error: Failed to create compute gpu_program!\n");
        return;
    }

    // Build the gpu_program executable
    err = clBuildProgram(gpu_program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t len;
        char buffer[2048];

        printf("Error: Failed to build gpu_program executable!\n");
        clGetProgramBuildInfo(gpu_program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
        return;
    }

    // Create the compute kernel
    grid_update_kernel = clCreateKernel(gpu_program, "gameOfLife", &err);
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

    // ----------- CPU KERNEL ----------- //
    // Create the compute gpu_program from the source character array
    cpu_program = clCreateProgramWithSource(context, 1, (const char **)&cpuKernelSource, NULL, &err);
    if (!cpu_program) {
        printf("Error: Failed to create compute gpu_program!\n");
        return;
    }

    // Build the gpu_program executable
    err = clBuildProgram(cpu_program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t len;
        char buffer[2048];

        printf("Error: Failed to build gpu_program executable!\n");
        clGetProgramBuildInfo(cpu_program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
        return;
    }

    // Create the compute kernel
    pixels_update_kernel = clCreateKernel(cpu_program, "writeToPixelBuffer", &err);
    if (!pixels_update_kernel || err != CL_SUCCESS) {
        printf("Error: Failed to create compute grid_update_kernel!\n");
        return;
    }

    pixel_buffer_mem = clCreateFromGLBuffer(context, CL_MEM_WRITE_ONLY, pixelBuffer, &err);

    if (!pixel_buffer_mem) {
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
    if (gpu_program) clReleaseProgram(gpu_program);
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
    glClear(GL_COLOR_BUFFER_BIT);

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pixelBuffer);
    glDrawPixels(WIDTH, HEIGHT, GL_RGB, GL_UNSIGNED_BYTE, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    glutSwapBuffers();
}

void idleFunc() {
    updateGridState();
    setPixelsNoCL();
    //setPixels();
    glutPostRedisplay();
}

void keyboardFunc(unsigned char key, int x, int y) {
    // Exit the glut window when user hits escape or q
    if (key == 27 || key == 'q') {
        exit(0);
    }
}

void setPixels() {
    cl_int err = clEnqueueAcquireGLObjects(commands, 1, &pixel_buffer_mem, 0, nullptr, nullptr);

    err |= clSetKernelArg(pixels_update_kernel, 0, sizeof(cl_mem), &species_ids_mem);
    err |= clSetKernelArg(pixels_update_kernel, 1, sizeof(cl_mem), &pixel_buffer_mem);
    err |= clSetKernelArg(pixels_update_kernel, 2, sizeof(int), &WIDTH);
    err |= clSetKernelArg(pixels_update_kernel, 3, sizeof(int), &HEIGHT);

    if (err != CL_SUCCESS) {
        printf("Error: Failed to set grid_update_kernel arguments! %d\n", err);
        return;
    }

    // Create a 2D array of WIDTH * HEIGHT work items
    size_t globalWorkSize[2] = {WIDTH, HEIGHT};
    // Execute the grid_update_kernel
    err = clEnqueueNDRangeKernel(commands, pixels_update_kernel, 2, NULL, globalWorkSize, NULL, 0, NULL, NULL);
    if (err) {
        printf("Error: Failed to execute grid_update_kernel!\n");
        return;
    }

    clEnqueueReleaseGLObjects(commands, 1, &pixel_buffer_mem, 0, nullptr, nullptr);
    clFinish(commands);
}

void setPixelsNoCL() {
    for(int y = 0; y < HEIGHT; y++) {
        for(int x = 0; x < WIDTH; x++) {
            int cellIndex = y * WIDTH + x;
            int speciesID = oldSpeciesIDs[cellIndex];
            // Map speciesID to RGB color
            GLubyte r, g, b;
            switch (speciesID) {
                case -1:   r = 53;  g = 27;  b = 8;    break;  // DEAD: Saddle brown
                case 1:    r = 216; g = 191; b = 216;  break;  // SPECIES 1: Thistle
                case 2:    r = 95;  g = 158; b = 160;  break;  // SPECIES 2: Cadet blue
                case 3:    r = 46;  g = 139; b = 87;   break;  // SPECIES 3: Sea green
                case 4:    r = 245; g = 222; b = 179;  break;  // SPECIES 4: Wheat
                case 5:    r = 189; g = 183; b = 107;  break;  // SPECIES 5: Dark khaki
                case 6:    r = 255; g = 215; b = 0;    break;  // SPECIES 6: Gold
                case 7:    r = 255; g = 69;  b = 0;    break;  // SPECIES 7: Orange red
                case 8:    r = 178; g = 34;  b = 34;   break;  // SPECIES 8: Firebrick
                case 9:    r = 219; g = 112; b = 147;  break;  // SPECIES 9: Pale violet red
                case 10:   r = 139; g = 0;   b = 0;    break;  // SPECIES 10: Dark red
                default:   r = 255; g = 0;   b = 255;  break;  // ERROR: Magenta
            }

            // Update CPU pixel buffer (row-major RGB)
            int pixelIndex = cellIndex * 3;
            gridData[pixelIndex]     = r;
            gridData[pixelIndex + 1] = g;
            gridData[pixelIndex + 2] = b;
        }
    }

    // Upload the CPU buffer to the OpenGL pixel buffer
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pixelBuffer);
    glBufferSubData(GL_PIXEL_UNPACK_BUFFER, 0, WIDTH * HEIGHT * 3, gridData);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
}