/**
 * References:
 * - OpenCL initialisation, kernel, and function calls: https://developer.apple.com/library/archive/samplecode/OpenCL_Hello_World_Example/Listings/hello_c.html#//apple_ref/doc/uid/DTS40008187-hello_c-DontLinkElementID_4
 * - Pseudo-random next species selection: https://www.reedbeta.com/blog/hash-functions-for-gpu-rendering/
 *      - I used the 32-bit PCG hash algorithm, which the author credited to Jarzynski and Olano
 *      - The family of PCG (permuted congruential generator) algorithms was developed by M.E. O'Neill
 * - Creating and using Pixel Buffer Objects (PBOs): https://www.songho.ca/opengl/gl_vbo.html#create
 * - Profiling with events: Lecture slides
 **/

#include <iostream>
#include <GLUT/glut.h>
#include <OpenGL/OpenGL.h>
#include <OpenCL/opencl.h>

#include "Configs.h"
#include "KernelSource.h"

// ----------- OPENCL ----------- //
cl_device_id device_id;
cl_context context;
cl_command_queue gpu_commands;
cl_command_queue cpu_commands;
cl_program gpu_program;
cl_program cpu_program;
cl_kernel grid_update_kernel;
cl_kernel pixels_update_kernel;
cl_mem grid_mem;
cl_mem grid_cpu_mem;
cl_mem next_grid_mem;
cl_mem pixel_buffer_mem;
void initialiseOpenCL();
void cleanupOpenCL();

// ----------- GAME OF LIFE ----------- //
std::vector<int> grid;          // Previous species IDs
std::vector<int> nextGrid;      // Updated species IDs
int getDesiredNumberOfSpecies();
void initialiseGrid();
uint playGameOfLife();

// ----------- OPENGL ----------- //
GLuint pixelBuffer;
void initialiseOpenGL(int argc, char** argv);
void displayFunc();             // Display callback
void idleFunc();                // Idle callback
void keyboardFunc(unsigned char key, int x, int y); // Keyboard callback

// ----------- TEST VARIABLES ----------- //
bool testModeEnabled = true;
std::vector<double>hostWaitTimeus;
std::vector<double>kernelExecutionTimeus;
int iteration = 0;
int MAX_ITERATIONS = 100;

int main(int argc, char** argv) {
    NUMBER_OF_SPECIES = getDesiredNumberOfSpecies();

    if(testModeEnabled) {
        // Initialise test buffers
        hostWaitTimeus.resize(MAX_ITERATIONS);
        kernelExecutionTimeus.resize(MAX_ITERATIONS);
    }

    // Register cleanup function
    atexit(cleanupOpenCL);

    initialiseOpenGL(argc, argv);
    initialiseOpenCL();
    initialiseGrid();

    glutMainLoop();

    return 0;
}

void initialiseOpenCL() {
    cl_int err[2];

    // Connect to a compute device
    int gpu = 1;
    err[0] = clGetDeviceIDs(NULL, gpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU, 1, &device_id, NULL);
    if (err[0] != CL_SUCCESS) {
        printf("Error: Failed to create a device group!\n");
        return;
    }

    // Create OpenCL context with share group
    CGLContextObj glContext = CGLGetCurrentContext();
    CGLShareGroupObj shareGroup = CGLGetShareGroup(glContext);

    cl_context_properties props[] = {
            CL_CONTEXT_PROPERTY_USE_CGL_SHAREGROUP_APPLE,
            (cl_context_properties)shareGroup,
            0
    };

    context = clCreateContext(props, 1, &device_id, nullptr, nullptr, &err[0]);
    if (err[0] != CL_SUCCESS) {
        printf("Error: Failed to create a compute context!\n");
        return;
    }

    // Create CPU and GPU device command queues
    gpu_commands = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &err[0]);
    cpu_commands = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &err[1]);
    if (!gpu_commands || !cpu_commands) {
        printf("Error: Failed to create a command queue!\n");
        return;
    }

    // Create the compute programs from the source character arrays
    gpu_program = clCreateProgramWithSource(context, 1, (const char **)&gpuKernelSource, NULL, &err[0]);
    cpu_program = clCreateProgramWithSource(context, 1, (const char **)&cpuKernelSource, NULL, &err[1]);
    if (!gpu_program || !cpu_program) {
        printf("Error: Failed to create compute gpu_program!\n");
        return;
    }

    // Build the gpu_program and cpu_program executables
    err[0] = clBuildProgram(gpu_program, 0, NULL, NULL, NULL, NULL);
    err[1] = clBuildProgram(cpu_program, 0, NULL, NULL, NULL, NULL);
    if (err[0] != CL_SUCCESS || err[1] != CL_SUCCESS) {
        printf("Error: Failed to build program executable!\n");
        return;
    }

    // Create the GPU and CPU compute kernels
    grid_update_kernel = clCreateKernel(gpu_program, "gameOfLife", &err[0]);
    pixels_update_kernel = clCreateKernel(cpu_program, "writeToPixelBuffer", &err[1]);
    if (!grid_update_kernel || err[0] != CL_SUCCESS || !pixels_update_kernel || err[1] != CL_SUCCESS) {
        printf("Error: Failed to create compute kernel!\n");
        return;
    }

    // Create GPU buffers
    grid_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * WIDTH * HEIGHT, NULL, &err[0]);
    next_grid_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * WIDTH * HEIGHT, NULL, &err[0]);

    if (!grid_mem || !next_grid_mem) {
        printf("Error: Failed to allocate device memory!\n");
        return;
    }

    // Create CPU buffers
    grid_cpu_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * WIDTH * HEIGHT, NULL, &err[1]);
    pixel_buffer_mem = clCreateFromGLBuffer(context, CL_MEM_WRITE_ONLY, pixelBuffer, &err[1]);

    if (!pixel_buffer_mem || !grid_cpu_mem) {
        printf("Error: Failed to allocate device memory!\n");
        return;
    }

    std::cout << "OpenCL initialized successfully!" << std::endl;
}

void cleanupOpenCL() {
    std::cout << "Freeing memory allocated by OpenCL\n";

    if (grid_mem) clReleaseMemObject(grid_mem);
    if (next_grid_mem) clReleaseMemObject(next_grid_mem);
    if (grid_cpu_mem) clReleaseMemObject(grid_cpu_mem);
    if (pixel_buffer_mem) clReleaseMemObject(pixel_buffer_mem);
    if (gpu_program) clReleaseProgram(gpu_program);
    if (cpu_program) clReleaseProgram(cpu_program);
    if (grid_update_kernel) clReleaseKernel(grid_update_kernel);
    if (pixels_update_kernel) clReleaseKernel(pixels_update_kernel);
    if (gpu_commands) clReleaseCommandQueue(gpu_commands);
    if (cpu_commands) clReleaseCommandQueue(cpu_commands);
    if (context) clReleaseContext(context);

    // Get average times
    if(testModeEnabled) {
        double hostWaitTimeSum = 0, kernelExecTimeSum = 0;
        for(int i = 0; i < MAX_ITERATIONS; i++) {
            hostWaitTimeSum += hostWaitTimeus[i];
            kernelExecTimeSum += kernelExecutionTimeus[i];
        }
        std::cout << "Average host wait time: " << hostWaitTimeSum / MAX_ITERATIONS << "us\n";
        std::cout << "Average kernel execution time " << kernelExecTimeSum / MAX_ITERATIONS << "us\n";
    }
}

int getDesiredNumberOfSpecies() {
    int numberOfSpecies = 0;
    std::cout << "**********************************************************\n"
              << "\t\t\tWelcome to Game of Life!\n"
              << "**********************************************************\n"
              << "Enter your desired number of species (5-10): ";
    std::cin >> numberOfSpecies;

    while(std::cin.fail() || numberOfSpecies < 5 || numberOfSpecies > 10) {
        std::cin.clear();
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); // Discard invalid input
        std::cout << "Enter a valid number of species (5-10): ";
        std::cin >> numberOfSpecies;
    }

    return numberOfSpecies;
}

void initialiseGrid() {
    grid.resize(WIDTH * HEIGHT);
    nextGrid.resize(WIDTH * HEIGHT);

    std::srand(static_cast<unsigned int>(std::time(nullptr)));
    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            int cellIndex = y * WIDTH + x;
            // Species ID ranges from 1 to NUMBER_OF_SPECIES
            int speciesID = std::rand() % NUMBER_OF_SPECIES + 1;
            grid[cellIndex]= speciesID;
        }
    }

    nextGrid = grid;
}

void initialiseOpenGL(int argc, char** argv) {
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
    glutInitWindowSize(WIDTH, HEIGHT);
    glutInitWindowPosition(100, 100);
    glutCreateWindow("Game of Life");

    // Create Pixel Buffer Object (PBO)
    glGenBuffers(1, &pixelBuffer); // Create 1 buffer object, store its ID in pixelBuffer
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pixelBuffer); // Bind buffer to its ID
    glBufferData(GL_PIXEL_UNPACK_BUFFER,WIDTH * HEIGHT * 3, nullptr, GL_STREAM_DRAW); // Allocate (WIDTH * HEIGHT * 3) bytes to PBO
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0); // Unbind PBO

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

// "Host program"
void idleFunc() {

    uint returnVal = playGameOfLife();

    if(returnVal != 0) {
        glutPostRedisplay();
    }
    else {
        std::cout << "Something went wrong with the OpenCL setup and execution, exiting program\n";
        exit(0);
    }

    if(testModeEnabled) {
        iteration++;
        if(iteration >= MAX_ITERATIONS) {
            exit(0);
        }
    }
}

void keyboardFunc(unsigned char key, int x, int y) {
    // Exit the glut window when user hits escape or q
    if (key == 27 || key == 'q') {
        exit(0);
    }
}


uint playGameOfLife() {
    cl_int err;
    cl_event profiling_events[2];

    // ----------------- Swap host buffers -----------------
    std::swap(nextGrid, grid);

    // ----------------- Write grid N to GPU buffer -----------------
    err = clEnqueueWriteBuffer(gpu_commands, grid_mem, CL_TRUE, 0,
                               sizeof(int) * WIDTH * HEIGHT,
                               grid.data(), 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        printf("Error: Failed to write grid to GPU memory!\n");
        return 0;
    }

    // ----------------- Execute GPU kernel -----------------
    size_t global[2] = {WIDTH, HEIGHT};

    clSetKernelArg(grid_update_kernel, 0, sizeof(cl_mem), &grid_mem);
    clSetKernelArg(grid_update_kernel, 1, sizeof(cl_mem), &next_grid_mem);
    clSetKernelArg(grid_update_kernel, 2, sizeof(int), &WIDTH);
    clSetKernelArg(grid_update_kernel, 3, sizeof(int), &HEIGHT);
    clSetKernelArg(grid_update_kernel, 4, sizeof(int), &NUMBER_OF_SPECIES);

    // Start host side timer
    auto start = std::chrono::system_clock::now();
    err = clEnqueueNDRangeKernel(gpu_commands, grid_update_kernel,
                                 2, NULL, global, NULL, 0, NULL, &profiling_events[0]);
    if (err != CL_SUCCESS) {
        printf("Error: Failed to launch compute kernel!\n");
        return 0;
    }

    // ----------------- Write grid N to "CPU" buffer -----------------
    err = clEnqueueWriteBuffer(cpu_commands, grid_cpu_mem, CL_TRUE, 0,
                               sizeof(int) * WIDTH * HEIGHT,
                               grid.data(), 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        printf("Error: Failed to write CPU-grid buffer!\n");
        return 0;
    }

    // ----------------- Acquire OpenGL PBO for "CPU" kernel -----------------
    err = clEnqueueAcquireGLObjects(cpu_commands,
                                    1, &pixel_buffer_mem,
                                    0, NULL, NULL);
    if (err != CL_SUCCESS) {
        printf("Error: Failed to acquire GL buffer!\n");
        return 0;
    }

    // ----------------- Execute "CPU" kernel -----------------
    clSetKernelArg(pixels_update_kernel, 0, sizeof(cl_mem), &grid_cpu_mem);
    clSetKernelArg(pixels_update_kernel, 1, sizeof(cl_mem), &pixel_buffer_mem);
    clSetKernelArg(pixels_update_kernel, 2, sizeof(int), &WIDTH);
    clSetKernelArg(pixels_update_kernel, 3, sizeof(int), &HEIGHT);

    err = clEnqueueNDRangeKernel(cpu_commands, pixels_update_kernel,
                                 2, NULL, global, NULL,
                                 0, NULL, &profiling_events[1]);
    if (err != CL_SUCCESS) {
        printf("Error: Failed to launch pixel kernel!\n");
        return 0;
    }

    // ----------------- Release OpenGL PBO -----------------
    clEnqueueReleaseGLObjects(cpu_commands, 1, &pixel_buffer_mem, 0, NULL, NULL);

    // ----------------- Wait for GPU and "CPU" devices to service their commands -----------------
    clFinish(gpu_commands);
    clFinish(cpu_commands);

    // Stop host-side timer
    auto end = std::chrono::system_clock::now();

    // Print host-side computation time
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // ----------------- Get kernel runtimes -----------------
    clWaitForEvents(2, profiling_events);
    // Get kernel start and end times in nanoseconds
    cl_ulong gpu_kernel_start, gpu_kernel_end, cpu_kernel_start, cpu_kernel_end;
    size_t return_bytes;
    err = clGetEventProfilingInfo(profiling_events[0], CL_PROFILING_COMMAND_START, sizeof(cl_long), &gpu_kernel_start, &return_bytes);
    err |= clGetEventProfilingInfo(profiling_events[0], CL_PROFILING_COMMAND_END, sizeof(cl_long), &gpu_kernel_end, &return_bytes);
    err |= clGetEventProfilingInfo(profiling_events[1], CL_PROFILING_COMMAND_START, sizeof(cl_long), &cpu_kernel_start, &return_bytes);
    err |= clGetEventProfilingInfo(profiling_events[1], CL_PROFILING_COMMAND_END, sizeof(cl_long), &cpu_kernel_end, &return_bytes);
    if (err != CL_SUCCESS) {
        printf("Error: Failed to get event profiling info!\n");
        return 0;
    }
    double gpuKernelRuntime, cpuKernelRuntime, totalRuntime;
    gpuKernelRuntime = (double)(gpu_kernel_end - gpu_kernel_start) / 1000.0;
    cpuKernelRuntime = (double)(cpu_kernel_end - cpu_kernel_start) / 1000.0;
    totalRuntime = gpuKernelRuntime + cpuKernelRuntime;

    if(testModeEnabled) {
        hostWaitTimeus[iteration] = duration.count();
        kernelExecutionTimeus[iteration] = totalRuntime;
    }

    std::cout << "Kernel Runtime Info:\n";
    std::cout << "\tGPU next grid computation:\t\t\t" << gpuKernelRuntime << "us\n";
    std::cout << "\tCPU pixels:\t\t\t\t\t\t\t" << cpuKernelRuntime << "us\n";
    std::cout << "\tTotal runtime (sequential):\t\t\t" << totalRuntime << "us\n";
    std::cout << "\tTotal runtime (parallel):\t\t\t" << std::max(gpuKernelRuntime, cpuKernelRuntime) << "us\n";
    std::cout << "\tHost side wait time (sequential):\t" << duration.count() << "us\n";

    // ----------------- Read grid N+1 -----------------
    err = clEnqueueReadBuffer(gpu_commands, next_grid_mem, CL_TRUE, 0,
                              sizeof(int) * WIDTH * HEIGHT,
                              nextGrid.data(), 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        printf("Error: Failed to read back updated grid!\n");
        return 0;
    }

    return 1;
}
