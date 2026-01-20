## Multispecies Game of Life Implementation using OpenCL

This project implements Conway's Game of Life simulation using a heterogenous CPU-GPU approach. I used OpenCL to offload the next grid state (the fine-grained, data parallel computation) computation to the GPU. The CPU 
does some pre-processing for the rendering (assigning colours to live pixels), then submits the next frame buffer to OpenGL for rendering.

<p align="center">
  <img src="https://github.com/en4395/Photo-Dump/blob/main/GAME_OF_LIFE.gif" alt="Game of Life Simulation"  width="300" />
</p>
