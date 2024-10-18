# Matrix Multiplication with Multithreading: Performance Analysis

## Project Overview

This project demonstrates the implementation of matrix multiplication in Python using three different methods:
1. **Standard Matrix Multiplication** (Single-threaded)
2. **Multithreaded Matrix Multiplication** (One Thread per Row)
3. **Multithreaded Matrix Multiplication** (One Thread per Cell)

The primary goal is to compare the performance of these methods to understand how multithreading can improve or affect the performance of matrix operations, especially for larger matrices.

## Features

- **Standard Matrix Multiplication**: The basic method of multiplying two matrices using a triple-nested loop.
- **Multithreaded Matrix Multiplication (One Thread per Row)**: Utilizes multithreading where each row of the result matrix is computed by a separate thread.
- **Multithreaded Matrix Multiplication (One Thread per Cell)**: Creates a separate thread for each cell in the result matrix, offering maximum granularity in parallelism.
- **Performance Analysis**: Times the execution of each method and compares their performance on randomly generated matrices.

## File Structure

- `matrix_multiplication.py`: Main Python script containing the implementations of the three matrix multiplication methods and the performance analysis.
- `README.md`: Project description and instructions.

## Prerequisites

Before running the project, ensure you have the following installed:

- Python 3.x
- numpy (for matrix generation)

You can install the necessary libraries using `pip`:

```bash
pip install numpy
```

## How to run 
```bash
python3 matrix_multiplication.py
```
