#pragma once

#include <math.h>

typedef unsigned int uint;
typedef long long ll;
using namespace std;


__device__
struct BB{
    // Bottom left x
    int x;
    // Bottom left y
    int y;
    // width
    int w;
    // height
    int h;

  __device__
    bool rotate(int rot, int rows, int cols);
};

__device__
struct Point{
  float x;
  float y;
};

__device__
ll get_value(ll* arr, int i, int j, int rows, int cols);

__device__
void get_value(int* arr, int i, int j, int k, int rows, int cols, float *px);

__device__
ll get_prefix_sum(const BB& bb, int rows, int cols, ll* ps_mat);

__device__
void bilinear_interpolate(Point p, int ch, int rows, int cols, int* data, float data_px[]);

__device__ __host__
Point rotate_point(Point anchor, Point grid_pos, int rot, int query_rows);

__device__ __host__
void rotate_matrix(int rows, int cols, int rot, float* out);