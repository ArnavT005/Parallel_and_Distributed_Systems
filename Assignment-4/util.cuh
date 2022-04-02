#pragma once

#include <math.h>
#include "img.hpp"

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
    void rotate(int rot);
  __device__
    BB intersect(const BB& other) const;
};

__device__
struct Point{
  float x;
  float y;
};

__device__
int get_value(int* arr, int i, int j, int k, int rows, int cols);

__device__
float get_prefix_sum(const BB& bb, int rows, int cols, float* ps_mat);

__device__
float bilinear_interpolate(Point p, int ch, int rows, int cols, int* data);

__device__
Point rotate_point(Point anchor, Point grid_pos, int rot, int query_rows);