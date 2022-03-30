#include "util.cuh"

__device__
Point rotate_point(Point anchor, Point grid_pos, int rot){
  if (rot == 0){
    return Point{grid_pos.x + anchor.x, grid_pos.y + anchor.y};
  }
  float angle = (float)rot * M_PI/180;
  float x = (float)grid_pos.x * cos(angle) - (float)grid_pos.y * sin(angle);
  float y = (float)grid_pos.x * sin(angle) + (float)grid_pos.y * cos(angle);
  return Point{x + anchor.x, y + anchor.y};
}

// rotate the bounding box by 45 degrees
__device__
void BB::rotate(int rot){
  //  switch(rot){
  //   case 45:
  //     float xr = x + (float)w/sqrt(2);
  //     float xl = x - (float)h/sqrt(2);
  //     float ht = y + (float)h/sqrt(2) + (float)w/sqrt(2);
  //     x = ceil(xl);
  //     y = y;
  //     w = (int)xr - x;
  //     h = (int)ht - y;

  //     break;
  //   case -45:
  //     float xr = x + (float)w/sqrt(2);
  //     float xl = x - (float)h/sqrt(2);
  //     float ht = y - (float)h/sqrt(2) - (float)w/sqrt(2);
  //     x = ceil(xl);
  //     y = y;
  //     w = (int)xr - x;
  //     h = ceil(ht) - y;
  //     break;
  //   default:
  //   break;
  // }
  // float angle = rot*M_PI/180;
  auto anchor = Point{(float)x,(float)y};
  auto p1 = rotate_point(anchor, Point{(float)x+w,(float)y}, rot);
  auto p2 = rotate_point(anchor, Point{(float)x+w,(float)y+h}, rot);
  auto p3 = rotate_point(anchor, Point{(float)x,(float)y+h}, rot);
  auto xl = min(p1.x, min(p2.x, p3.x));
  auto xr = max(p1.x, max(p2.x, p3.x));
  auto yt = max(p1.y, max(p2.y, p3.y));
  auto yb = min(p1.y, min(p2.y, p3.y));
  x = ceil(xl);
  y = ceil(yb);
  w = (int)xr - x;
  h = (int)yt - y;

}
__device__
int get_value(int* arr, int i, int j, int rows, int cols){
  if(i < 0 || i >= rows || j < 0 || j >= cols){
    return 255;
  }
  return arr[i*cols + j];
}

__device__
BB BB::intersect(const BB& other) const {
  BB ret;
  ret.x = max(x, other.x);
  ret.y = max(y, other.y);
  ret.w = min(x+w, other.x+other.w) - ret.x;
  ret.h = min(y+h, other.y+other.h) - ret.y;
  return ret;
}

__device__
float get_prefix_sum(const BB& bb, int rows, int cols,  int* ps_mat){
  auto mat_bb = BB{0,0,cols-1,rows-1};
  auto intersect_bb = bb.intersect(mat_bb);
  float ret = 0;
  ret += get_value(ps_mat, intersect_bb.y, intersect_bb.x, rows, cols);
  ret += get_value(ps_mat, intersect_bb.y+intersect_bb.h, intersect_bb.x+intersect_bb.w, rows, cols);
  ret -= get_value(ps_mat, intersect_bb.y+intersect_bb.h, intersect_bb.x, rows, cols);
  ret -= get_value(ps_mat, intersect_bb.y, intersect_bb.x+intersect_bb.w, rows, cols);
  auto remaining_pixels = bb.w*bb.h - intersect_bb.w * intersect_bb.h;
  ret += remaining_pixels*255;
  return ret;

}


__device__
float bilinear_interpolate(Point p, int rows, int cols, int* data){
  
  int xl = floor(p.x);
  int yl = floor(p.y);
  int xh = ceil(p.x);
  int yh = ceil(p.y);
  float x_frac = (p.x - xl)/(xh - xl);
  float y_frac = (p.y - yl)/(yh - yl);
  float x_inv_frac = 1 - x_frac;
  float y_inv_frac = 1 - y_frac;
  auto bl_val = get_value(data, yl, xl, rows, cols);
  auto br_val = get_value(data, yl, xh, rows, cols);
  auto tl_val = get_value(data, yh, xl, rows, cols);
  auto tr_val = get_value(data, yh, xh, rows, cols);
  float ret = (bl_val * x_inv_frac + br_val * x_frac) * y_inv_frac + (tl_val * x_inv_frac + tr_val * x_frac) * y_frac;
  return ret;
}

