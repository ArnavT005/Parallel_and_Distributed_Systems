#include "util.cuh"

__device__
Point rotate_point(Point anchor, Point grid_pos, int rot, int query_rows){
  if (rot == 0){
    return Point{anchor.x + grid_pos.x, anchor.y - (query_rows - 1) + grid_pos.y};
  }
  float angle = (float)rot * M_PI/180;
  float x = (float)grid_pos.x * cos(angle) - ((float)(query_rows - 1 - grid_pos.y) * sin(angle));
  float y = -((float)grid_pos.x * sin(angle)) - ((float)(query_rows - 1 - grid_pos.y) * cos(angle));
  return Point{anchor.x + x, anchor.y + y};
}

// rotate the bounding box by 45 degrees
__device__
void BB::rotate(int rot){
  auto anchor = Point{(float)x,(float)y};
  auto p1 = rotate_point(anchor, Point{(float)w,(float) 0}, rot, h + 1);
  auto p2 = rotate_point(anchor, Point{(float)0,(float)0}, rot, h + 1);
  auto p3 = rotate_point(anchor, Point{(float)w,(float)h}, rot, h + 1);
  auto p4 = rotate_point(anchor, Point{(float)0, (float)h}, rot, h + 1);
  auto xl = min(p1.x, min(p2.x, min(p3.x, p4.x)));
  auto xr = max(p1.x, max(p2.x, max(p3.x, p4.x)));
  auto yt = max(p1.y, max(p2.y, max(p3.y, p4.y)));
  auto yb = min(p1.y, min(p2.y, min(p3.y, p4.y)));
  x = ceil(xl);
  y = floor(yt);
  w = floor(xr) - x;
  h = y - ceil(yb);
}

__device__
BB BB::intersect(const BB& other) const {
  BB ret;
  ret.x = max(x, other.x);
  ret.y = min(y, other.y);
  ret.w = min(x+w, other.x+other.w) - ret.x;
  ret.h = ret.y - max(y - h, other.y - other.h);
  return ret;
}

__device__
int get_value(float* arr, int i, int j, int rows, int cols){
  if(i < 0 || i >= rows || j < 0 || j >= cols){
    return 0;
  }
  return arr[i*cols + j];
}
__device__
int get_value(int* arr, int i, int j, int k, int rows, int cols){
  if(i < 0 || i >= rows || j < 0 || j >= cols){
    return 0;
  }
  return arr[i * cols * 3 + j * 3 + k];
}

__device__
float get_prefix_sum(const BB& bb, int rows, int cols,  float* ps_mat){
  auto mat_bb = BB{0, rows - 1, cols-1, rows-1};
  auto intersect_bb = bb.intersect(mat_bb);
  float ret = 0;
  ret += get_value(ps_mat, intersect_bb.y, intersect_bb.x+intersect_bb.w, rows, cols);
  ret -= get_value(ps_mat, intersect_bb.y, intersect_bb.x - 1, rows, cols);
  ret -= get_value(ps_mat, intersect_bb.y-intersect_bb.h - 1, intersect_bb.x + intersect_bb.w, rows, cols);
  ret += get_value(ps_mat, intersect_bb.y - intersect_bb.h - 1, intersect_bb.x - 1, rows, cols);
  return ret;
}


__device__
float bilinear_interpolate(Point p, int ch, int rows, int cols, int* data){
  
  int xl = floor(p.x);
  int yl = floor(p.y);
  int xh = ceil(p.x);
  int yh = ceil(p.y);
  auto bl_val = get_value(data, yl, xl, ch, rows, cols);
  auto br_val = get_value(data, yl, xh, ch, rows, cols);
  auto tl_val = get_value(data, yh, xl, ch, rows, cols);
  auto tr_val = get_value(data, yh, xh, ch, rows, cols);
  float x_frac, y_frac, x_inv_frac, y_inv_frac;
  if(xl == xh) {
    if (yl == yh) {
      return bl_val;
    }
    y_frac = (p.y - yl)/(yh - yl);
    y_inv_frac = 1 - y_frac;
    return (bl_val * y_inv_frac + tl_val * y_frac);
  }
  if(yl == yh) {
    x_frac = (p.x - xl)/(xh - xl);
    x_inv_frac = 1 - x_frac;
    return (bl_val * x_inv_frac + br_val * x_frac);
  }
  x_frac = (p.x - xl)/(xh - xl);
  y_frac = (p.y - yl)/(yh - yl);
  
  x_inv_frac = 1 - x_frac;
  y_inv_frac = 1 - y_frac;
  
  float ret = (bl_val * x_inv_frac + br_val * x_frac) * y_inv_frac + (tl_val * x_inv_frac + tr_val * x_frac) * y_frac;
  return ret;
}

