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

__device__
ll get_value(ll* arr, int i, int j, int rows, int cols){
  if(i < 0 || i >= rows || j < 0 || j >= cols){
    return 0;
  }
  return arr[i*cols + j];
}
__device__
void get_value(int* arr, int i, int j, int k, int rows, int cols, float *px){
  if(i < 0 || i >= rows || j < 0 || j >= cols){
    return;
  }
  for (int ch = 0; ch < 3; ch ++) {
    px[ch] = arr[i * cols * 3 + j * 3 + ch];
  }
}

__device__
ll get_prefix_sum(const BB& bb, int rows, int cols, ll* ps_mat){
  ll ret = 0;
  ret += get_value(ps_mat, bb.y, bb.x + bb.w, rows, cols);
  ret -= get_value(ps_mat, bb.y, bb.x - 1, rows, cols);
  ret -= get_value(ps_mat, bb.y - bb.h - 1, bb.x + bb.w, rows, cols);
  ret += get_value(ps_mat, bb.y - bb.h - 1, bb.x - 1, rows, cols);
  return ret;
}

__device__
void bilinear_interpolate(Point p, int ch, int rows, int cols, int* data, float *data_px){
  
  int xl = floor(p.x);
  int yl = floor(p.y);
  int xh = ceil(p.x);
  int yh = ceil(p.y);
  float bl_val[3]{0, 0, 0}, br_val[3]{0, 0, 0}, tl_val[3]{0, 0, 0}, tr_val[3]{0, 0, 0};
  get_value(data, yl, xl, ch, rows, cols, bl_val);
  get_value(data, yl, xh, ch, rows, cols, br_val);
  get_value(data, yh, xl, ch, rows, cols, tl_val);
  get_value(data, yh, xh, ch, rows, cols, tr_val);
  float x_frac, y_frac, x_inv_frac, y_inv_frac;
  if(xl == xh) {
    if (yl == yh) {
      data_px[0] = bl_val[0];
      data_px[1] = bl_val[1];
      data_px[2] = bl_val[2];
      return;
    }
    y_frac = (p.y - yl)/(yh - yl);
    y_inv_frac = 1 - y_frac;
    data_px[0] = bl_val[0] * y_inv_frac + tl_val[0] * y_frac;
    data_px[1] = bl_val[1] * y_inv_frac + tl_val[1] * y_frac;
    data_px[2] = bl_val[2] * y_inv_frac + tl_val[2] * y_frac;
    return;
  }
  if(yl == yh) {
    x_frac = (p.x - xl)/(xh - xl);
    x_inv_frac = 1 - x_frac;
    data_px[0] = bl_val[0] * x_inv_frac + br_val[0] * x_frac;
    data_px[1] = bl_val[1] * x_inv_frac + br_val[1] * x_frac;
    data_px[2] = bl_val[2] * x_inv_frac + br_val[2] * x_frac;
    return;
  }
  x_frac = (p.x - xl)/(xh - xl);
  y_frac = (p.y - yl)/(yh - yl);
  
  x_inv_frac = 1 - x_frac;
  y_inv_frac = 1 - y_frac;
  data_px[0] = (bl_val[0] * x_inv_frac + br_val[0] * x_frac) * y_inv_frac + (tl_val[0] * x_inv_frac + tr_val[0] * x_frac) * y_frac;
  data_px[1] = (bl_val[1] * x_inv_frac + br_val[1] * x_frac) * y_inv_frac + (tl_val[1] * x_inv_frac + tr_val[1] * x_frac) * y_frac;
  data_px[2] = (bl_val[2] * x_inv_frac + br_val[2] * x_frac) * y_inv_frac + (tl_val[2] * x_inv_frac + tr_val[2] * x_frac) * y_frac;
  return;
}

__device__
bool BB::rotate(int rot, int rows, int cols){
  auto anchor = Point{(float)x,(float)y};
  auto p1 = rotate_point(anchor, Point{(float)w,(float) 0}, rot, h + 1);
  auto p2 = rotate_point(anchor, Point{(float)0,(float)0}, rot, h + 1);
  auto p3 = rotate_point(anchor, Point{(float)w,(float)h}, rot, h + 1);
  auto p4 = rotate_point(anchor, Point{(float)0, (float)h}, rot, h + 1);
  auto xl = min(p1.x, min(p2.x, min(p3.x, p4.x)));
  auto xr = max(p1.x, max(p2.x, max(p3.x, p4.x)));
  auto yt = min(p1.y, min(p2.y, min(p3.y, p4.y)));
  auto yb = max(p1.y, max(p2.y, max(p3.y, p4.y)));
  if (xl < 0 || xr >= cols || yb >= rows || yt < 0) {
    return false;
  }
  x = ceil(xl);
  y = floor(yb);
  w = floor(xr) - x;
  h = y - ceil(yt);
  return true;
}


void rotate_matrix(int rows, int cols, int rot, float* out){
  for(int i=0;i<rows;i++){
    for(int j=0;j<cols;j++){
      auto p = Point{(float)j,(float)i};
      auto p_rot = rotate_point(Point{(float)0, (float)0}, p, rot, rows);
      out[(i*cols + j)*2] = p_rot.x;
      out[(i*cols + j)*2 + 1] = p_rot.y;
    }
  }
}