#pragma once
void qx_compute_max_abs_gradient(unsigned char*texture, unsigned char*gradient_x, unsigned char*gradient_y, int h, int w);
void qx_rbf(float*out, float*in, unsigned char*gradient_x, unsigned char*gradient_y, float*temp, float*temp_2w, int h, int w, float sigma_spatial, float sigma_range);
void qx_zdepth_upsampling_using_rbf(float*out_hw, float*in_hw, unsigned char*RGB_hw3, float*temp_hw, float*temp_2w, unsigned char*gradient_x_hw, unsigned char*gradient_y_hw, float*ones, float*ones_temp, float*ones_temp_2w, int h, int w, float sigma_spatial = 0.03f, float sigma_range = 0.1f);