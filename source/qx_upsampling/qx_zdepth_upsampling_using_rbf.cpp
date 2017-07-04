#include <iostream>
using namespace std;
#include <vector>
#include <cmath>

#include "string.h"
#include "stdio.h"

//#include "qx_basic.h"
#include "qx_zdepth_upsampling_using_rbf.h"
inline unsigned char qx_max_u3(const unsigned char r, const unsigned char g, const unsigned char b) { int ret = r; if (ret<b) ret = b; if (ret<g) ret = g; return ret; }

inline unsigned char qx_compute_RGB_distance_max(const unsigned char*image, const unsigned int idt, const unsigned int id)
{
	unsigned int id3 = id * 3;
	unsigned int idt3 = idt * 3;
	unsigned char r = abs((int)image[idt3++] - (int)image[id3++]);
	unsigned char g = abs((int)image[idt3++] - (int)image[id3++]);
	unsigned char b = abs((int)image[idt3++] - (int)image[id3++]);
	return qx_max_u3(r, g, b);
}

void qx_compute_max_abs_gradient(unsigned char*texture, unsigned char*gradient_x, unsigned char*gradient_y, int h, int w)
{
	int y, x;
	unsigned char I[24], Ix[24], Iy[24], Gx[8], Gy[8];
	for (y = 0; y<h - 1; y++)
	{
		int yn = y + 1;
		for (x = 0; x<w - 1; x++)
		{
			int xn = x + 1;
			int id = (y*w + x);
			gradient_x[id] = qx_compute_RGB_distance_max(texture, (y*w + xn), id);
			gradient_y[id] = qx_compute_RGB_distance_max(texture, (yn*w + x), id);
		}
		int id = (y*w + w - 1);
		gradient_x[id] = gradient_x[id - 1];
		gradient_y[id] = qx_compute_RGB_distance_max(texture, (yn*w + w - 1), id);
	}
	memcpy(&gradient_y[(h - 1)*w], &gradient_y[(h - 2)*w], sizeof(char)*w);
	for (x = 0; x<w - 1; x++)
	{
		int xn = x + 1;
		int id = (y*w + x);
		gradient_x[id] = qx_compute_RGB_distance_max(texture, (y*w + xn), id);
	}
	int id = (y*w + w - 1);
	gradient_x[id] = gradient_x[id - 1];
}
void qx_rbf(float*out, float*in, unsigned char*gradient_x, unsigned char*gradient_y, float*temp, float*temp_2w, int h, int w, float sigma_spatial, float sigma_range)
{
    float alpha = (float)exp(-sqrt(2.0) / (sigma_spatial*w));
	sigma_range *= 255;
	float range_table[256];
	for (int i = 0; i<256; i++) range_table[i] = (float)exp(-(double)i / sigma_range);
	float*in_ = in;/*horizontal filtering*/
	float*out_ = temp;
	for (int y = 0; y<h; y++)
	{
		int yw = y*w;
		out_[yw] = in_[yw];
		float yc, yp = in_[yw];
		for (int x = 1; x<w; x++)
		{
			int id = yw + x;
			float weight = range_table[gradient_x[id-1]];
			float alpha_ = weight*alpha;
			out_[id] = yc = (1 - alpha)*in_[id] + alpha_*yp;
			yp = yc;
		}
		int w1 = w - 1;
		int yw1 = y*w1;
		yp = in_[yw1];
		out_[yw1] = 0.5*(out_[yw1] + yp);
		for (int x = w - 2; x >= 0; x--)
		{
			int id = yw + x;
			float weight = range_table[gradient_x[id]];
			float alpha_ = weight*alpha;
			yc = (1 - alpha)*in_[id] + alpha_*yp;
			out_[id] = 0.5*(temp[id] + yc);
			yp = yc;
		}
	}

	alpha = (float)exp(-sqrt(2.0) / (sigma_spatial*h));
	in_ = temp;/*vertical filtering*/
	out_ = out;
	float*ycy, *ypy, *xcy, *xpy;
	memcpy(out_, in_, sizeof(float)*w);
	for (int y = 1; y<h; y++)
	{
		int yw = y*w;
		xcy = &in_[yw];
		ypy = &out_[yw - w];
		ycy = &out_[yw];
		for (int x = 0; x<w; x++)
		{
			int id = yw + x;
			float weight = range_table[gradient_y[id-w]];
			float alpha_ = weight*alpha;
			ycy[x] = (1 - alpha)*xcy[x] + alpha_*ypy[x];
		}
	}
	int h1 = h - 1;
	int h1w = h1*w;
	ycy = temp_2w;
	ypy = &temp_2w[w];
	memcpy(ypy, &in_[h1w], sizeof(float)*w);
	for (int x = 0; x<w; x++) out_[h1w+x] = 0.5*(out[h1w+x] + ypy[x]);
	for (int y = h1 - 1; y >= 0; y--)
	{
		int yw = y*w;
		xcy = &in_[yw];
		for (int x = 0; x<w; x++)
		{
			int id = yw + x;
			float weight = range_table[gradient_y[id]];
			float alpha_ = weight*alpha;
			ycy[x] = (1 - alpha)*xcy[x] + alpha_*ypy[x];
			out_[id] = 0.5*(out[id] + ycy[x]);
		}
		memcpy(ypy, ycy, sizeof(float)*w);
	}
}
//guidance filter implemented by realtime bilteral filter
void qx_zdepth_upsampling_using_rbf(float*out_hw, float*in_hw, unsigned char*RGB_hw3, float*temp_hw, float*temp_2w, unsigned char*gradient_x_hw, unsigned char*gradient_y_hw, float*ones, float*ones_temp, float*ones_temp_2w, int h, int w, float sigma_spatial, float sigma_range)
{
	unsigned char*guidance = RGB_hw3, *gradient_x = gradient_x_hw, *gradient_y = gradient_y_hw;
	float*out = out_hw, *in = in_hw, *temp = temp_hw;
	qx_compute_max_abs_gradient(guidance, gradient_x, gradient_y, h, w); //compute gradient


	for (int i=0;i<h*w;i++)
	{
		if (in[i] > 0.000001) ones[i] = 1.f;
		else ones[i] = 0.f;
	}
	//image_display(guidance, h, w, 3);
	//image_display(in, h, w,1);
	memcpy(out, in, sizeof(float)*h*w);
	for (int i = 0; i < 1; i++)
    {
		qx_rbf(out, out, gradient_x, gradient_y, temp, temp_2w, h, w, sigma_spatial, sigma_range);
		qx_rbf(ones, ones, gradient_x, gradient_y, ones_temp, ones_temp_2w, h, w, sigma_spatial, sigma_range);
		sigma_spatial *= 0.5f;
		sigma_range *= 0.5f;
	}
	for (int i = 0; i < h*w; i++)
	{
		if(ones[i]>0.00001f&&in[i]<0.5f) out[i]= out[i]/ones[i];
		else out[i] = in[i];
	}
	//image_display(out, h, w,1);
}
