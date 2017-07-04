#pragma once
#include <iostream>
using namespace std;
#include <vector>

#include "stdlib.h"

//#include <imdebug.h>
//#pragma comment(lib,"imdebug.lib")
//template <typename T>inline void image_display(T*image, int h, int w, int nr_channel)
//{
//	if (typeid(image[0]).name() == typeid(unsigned char).name())
//	{
//		switch (nr_channel)
//		{
//		case 1:
//			imdebug("lum *auto w=%d h=%d %p", w, h, image);
//			break;
//		case 3:
//			imdebug("rgb *auto w=%d h=%d %p", w, h, image);
//			break;
//		case 4:
//			imdebug("rgba *auto w=%d h=%d %p", w, h, image);
//			break;
//		default:
//			vector<unsigned char>temp;
//			temp.resize(h*w * 4);
//			for (int i = 0; i<(nr_channel >> 2); i++)
//			{
//				for (int y = 0; y<h; y++) for (int x = 0; x<w; x++) for (int c = 0; c<4; c++)
//					temp[y*w * 4 + x * 4 + c] = image[y*w * 4 + x * 4 + c + (i << 2)];
//				imdebug("rgba *auto w=%d h=%d %p", w, h, &temp[0]);
//			}
//		}
//	}
//	else if (typeid(image[0]).name() == typeid(float).name())
//	{
//		switch (nr_channel)
//		{
//		case 1:
//			imdebug("lum *auto b=32f w=%d h=%d %p", w, h, image);
//			break;
//		case 3:
//			imdebug("rgb *auto b=32f w=%d h=%d %p", w, h, image);
//			break;
//		case 4:
//			imdebug("rgba *auto b=32f w=%d h=%d %p", w, h, image);
//			break;
//		default:
//			vector<float>temp;
//			temp.resize(h*w * 4);
//			for (int i = 0; i<(nr_channel >> 2); i++)
//			{
//				for (int y = 0; y<h; y++) for (int x = 0; x<w; x++) for (int c = 0; c<4; c++)
//					temp[y*w * 4 + x * 4 + c] = image[y*w * 4 + x * 4 + c + (i << 2)];
//				imdebug("rgba *auto b=32f w=%d h=%d %p", w, h, &temp[0]);
//			}
//		}
//	}
//}

#include <iostream>
using namespace std;
#include <vector>
#include <fstream>

class qx_ppm
{
public:
	void load(char*image_filename);
	void save(char*image_filename);
	int get_w() { return m_width; }
	int get_h() { return m_height; }
	unsigned char*get_data() { return &m_image[0]; }

    //add
    void set_data(vector<unsigned char>& in)
    {
        m_image.clear();
        m_image.resize(in.size());
        copy(in.begin(), in.end(), m_image.begin());
    }

private:
	int m_type;
	int m_number_of_channel;
	//m_number_of_channel=1: the input image is a grayscale image
	//m_number_of_channel=3: the input image is a color image
	//see line ???? for details
	int m_width;//image width
	int m_height;//image height
	vector<unsigned char>m_image;//input image
};
inline void separate_string_into_two_nums(int*num1, int*num2, char*str)
{
	int id_of_spacekey;
	for (int i = 0; str[i] != NULL; i++)
	{
		if (str[i] == ' ')
		{
			id_of_spacekey = i;
			break;
		}
	}
	*num2 = atoi(&str[id_of_spacekey]);
	str[id_of_spacekey] = NULL;
	*num1 = atoi(str);
}


void qx_ppm::load(char* image_filename)
{
	ifstream fin;
	char line[256];
	fin.open(image_filename, ios::binary);
	if (fin.fail())
	{
		cout << "Please check input filename: " << image_filename << endl;
		exit(1);
	}
	fin.getline(line, 256);//get first line

	if (line[0] != 'p'&&line[0] != 'P')//check invalid header
	{
		cout << "Bad header in image file." << endl;
		exit(-1);
	}
	m_type = atoi(&(line[1]));
	switch (m_type)
	{
	case 2://PGM image in ASCII mode
	case 5://PGM image in Binary mode
		m_number_of_channel = 1;
		break;
	case 3://PPM image in ASCII mode
	case 6://PPM image in Binary mode
		m_number_of_channel = 3;
		break;
	default:
		cout << "Bad header in image file." << endl;
		exit(-1);
		break;
	}

	fin.getline(line, 256);//get 2nd line
	while (line[0] == '#')//omit comments
	{
		fin.getline(line, 256);
	}
	separate_string_into_two_nums(&m_width, &m_height, line);//get the width and height from char line[256]
	m_image.resize(m_height*m_width*m_number_of_channel);//allocate memory to store the image
	fin.getline(line, 256);//get the maximum value
						   //m_maxval=atoi(line);
	int pixel_value;
	switch (m_type)
	{
	case 2://read PGM image in ASCII mode
		for (int y = 0; y<m_height; y++)
		{
			for (int x = 0; x<m_width; x++)
			{
				fin >> pixel_value;
				int id = y*m_width + x;
				m_image[id] = pixel_value;
			}
		}
		break;
	case 5://read PGM image in binary mode
		fin.read((char*)(&m_image.front()), sizeof(unsigned char)*m_image.size());
		break;
	case 3:/*Please complete this case: read PPM in ASCII mode*/
		for (int y = 0; y<m_height; y++)
		{
			for (int x = 0; x<m_width; x++)
			{
				for (int c = 0; c<m_number_of_channel; c++)
				{
					fin >> pixel_value;
					int id = (y*m_width + x)*m_number_of_channel + c;
					m_image[id] = pixel_value;
				}
			}
		}
		break;
	case 6:/*Please complete this case: read PPM in binary mode*/
		fin.read((char*)(&m_image.front()), sizeof(unsigned char)*m_image.size());
		break;
	default:
		cout << "Can not open image [" << image_filename << "]!!" << endl;
		break;
	}
	fin.close();
}


void qx_ppm::save(char*image_filename)
{
	unsigned char max_value = 0;
	for (int y = 0; y<m_height; y++)
	{
		for (int x = 0; x<m_width; x++)
		{
			for (int c = 0; c<m_number_of_channel; c++)
			{
				int id = (y*m_width + x)*m_number_of_channel + c;
				unsigned char current_value = m_image[id];
				if (max_value<current_value)
					max_value = current_value;
			}
		}
	}
	ofstream fout;
	fout.open(image_filename, ios::binary);
	if (fout.fail())
	{
		cout << "Can't open the file " << image_filename << ", exit!" << endl;
		cin.get();
		exit(1);
	}
	fout << "P" << m_type << endl << m_width << " " << m_height << endl << (int)max_value << endl;
	switch (m_type)
	{
	case 2://write PGM image in ASCII mode
		for (int y = 0; y<m_height; y++)
		{
			for (int x = 0; x<m_width; x++)
			{
				int id = y*m_width + x;
				fout << (int)m_image[id] << " ";
			}
		}
		break;
	case 3:/*Please complete this case: write PPM in ASCII mode*/
		for (int y = 0; y<m_height; y++)
		{
			for (int x = 0; x<m_width; x++)
			{
				for (int c = 0; c<m_number_of_channel; c++)
				{
					int id = (y*m_width + x)*m_number_of_channel + c;
					fout << (int)m_image[id] << " ";
				}
			}
		}
		break;
	case 5://write PGM image in binary mode
	case 6:/*Please complete this case: write PPM in binary mode*/
		fout.write((char*)(&m_image.front()), sizeof(unsigned char)*m_image.size());
		break;
	default:
		cout << "Cannot save image [" << image_filename << "]!!" << endl;
		break;
	}
	fout.close();
}
