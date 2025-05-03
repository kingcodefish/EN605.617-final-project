#pragma once

#include "backends.h"

class BACKENDS_EXPORT ImageProcessor
{
public:
	ImageProcessor() : m_texID(0), m_width(0), m_height(0) {}
	virtual ~ImageProcessor() {}

	// TODO: smart pointers
	void setImageData(unsigned int texID, int width, int height)
	{
		m_texID = texID;
		m_width = width;
		m_height = height;
	}
	unsigned int getImageData() { return m_texID; }

	virtual void sobelFilter() = 0;

protected:
	// Storage format is flattened RGBA
	unsigned int m_texID;

	int m_width;
	int m_height;
};
