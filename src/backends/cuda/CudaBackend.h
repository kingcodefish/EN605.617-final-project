#pragma once

#include "ImageProcessor.h"

class BACKENDS_EXPORT CudaImageProcessor : public ImageProcessor
{
public:
	CudaImageProcessor() : ImageProcessor() {}
	~CudaImageProcessor() {}

	virtual void sobelFilter() override;

private:
};
