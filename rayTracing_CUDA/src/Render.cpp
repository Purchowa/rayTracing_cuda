#pragma once
#include "Render.h"

void Render::render() {
	try {
		kernel.runKernel();
		image->SetData(imageBuffer);
	}
	catch (const std::invalid_argument& ex) {
		std::cerr << ex.what() << '\n';
		std::exit(-1);
	}
}

void Render::onResize(uint32_t width, uint32_t height) {
	if (image) { // if exists
		if (image->GetWidth() != width || image->GetHeight() != height) {
			image->Resize(width, height);
			resizeImageBuffer(width, height);
			kernel.setBufferSize(width * height);
			kernel.setBuffer(imageBuffer);
		}
		return;
	}
	else {
		image = std::make_shared<Walnut::Image>(width, height, Walnut::ImageFormat::RGBA);
		resizeImageBuffer(width, height);
		kernel.setBufferSize(width * height);
		kernel.setBuffer(imageBuffer);
	}
}

float Render::getRednderTimeMs()
{
	return kernel.getKernelTimeMs();
}

std::shared_ptr<Walnut::Image> Render::getFinalImage()
{
	return image;
}

Render::~Render() {
	delete[] imageBuffer;
}

void Render::resizeImageBuffer(uint32_t x, uint32_t y)
{
	delete[] imageBuffer;
	imageBuffer = new uint32_t[x * (size_t)y];
}