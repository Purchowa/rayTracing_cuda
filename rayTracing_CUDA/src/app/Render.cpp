#pragma once
#include "Render.h"

Render::Render() {
	hittables = new Sphere({ 0.f, 0.f, 2.f }, 0.5f);
}

void Render::render() {
	try {
		kernel.runKernel(hittables);
		image->SetData(imageBuffer);
	}
	catch (const std::invalid_argument& ex) {
		std::cerr << ex.what() << '\n';
		std::exit(-1);
	}
}

void Render::onResize(uint32_t width, uint32_t height) {
	if (image) { // if exists
		if (image->GetWidth() == width && image->GetHeight() == height) {
			return;
		}
		image->Resize(width, height);
	}
	else {
		image = std::make_shared<Walnut::Image>(width, height, Walnut::ImageFormat::RGBA);
	}
	reallocateImageBuffer(width, height);
	kernel.setImgDim({ width, height });
	kernel.setBuffer(imageBuffer);
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
	delete hittables;

	imageBuffer = nullptr;
	hittables = nullptr;
}

void Render::reallocateImageBuffer(uint32_t x, uint32_t y)
{
	delete[] imageBuffer;
	imageBuffer = new uint32_t[x * (size_t)y];
}