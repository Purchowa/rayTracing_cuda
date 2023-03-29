#pragma once
#include "Kernel.h"
#include "Walnut/Image.h"
#include <memory>

class Render {
public:
	void onResize(uint32_t nImgWidth, uint32_t nImgHeight); // gets called every frame
	void render();
	float getRednderTimeMs();
	std::shared_ptr<Walnut::Image> getFinalImage();

	~Render();
private:
	Kernel kernel;

	uint32_t imageWidth = 0, imageHeight = 0;
	uint32_t* imageBuffer = nullptr;
	std::shared_ptr<Walnut::Image> image;

	void resizeImageBuffer(uint32_t x, uint32_t y);
};