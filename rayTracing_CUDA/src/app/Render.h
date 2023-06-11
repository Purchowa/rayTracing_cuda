#pragma once
#include "../scene/kernel/Kernel.h"
#include "../scene/Scene.h"
#include "Walnut/Image.h"
#include "../scene/camera/Camera.h"
#include <memory>

struct Settings {
	bool accumulate{ false };
};

class Render {
public:
	Render();
	void onResize(uint32_t nImgWidth, uint32_t nImgHeight);
	void render(Scene& scene, Camera& camera);
	float getRednderTimeMs();
	std::shared_ptr<Walnut::Image> getFinalImage();
	~Render();
private:
	

	Kernel kernel;
	uint32_t imageWidth = 0;
	uint32_t imageHeight = 0;
	uint32_t* imageBuffer = nullptr;
	glm::vec4* accColor = nullptr;
	std::shared_ptr<Walnut::Image> image;

	Settings settings;

	void reallocateImageBuffer(uint32_t x, uint32_t y);

};

