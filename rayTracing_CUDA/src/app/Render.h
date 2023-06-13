#pragma once
#include <memory>
#include "Walnut/Image.h"

#include "../scene/kernel/Kernel.h"
#include "../scene/Scene.h"
#include "../scene/camera/Camera.h"
#include "Settings.h"

class Render {
public:
	Render();
	void onResize(uint32_t nImgWidth, uint32_t nImgHeight);
	void render(Scene& scene, Camera& camera);
	float getRednderTimeMs();
	std::shared_ptr<Walnut::Image> getFinalImage();
	Settings& getSettingsRef() { return settings; };
	~Render();
private:
	Kernel kernel;
	uint32_t imageWidth = 0;
	uint32_t imageHeight = 0;
	uint32_t* imageBuffer = nullptr;
	glm::vec3* accColor = nullptr;
	std::shared_ptr<Walnut::Image> image;

	Settings settings;

	void reallocateImageBuffer(uint32_t x, uint32_t y);

};

