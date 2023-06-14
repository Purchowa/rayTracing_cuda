#pragma once
#include "Render.h"

Render::Render() {}

void Render::render(Scene &scene, Camera &camera) {
  try {
    kernel.runKernel(scene, camera, settings);
    image->SetData(imageBuffer);
  } catch (const std::invalid_argument &ex) {
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
  } else {
    image = std::make_shared<Walnut::Image>(width, height,
                                            Walnut::ImageFormat::RGBA);
  }
  reallocateImageBuffer(width, height);
  kernel.setImgDim({width, height});
  kernel.setBuffer(imageBuffer, accColor);
}

float Render::getRednderTimeMs() { return kernel.getKernelTimeMs(); }

std::shared_ptr<Walnut::Image> Render::getFinalImage() { return image; }

Render::~Render() {
  delete[] imageBuffer;
  delete[] accColor;

  imageBuffer = nullptr;
  accColor = nullptr;
}


void Render::reallocateImageBuffer(uint32_t x, uint32_t y)
{
	delete[] imageBuffer;
	delete[] accColor;
	imageBuffer = new uint32_t[x * (size_t)y];
	accColor = new glm::vec3[x * (size_t)y];
	std::memset(accColor, 0, sizeof(*accColor) * (x * (size_t)y));
	std::memset(imageBuffer, 0, sizeof(*imageBuffer) * (x * (size_t)y));

}

