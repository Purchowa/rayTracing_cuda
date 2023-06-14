#pragma once
#include "Walnut/Image.h"
#include <memory>

#include "../scene/Scene.h"
#include "../scene/camera/Camera.h"
#include "../scene/kernel/Kernel.h"
#include "Settings.h"

/**
 * @class Render
 * @brief The Render class handles rendering a scene using a specified camera.
 */
class Render {
public:

  /**
   * @brief Default constructor for the Render class.
   */
  Render();

  /**
   * @brief Handles the resize event by updating the image size and buffer.
   *
   * @param width The new width of the image.
   * @param height The new height of the image.
   */
  void onResize(uint32_t nImgWidth, uint32_t nImgHeight);

  /**
   * @brief Renders the scene using the specified camera.
   *
   * @param scene The scene to render.
   * @param camera The camera to use for rendering.
   */
  void render(Scene &scene, Camera &camera);

  /**
   * @brief Retrieves the rendering time in milliseconds.
   *
   * @return The rendering time in milliseconds.
   */
  float getRednderTimeMs();

  /**
   * @brief Retrieves the final rendered image.
   *
   * @return A shared pointer to the final rendered image.
   */
  std::shared_ptr<Walnut::Image> getFinalImage();
  
  Settings &getSettingsRef() { return settings; };

  /**
   * @brief Destructor for the Render class.
   * Deletes the image buffer and accumulated color buffer.
   */
  ~Render();

private:

  Kernel kernel;
  uint32_t imageWidth = 0;
  uint32_t imageHeight = 0;
  uint32_t *imageBuffer = nullptr;
  glm::vec4 *accColor = nullptr;
  std::shared_ptr<Walnut::Image> image;

	Kernel kernel;
	uint32_t imageWidth = 0;
	uint32_t imageHeight = 0;
	uint32_t* imageBuffer = nullptr;
	glm::vec3* accColor = nullptr;
	std::shared_ptr<Walnut::Image> image;

  Settings settings;

  /**
   * @brief Reallocates the image buffer and accumulated color buffer with the
   * given size.
   *
   * @param x The width of the new image.
   * @param y The height of the new image.
   */
  void reallocateImageBuffer(uint32_t x, uint32_t y);
};
