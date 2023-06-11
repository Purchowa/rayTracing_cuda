#pragma once
#include <cuda_runtime.h>
#include <glm/glm.hpp>

/**
 * @class Camera
 * @brief Class representing a camera in 3D space
 */
class Camera {
public:
    /**
     * @brief Constructor that initializes the camera object with the given values
     * @param verticalFOV Vertical field of view angle (FOV) of the camera
     * @param nearClip Near clipping distance
     * @param farClip Far clipping distance
     */
    Camera(float verticalFOV, float nearClip, float farClip);

    /**
     * @brief Default constructor
     * Initializes the camera object with default values
     */
    Camera();

    /**
     * @brief Method called every frame, updates the camera state
     * @param ts Time since the last frame
     */
    void OnUpdate(float ts);

    /**
     * @brief Method called when the view size changes
     * @param width New view width
     * @param height New view height
     */
    void OnResize(uint32_t width, uint32_t height);

    /**
     * @brief Returns the projection matrix.
     * @return Constant reference to the projection matrix.
     */
    const glm::mat4& GetProjection() const { return m_Projection; }

    /**
     * @brief Returns the inverse projection matrix.
     * @return Constant reference to the inverse projection matrix.
     */
    const glm::mat4& GetInverseProjection() const { return m_InverseProjection; }

    /**
     * @brief Returns the view matrix.
     * @return Constant reference to the view matrix.
     */
    const glm::mat4& GetView() const { return m_View; }

    /**
     * @brief Returns the inverse view matrix.
     * @return Constant reference to the inverse view matrix.
     */
    const glm::mat4& GetInverseView() const { return m_InverseView; }

    /**
     * @brief Returns the direction.
     * @return Constant reference to the direction vector.
     */
    const glm::vec3& GetDirection() const { return m_ForwardDirection; }

    /**
     * @brief Returns the position.
     * @return Constant reference to the position vector.
     *
     * @note This function is available both on the device and on the host.
     */
    __device__ __host__ const glm::vec3& GetPosition() const {
        return m_Position;
    }

    /**
     * @brief Method for calculating the ray direction for the given coordinates
     * @param coord Screen coordinates
     * @return Ray direction in 3D space
     */
    __device__ glm::vec3 calculateRayDirection(const glm::vec2& coord) const;

	__device__ __host__ bool Moved() const { return moved; }
    /**
     * @brief Method for returning the camera rotation speed
     * @return Camera rotation speed
     */
    float GetRotationSpeed();

private:
    /**
     * @brief Method for calculating the camera projection matrix
     */
    void RecalculateProjection();

    /**
     * @brief Method for calculating the camera view matrix
     */
    void RecalculateView();

    glm::mat4 m_Projection{ 1.0f };        /**< Camera projection matrix */
    glm::mat4 m_View{ 1.0f };              /**< Camera view matrix */
    glm::mat4 m_InverseProjection{ 1.0f }; /**< Inverse camera projection matrix */
    glm::mat4 m_InverseView{ 1.0f };       /**< Inverse camera view matrix */

    float m_VerticalFOV = 45.0f; /**< Vertical FOV angle of the camera */
    float m_NearClip = 0.1f;     /**< Near clipping distance

   */
    float m_FarClip = 100.0f;    /**< Far clipping distance */

    glm::vec3 m_Position{ 0.0f, 0.0f, 0.0f };         /**< Camera position */
    glm::vec3 m_ForwardDirection{ 0.0f, 0.0f, 0.0f }; /**< Camera forward direction */

    glm::vec2 m_LastMousePosition{ 0.0f,
                                  0.0f }; /**< Last mouse cursor position */


	bool moved{ false };
    uint32_t m_ViewportWidth = 0;  /**< View width */
    uint32_t m_ViewportHeight = 0; /**< View height */
};