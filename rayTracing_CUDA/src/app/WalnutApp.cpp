#include <memory>

#include "Walnut/Application.h"
#include "Walnut/EntryPoint.h"

#include "Walnut/Image.h"
#include "Render.h"
#include <cstring>
#include <glm/gtc/type_ptr.hpp>
#include "../camera/Camera.h"

using namespace Walnut;

class MainLayer : public Walnut::Layer {
public:
	MainLayer(): render() {
		scene.sphere.reserve(5);

	}
	virtual void OnUpdate(float ts) override {
		m_camera.OnUpdate(ts);
	}

	virtual void OnUIRender() override {
		ImGui::Begin("Options");
		ImGui::Text("Last render: %.3fms", render.getRednderTimeMs());
		if (ImGui::Button("Render")) {
			renderImage();
		}
		ImGui::Text("Real time render");
		if (ImGui::Button("on")) {
			realTimeRender = true;
		}
		if (ImGui::Button("off")) {
			realTimeRender = false;
		}
		ImGui::Text("Scene resolution: %d x %d", imageWidth, imageHeight);
		ImGui::End();

		// Scene
		ImGui::Begin("Scene");
		if (ImGui::Button("+")) {
			scene.sphere.emplace_back();
		}
		if (!scene.sphere.empty()) {
			if (ImGui::Button("-")) {
				scene.sphere.pop_back();
			}
		}
		for (int i = 0; i < scene.sphere.size(); i++) {
			ImGui::PushID(i);
			ImGui::Text("Sphere %d", i+1);
			ImGui::DragFloat3("Postition", glm::value_ptr(scene.sphere[i].getPositionRef()), 0.05f); // glm::value_ptr is same as &..getPositionRef.x
			ImGui::DragFloat("Radius", &scene.sphere[i].getRadiusRef(), 0.01f);
			ImGui::ColorEdit4("Color", glm::value_ptr(scene.sphere[i].getColorRef()));
			ImGui::Separator();
			ImGui::PopID();
		}
		ImGui::End();

		// Viewport
		ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2{ 0.f, 0.f });
		ImGui::Begin("View");

		imageWidth = ImGui::GetContentRegionAvail().x;
		imageHeight = ImGui::GetContentRegionAvail().y;

		auto finalImage = render.getFinalImage();
		if (finalImage){
			ImGui::Image(finalImage->GetDescriptorSet(), { (float)finalImage->GetWidth(), (float)finalImage->GetHeight() });
		}	
		ImGui::End();
		ImGui::PopStyleVar();
		if (realTimeRender) {
			renderImage();
		}
	}

private:
	Render render;
	Scene scene;
	Camera 	m_camera;
	uint32_t imageWidth = 0;
	uint32_t imageHeight = 0;
	bool realTimeRender = false;

	void renderImage() {
		render.onResize(imageWidth, imageHeight);
		m_camera.OnResize(imageWidth, imageHeight);
		render.render(scene, m_camera);
;
	}
};

Walnut::Application* Walnut::CreateApplication(int argc, char** argv) {
	Walnut::ApplicationSpecification spec;
	spec.Name = "rayTracing_CUDA";

	Walnut::Application* app = new Walnut::Application(spec);
	app->PushLayer<MainLayer>();
	app->SetMenubarCallback([app]() {
		if (ImGui::BeginMenu("File")) {
			if (ImGui::MenuItem("Exit")) {
				app->Close();
			}
			ImGui::EndMenu();
		}
	});
	return app;
}