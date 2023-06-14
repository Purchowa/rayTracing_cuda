#include <memory>

#include "Walnut/Application.h"
#include "Walnut/EntryPoint.h"

#include "Walnut/Image.h"
#include "Render.h"
#include <cstring>
#include <glm/gtc/type_ptr.hpp>

using namespace Walnut;

/**
* @class MainLayer
* @brief Prepares the main application window.
*/
class MainLayer : public Walnut::Layer {
public:
	MainLayer() {
		scene.material.reserve(5);
		scene.material.emplace_back(Material({ 0.4f, 0.4f, 0.4f}, 1.f, 0.f));
		scene.material.emplace_back(Material({ 0.3f, 0.4f, 0.f}, 0.6f, 6.f));
		scene.material.emplace_back(Material({ 0.4, 0.4, 0.4}, 0.0f, 0.f));

		scene.sphere.reserve(5);
		scene.sphere.emplace_back(Sphere({ 0.f, -100.5f, -1.f }, 100.f, 0)); // world
		scene.sphere.emplace_back(Sphere({ 0.f, 0.f, -1.f }, 0.5f, 1));
		scene.sphere.emplace_back(Sphere({ -1.2f, 0.4f, -1.f }, 0.6f, 2));
	}
	virtual void OnUpdate(float ts) override {
		camera.OnUpdate(ts);
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

		ImGui::Checkbox("Auto accumulate", &render.getSettingsRef().accumulate);

		ImGui::Text("Scene resolution: %d x %d", imageWidth, imageHeight);
		ImGui::End();

		// Scene
		ImGui::Begin("Objects");

		if (ImGui::Button("+")) {
			scene.sphere.emplace_back();
		}
		if (1 < scene.sphere.size()) {
			if (ImGui::Button("-")) {
				scene.sphere.pop_back();
			}
		}

		ImGui::Text("Environment");
		ImGui::ColorEdit3("Background color", glm::value_ptr(render.getSettingsRef().backgroundColor));
		ImGui::Separator();

		for (int i = 0; i < scene.sphere.size(); i++) {
			ImGui::PushID(i);
			if (i == 0)
				ImGui::Text("Ground");
			else
				ImGui::Text("Sphere %d", i);
			
			ImGui::DragFloat3("Position", glm::value_ptr(scene.sphere[i].getPositionRef()), 0.05f); // glm::value_ptr is same as &..getPositionRef.x
			ImGui::DragFloat("Radius", &scene.sphere[i].getRadiusRef(), 0.05f, 0.f, FLT_MAX);
			if ((scene.material.size() - 1) != 0) {
				ImGui::DragInt("Material id", &scene.sphere[i].getMaterialIdRef(), 1.f, 0, scene.material.size() - 1);
			}
			ImGui::Separator();
			ImGui::PopID();
		}
		ImGui::End();

		ImGui::Begin("Materials");
		if (ImGui::Button("+")) {
			scene.material.emplace_back();
		}
		if (1 < scene.material.size()) {
			if (ImGui::Button("-")) {
				for (auto& sphere : scene.sphere) {
					if (sphere.getMaterialIdx() == scene.material.size() - 1) {
						sphere.setMaterialIdx(0);
					}
				}
				scene.material.pop_back();
			}
		}

		for (int i = 0; i < scene.material.size(); i++) {
			ImGui::PushID(i);
			if (i == 0)
				ImGui::Text("Material ground");
			else
				ImGui::Text("Material %d", i);
			ImGui::ColorEdit3("Color", glm::value_ptr(scene.material[i].color));
			ImGui::DragFloat("Roughness", &scene.material[i].roughness, 0.01f, 0.f, 1.f);
			ImGui::DragFloat("Emission", &scene.material[i].emission, 0.01f, 0.f, FLT_MAX);
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
	Camera 	camera;
	uint32_t imageWidth = 0;
	uint32_t imageHeight = 0;
	bool realTimeRender = false;

	void renderImage() {
		render.onResize(imageWidth, imageHeight);
		camera.OnResize(imageWidth, imageHeight);
		render.render(scene, camera);
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