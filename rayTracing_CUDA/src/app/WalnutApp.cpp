#include <memory>

#include "Walnut/Application.h"
#include "Walnut/EntryPoint.h"

#include "Walnut/Image.h"
#include "Render.h"
#include <cstring>


using namespace Walnut;

class MainLayer : public Walnut::Layer {
public:
	MainLayer(): render(){}

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
		
		// Scene viewport
		ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2{ 0.f, 0.f });
		ImGui::Begin("Scene");

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
	uint32_t imageWidth = 0, imageHeight = 0;
	bool realTimeRender = false;

	void renderImage() {
		render.onResize(imageWidth, imageHeight);
		render.render();
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