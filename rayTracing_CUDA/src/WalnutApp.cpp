#include <memory>

#include "Walnut/Application.h"
#include "Walnut/EntryPoint.h"

#include "Walnut/Image.h"
#include "../../rayTracing_CUDA/Kernel.h"


using namespace Walnut;

class MainLayer : public Walnut::Layer {
public:
	MainLayer(): kernel() {}

	virtual void OnUIRender() override {
		ImGui::Begin("Options");
		ImGui::Text("Last render: %.3fms", kernel.getKernelTimeMs());
		if (ImGui::Button("Render")) {
			render();
		}	
		ImGui::End();
		
		// Scene viewport
		ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2{ 0.f, 0.f });
		ImGui::Begin("Scene");

		imageWidth = ImGui::GetContentRegionAvail().x;
		imageHeight = ImGui::GetContentRegionAvail().y;

		if (sceneImage) {
			ImGui::Image(sceneImage->GetDescriptorSet(), { (float)sceneImage->GetWidth(), (float)sceneImage->GetHeight() });
		}	
		ImGui::End();
		ImGui::PopStyleVar();
	}

	~MainLayer() {
		delete[] imageBuffer;
	}

private:
	Kernel kernel;

	uint32_t imageWidth = 0, imageHeight = 0;
	uint32_t* imageBuffer = nullptr;
	std::unique_ptr<Image> sceneImage;
	

	void render() {
		if (sceneImage == nullptr || imageWidth != sceneImage->GetWidth() || imageHeight != sceneImage->GetHeight()) {
			sceneImage = std::make_unique<Image>(imageWidth, imageHeight, ImageFormat::RGBA);
			delete[] imageBuffer;
			imageBuffer = new uint32_t[imageWidth * imageHeight];
			kernel.setBufferSize(imageWidth * imageHeight);
			kernel.setBuffer(imageBuffer);
		}
		try {
			kernel.runKernel();
			sceneImage->SetData(imageBuffer);
		}
		catch (const std::invalid_argument& ex) {
			std::cerr << ex.what() << '\n';
		}
		
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