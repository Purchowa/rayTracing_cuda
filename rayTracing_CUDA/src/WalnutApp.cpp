#include "Walnut/Application.h"
#include "Walnut/EntryPoint.h"

#include "Walnut/Image.h"
#include "../../rayTracing_CUDA/Kernel.h"

Walnut::Image;

class ExampleLayer : public Walnut::Layer {
public:
	ExampleLayer(): kernel() {}

	virtual void OnUIRender() override {
		ImGui::Begin("Opcje");
		if (ImGui::Button("Run")) {
			kernel.run_kernel();
		}	
		ImGui::End();
	}

private:
	Kernel kernel;
};

Walnut::Application* Walnut::CreateApplication(int argc, char** argv) {
	Walnut::ApplicationSpecification spec;
	spec.Name = "rayTracing_CUDA";

	Walnut::Application* app = new Walnut::Application(spec);
	app->PushLayer<ExampleLayer>();
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