#include <GLFW/glfw3.h>
#include <imgui/imgui.h>
#include <stdio.h>

static void glfw_error_callback(int error, const char* description)
{
    fprintf(stderr, "GLFW Error %d: %s\n", error, description);
}

// Main code
int main(int, char**)
{
    glfwSetErrorCallback(glfw_error_callback);
    if (!glfwInit())
        return 1;

    // Our state
    bool show_demo_window = true;
    bool show_another_window = false;
    ImVec4 clear_color = ImGui::GetStyle().Colors[ImGuiCol_WindowBg];

    // Main loop
    while (!glfwWindowShouldClose(window))
    {
        // Poll and handle events (inputs, window resize, etc.)
        // You can read the io.WantCaptureMouse, io.WantCaptureKeyboard flags to tell if dear imgui wants to use your inputs.
        // - When io.WantCaptureMouse is true, do not dispatch mouse input data to your main application, or clear/overwrite your copy of the mouse data.
        // - When io.WantCaptureKeyboard is true, do not dispatch keyboard input data to your main application, or clear/overwrite your copy of the keyboard data.
        // Generally you may always pass all inputs to dear imgui, and hide them from your application based on those two flags.
        glfwPollEvents();
        if (glfwGetWindowAttrib(window, GLFW_ICONIFIED) != 0)
        {
            ImGui_ImplGlfw_Sleep(10);
            continue;
        }

        // Start the Dear ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        static ImGuiWindowFlags flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_MenuBar | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings;

        const ImGuiViewport* viewport = ImGui::GetMainViewport();
        ImGui::SetNextWindowPos(viewport->Pos);
        ImGui::SetNextWindowSize(viewport->Size);
        if (ImGui::Begin("Main Window Layout", 0, flags))
        {

            if (ImGui::BeginMenuBar())
            {
                if (ImGui::BeginMenu("File"))
                {
                    if (ImGui::MenuItem("Open Recording")) {}
                    if (ImGui::MenuItem("Save Recording")) {}
                    if (ImGui::MenuItem("Exit")) {}
                    ImGui::EndMenu();
                }
                if (ImGui::BeginMenu("Edit"))
                {
                    if (ImGui::MenuItem("Undo", "CTRL+Z")) {}
                    if (ImGui::MenuItem("Redo", "CTRL+Y", false, false)) {} // Disabled item
                    ImGui::Separator();
                    if (ImGui::MenuItem("Cut", "CTRL+X")) {}
                    if (ImGui::MenuItem("Copy", "CTRL+C")) {}
                    if (ImGui::MenuItem("Paste", "CTRL+V")) {}
                    ImGui::EndMenu();
                }
                ImGui::EndMenuBar();
            }

            {
                float iconSize = 32.0f;

                ImGui::BeginChild("Toolbar", ImVec2(0, iconSize), false,
                    ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);

                // Optionally change spacing between buttons.
                ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(4, 4));

                // Icon Button 1
                if (ImGui::Button("Record"))
                {
                    // Handle icon 1 click event
                }

                //// Next button on the same horizontal line.
                ImGui::SameLine();

                ImGui::PushStyleColor(ImGuiCol_PopupBg, ImVec4(0.0f, 0.0f, 0.0f, 0.0f)); // 50% transparent background
                if (ImGui::BeginPopup("TransparentPopup", ImGuiWindowFlags_AlwaysAutoResize))
                {
                    // Option 1: Push a style color override for the window background.
                    // The fourth parameter is the alpha (transparency) value, where 0.0 is fully transparent.

                    // Optionally, you can disable the window decorations if needed.

                    // Create the window as a popup with the adjusted flags.
                    if (ImGui::BeginChild("PopupContent", ImVec2(300, 100)))
                    {
                        ImGui::Text("Set Recording area...");
                        // Additional UI elements here...
                    }
                    ImGui::EndChild();

                    ImGui::EndPopup();
                }

                // Pop the style override for the window background.
                ImGui::PopStyleColor();

                // Trigger or open the popup (for example, on a button click)
                if (ImGui::Button("Open Transparent Popup"))
                {
                    ImGui::OpenPopup("TransparentPopup");
                }

                //// Icon Button 2
                //if (ImGui::ImageButton(iconTexture2, ImVec2(iconSize, iconSize)))
                //{
                //    // Handle icon 2 click event
                //}

                ImGui::PopStyleVar(); // Restore item spacing

                ImGui::EndChild();
            }

            // Child 1: no border, enable horizontal scrollbar
            {
                ImGuiWindowFlags window_flags = ImGuiWindowFlags_HorizontalScrollbar;
                ImGui::BeginChild("ChildL", ImVec2(ImGui::GetContentRegionAvail().x * 0.5f, ImGui::GetContentRegionAvail().y), ImGuiChildFlags_Borders, window_flags);
                for (int i = 0; i < 100; i++)
                    ImGui::Text("%04d: scrollable region", i);
                ImGui::EndChild();
            }

            ImGui::SameLine();

            // Child 2: rounded border
            {
                ImGuiWindowFlags window_flags = ImGuiWindowFlags_None;
                ImGui::BeginChild("ChildR", ImVec2(0, ImGui::GetContentRegionAvail().y), ImGuiChildFlags_Borders, window_flags);
                if (ImGui::BeginTable("split", 2, ImGuiTableFlags_Resizable | ImGuiTableFlags_NoSavedSettings))
                {
                    for (int i = 0; i < 100; i++)
                    {
                        char buf[32];
                        sprintf(buf, "%03d", i);
                        ImGui::TableNextColumn();
                        ImGui::Button(buf, ImVec2(-FLT_MIN, 0.0f));
                    }
                    ImGui::EndTable();
                }
                ImGui::EndChild();
            }
        }
        ImGui::End();

        // Rendering
        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(clear_color.x * clear_color.w, clear_color.y * clear_color.w, clear_color.z * clear_color.w, clear_color.w);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
    }

    // Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}
