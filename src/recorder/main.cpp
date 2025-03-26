#include <imgui/imgui.h>
#include <imgui/backends/imgui_impl_glfw.h>
#include <imgui/backends/imgui_impl_opengl3.h>
#include <stdio.h>
#define GL_SILENCE_DEPRECATION
#include <GLFW/glfw3.h>

#include <iostream>

#ifdef WIN32
#include <windows.h>
#include <backends/Win32ContextObserver.h>
#endif

static void glfw_error_callback(int error, const char* description)
{
    fprintf(stderr, "GLFW Error %d: %s\n", error, description);
}

static HHOOK mouseHook;
static std::unique_ptr<recorder::ContextObserver> observer;
static std::unique_ptr<recorder::ContextObserver> recordObserver;
static std::atomic<HWND> highlightedHwnd = NULL; // Store the currently highlighted window

// Get the horizontal and vertical screen sizes in pixel
inline POINT get_window_resolution(const HWND window_handle)
{
    RECT rectangle;
    GetClientRect(window_handle, &rectangle);
    const POINT coordinates{ rectangle.right, rectangle.bottom };
    return coordinates;
}

#include <iostream>
#include <ole2.h>
#include <olectl.h>

inline bool save_bitmap(const LPCSTR file_path,
    const HBITMAP bitmap, const HPALETTE palette)
{
    PICTDESC pict_description;

    pict_description.cbSizeofstruct = sizeof(PICTDESC);
    pict_description.picType = PICTYPE_BITMAP;
    pict_description.bmp.hbitmap = bitmap;
    pict_description.bmp.hpal = palette;

    LPPICTURE picture;
    auto initial_result = OleCreatePictureIndirect(&pict_description, IID_IPicture, false,
        reinterpret_cast<void**>(&picture));

    if (!SUCCEEDED(initial_result))
    {
        return false;
    }

    LPSTREAM stream;
    initial_result = CreateStreamOnHGlobal(nullptr, true, &stream);

    if (!SUCCEEDED(initial_result))
    {
        picture->Release();
        return false;
    }

    LONG bytes_streamed;
    initial_result = picture->SaveAsFile(stream, true, &bytes_streamed);

    const auto file = CreateFile(file_path, GENERIC_WRITE, FILE_SHARE_READ, nullptr,
        CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);

    if (!SUCCEEDED(initial_result) || !file)
    {
        stream->Release();
        picture->Release();
        return false;
    }

    HGLOBAL mem = nullptr;
    GetHGlobalFromStream(stream, &mem);
    const auto data = GlobalLock(mem);

    DWORD bytes_written;
    auto result = WriteFile(file, data, bytes_streamed, &bytes_written, nullptr);
    result &= bytes_written == static_cast<DWORD>(bytes_streamed);

    GlobalUnlock(mem);
    CloseHandle(file);

    stream->Release();
    picture->Release();

    return result;
}

inline POINT get_client_window_position(const HWND window_handle)
{
    RECT rectangle;

    GetClientRect(window_handle, static_cast<LPRECT>(&rectangle));
    MapWindowPoints(window_handle, nullptr, reinterpret_cast<LPPOINT>(&rectangle), 2);

    const POINT coordinates = { rectangle.left, rectangle.top };

    return coordinates;
}

// https://stackoverflow.com/a/9525788/3764804
inline bool capture_screen_client_window(const HWND window_handle, const ImVec2 mouse_pos, const LPCSTR file_path)
{
    SetActiveWindow(window_handle);

    const auto hdc_source = GetDC(nullptr);
    const auto hdc_memory = CreateCompatibleDC(hdc_source);

    const auto window_resolution = get_window_resolution(window_handle);

    const auto width = window_resolution.x;
    const auto height = window_resolution.y;

    const auto client_window_position = get_client_window_position(window_handle);

    auto h_bitmap = CreateCompatibleBitmap(hdc_source, width, height);
    const auto h_bitmap_old = static_cast<HBITMAP>(SelectObject(hdc_memory, h_bitmap));

    BitBlt(hdc_memory, 0, 0, width, height, hdc_source, client_window_position.x, client_window_position.y, SRCCOPY);

    POINT mouseRelative = { mouse_pos.x - client_window_position.x, mouse_pos.y - client_window_position.y };
    HBRUSH hBlueBrush = CreateSolidBrush(RGB(255, 0, 0));
    SelectObject(hdc_memory, hBlueBrush);
    Ellipse(hdc_memory, mouseRelative.x - 10, mouseRelative.y - 10, mouseRelative.x + 10, mouseRelative.y + 10);
    DeleteObject(hBlueBrush);

    h_bitmap = static_cast<HBITMAP>(SelectObject(hdc_memory, h_bitmap_old));

    DeleteDC(hdc_source);
    DeleteDC(hdc_memory);

    const HPALETTE h_palette = nullptr;
    if (save_bitmap(file_path, h_bitmap, h_palette))
    {
        return true;
    }

    return false;
}

// Main code
int main(int, char**)
{
    MSG msg;

    glfwSetErrorCallback(glfw_error_callback);
    if (!glfwInit())
        return 1;

    // Decide GL+GLSL versions
    // GL 3.0 + GLSL 130
    const char* glsl_version = "#version 130";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    glfwWindowHint(GLFW_TRANSPARENT_FRAMEBUFFER, GLFW_TRUE);

    // Create window with graphics context
    GLFWwindow* window = glfwCreateWindow(1280, 720, "Test Recorder", nullptr, nullptr);
    if (window == nullptr)
        return 1;
    glfwMakeContextCurrent(window);
    glfwSwapInterval(0); // Disable vsync --- this messes with event processing
                         // due to the length of time for glfwSwapBuffers

    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls

    // Setup Dear ImGui style
    ImGui::StyleColorsDark();

    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    // Our state
    bool show_recording_window = false;
    bool recording = false;
    std::atomic<bool> selecting = false;
    ImVec4 clear_color = ImGui::GetStyle().Colors[ImGuiCol_WindowBg];

    // Main loop
    while (!glfwWindowShouldClose(window))
    {
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

                ImGui::PushStyleColor(ImGuiCol_Header, (ImVec4)ImColor(150, 0, 0));
                ImGui::PushStyleColor(ImGuiCol_HeaderHovered, (ImVec4)ImColor(100, 0, 0));
                ImGui::PushStyleColor(ImGuiCol_HeaderActive, (ImVec4)ImColor(130, 0, 0));
                if (ImGui::Selectable("Record", &recording, !highlightedHwnd ? ImGuiSelectableFlags_Disabled : 0, ImVec2(50, 20)) && !recording)
                {
                    recording = true;

                    // Pick observer based on platform
#ifdef WIN32
                    if (!recordObserver)
                    {
                        recordObserver = std::make_unique<recorder::Win32ContextObserver>(highlightedHwnd);
                    }
#else
                    // TODO: X11-based observer
#endif

                    auto callback = [&](recorder::ContextEvent* ev) -> bool {
                        if (ev->type == recorder::EventType::MOUSE)
                        {
                            auto asMouseEvent = dynamic_cast<recorder::MouseEvent*>(ev);
                            if (asMouseEvent->mouseBtn == recorder::MouseButton::LBUTTON)
                            {
                                std::cout << "Click occurred at (" << asMouseEvent->mousePos.x << ", " << asMouseEvent->mousePos.y << ")" << std::endl;
                                capture_screen_client_window((HWND)asMouseEvent->handle, asMouseEvent->mousePos, "C:\\My_Work\\test.bmp");
                            }
                        }
                        return false;
                        };

                    recordObserver->subscribe(recorder::EventType::MOUSE, callback);
                }
                ImGui::PopStyleColor(3);

                ImGui::SameLine();

                if (/*ImGui::Button("Select Application") && */!selecting)
                {
                    selecting = true;
                    highlightedHwnd = NULL;

                    // Pick observer based on platform
#ifdef WIN32
                    if (!observer)
                    {
                        observer = std::make_unique<recorder::Win32ContextObserver>(nullptr);
                    }
#else
                    // TODO: X11-based observer
#endif

                    auto callback = [&](recorder::ContextEvent* ev) -> bool {
                        if (ev->type == recorder::EventType::MOUSE)
                        {
                            auto asMouseEvent = dynamic_cast<recorder::MouseEvent*>(ev);
                            if (asMouseEvent->mouseBtn == recorder::MouseButton::LBUTTON)
                            {
                                std::cout << "HWND: " << asMouseEvent->handle << std::endl;
                                highlightedHwnd = static_cast<HWND>(asMouseEvent->handle);
                                selecting = false;
                                return true;
                            }
                        }
                        return false;
                        };

                    observer->subscribe(recorder::EventType::MOUSE, callback);
                }

                ImGui::PopStyleVar(); // Restore item spacing

                ImGui::EndChild();
            }

            // Child 1: no border, enable horizontal scrollbar
            {
                ImGuiWindowFlags window_flags = ImGuiWindowFlags_HorizontalScrollbar;
                ImGui::BeginChild("Test Steps", ImVec2(ImGui::GetContentRegionAvail().x * 0.5f, ImGui::GetContentRegionAvail().y), ImGuiChildFlags_Borders, window_flags);
                ImGui::Text("Test Item #1");
                ImGui::EndChild();
            }

            ImGui::SameLine();

            // Child 2: rounded border
            {
                ImGuiWindowFlags window_flags = ImGuiWindowFlags_None;
                ImGui::BeginChild("Preview", ImVec2(0, ImGui::GetContentRegionAvail().y), ImGuiChildFlags_Borders, window_flags);
                ImGui::Text("Property #1");
                ImGui::EndChild();
            }
        }
        ImGui::End();

        // Rendering
        ImGui::Render();

        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(clear_color.x * clear_color.w, clear_color.y * clear_color.w, clear_color.z * clear_color.w, 0.0);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
        {
            GLFWwindow* backup_current_context = glfwGetCurrentContext();
            ImGui::UpdatePlatformWindows();
            ImGui::RenderPlatformWindowsDefault();
            glfwMakeContextCurrent(backup_current_context);
        }

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
