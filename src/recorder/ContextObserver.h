#pragma once

#include <functional>
#include <imgui.h>

namespace recorder
{
    enum class EventType
    {
        MOUSE = 0,
        KEYBOARD
    };

    enum class MouseButton
    {
        LBUTTON = 0,
        RBUTTON
    };

    struct ContextEvent
    {
        virtual ~ContextEvent() {}

        EventType type;
        void* handle;
    };

    struct MouseEvent : public ContextEvent
    {
        ImVec2 mousePos;
        MouseButton mouseBtn;
    };

    struct KeyboardEvent : public ContextEvent
    {
        int keyCode;
    };

    class ContextObserver
    {
    public:
        ContextObserver(void* handle = nullptr) {}
        virtual ~ContextObserver() {}

        /*
        * \brief Subscribe to a given EventType with a callback function.
        */
        virtual void subscribe(EventType eventType,
            std::function<bool(ContextEvent*)> callback) = 0;
    };
}
