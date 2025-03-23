#pragma once

#include <functional>

namespace recorder
{
    enum class EventType
    {
        MOUSE = 0,
        KEYBOARD
    };

    struct ContextEvent
    {
        EventType type;
    };

    struct MouseEvent : public ContextEvent
    {

    };

    struct KeyboardEvent : public ContextEvent
    {
        
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
            std::function<void(ContextEvent)> callback) = 0;
    };
}
