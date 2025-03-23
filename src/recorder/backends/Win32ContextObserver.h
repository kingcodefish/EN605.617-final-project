#pragma once

#ifndef WIN32
static_assert(false, "Building WIN32 backend in non-WIN32 environment!");
#else

#include "ContextObserver.h"

#include <windows.h>

#include <thread>
#include <vector>

namespace recorder
{
    class Win32ContextObserver : public ContextObserver
    {
    public:
        Win32ContextObserver(void* handle = nullptr);
        virtual ~Win32ContextObserver();

        virtual void subscribe(EventType eventType,
            std::function<void(ContextEvent)> callback) override;
    private:
        HWND* m_handle;
        std::thread m_thread;
        std::vector<HHOOK*> m_hooks;
    };
}

#endif
