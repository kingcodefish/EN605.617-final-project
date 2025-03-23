#include "Win32ContextObserver.h"

#include <iostream>

namespace recorder
{
    struct InternalCallback
    {
        EventType eventType;
        HWND handle;
        std::function<void(ContextEvent)> callback;
    };

    static std::vector<InternalCallback> callbacks;

    LRESULT CALLBACK dispatchCallback(int nCode, WPARAM wParam, LPARAM lParam)
    {
        if (nCode == HC_ACTION)
        {
            if (wParam == WM_LBUTTONDOWN)
            {
                // Get the mouse position
                MSLLHOOKSTRUCT* pMouseStruct = (MSLLHOOKSTRUCT*)lParam;
                POINT pt = pMouseStruct->pt;

                // Get the HWND of the window under the mouse cursor
                HWND hwnd = WindowFromPoint(pt);
                if (hwnd)
                {
                    // The hwnd might not represent the full application, see if there
                    // is a common ancestor first
                    if (GetParent(hwnd))
                        hwnd = GetParent(hwnd);

                    for (auto& callback : callbacks)
                    {
                        if (callback.eventType == EventType::MOUSE &&
                            (callback.handle == nullptr || callback.handle == hwnd))
                        {
                            callback.callback({ EventType::MOUSE });
                        }
                    }
                    std::cout << "Clicked window HWND: " << hwnd << std::endl;
                }
            }
        }

        // TODO: Fix nullptr here --- needs to know current HHOOK
        return CallNextHookEx(nullptr, nCode, wParam, lParam);
    }

    void messageLoop()
    {
        MSG msg;
        while (GetMessage(&msg, NULL, 0, 0))
        {
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }
    }

    Win32ContextObserver::Win32ContextObserver(void* handle)
    : ContextObserver(handle), m_handle(nullptr)
    {
        if (handle)
        {
            m_handle = static_cast<HWND*>(handle);
        }
    }

    Win32ContextObserver::~Win32ContextObserver()
    {
        m_thread.join();

        for (auto&& hook : m_hooks)
        {
            UnhookWindowsHookEx(*hook);
        }
    }

    void Win32ContextObserver::subscribe(EventType eventType,
        std::function<void(ContextEvent)> callback)
    {
        HHOOK dispatchHook = SetWindowsHookEx(WH_MOUSE_LL, dispatchCallback, NULL, 0);
        if (!dispatchHook)
        {
            std::cerr << "Failed to set hook on event type! " << std::endl;
            return;
        }

        m_hooks.push_back(&dispatchHook);
        
        if (!m_thread.joinable())
        {
            m_thread = std::thread(messageLoop);
        }
    }
}
