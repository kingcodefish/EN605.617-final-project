#include "Win32ContextObserver.h"

#include <iostream>

namespace recorder
{
    struct InternalCallback
    {
        EventType eventType;
        HWND handle;
        std::function<bool(ContextEvent*)> callback;
    };

    static std::vector<InternalCallback> callbacks;
    static HHOOK dispatchHook = nullptr;

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

                    for (auto callbackItr = callbacks.begin();
                        callbackItr != callbacks.end();)
                    {
                        if (callbackItr->eventType == EventType::MOUSE &&
                            (callbackItr->handle == nullptr || callbackItr->handle == hwnd))
                        {
                            RECT rect;
                            GetWindowRect(hwnd, &rect);

                            MouseEvent ev;
                            ev.type = EventType::MOUSE;
                            ev.handle = (void*)hwnd;
                            ev.mouseBtn = MouseButton::LBUTTON;
                            ev.mousePos = ImVec2(pt.x - rect.left, pt.y - rect.top);

                            // If the callback "handles" this event, then we
                            // should erase the iterator.
                            if (callbackItr->callback(&ev))
                            {
                                callbackItr = callbacks.erase(callbackItr);
                            }
                            else
                            {
                                callbackItr++;
                            }
                        }
                        else
                        {
                            callbackItr++;
                        }
                    }
                }
            }
            else if (wParam == WM_KEYDOWN || wParam == WM_SYSKEYDOWN)
            {
                KBDLLHOOKSTRUCT* pKeyStruct = reinterpret_cast<KBDLLHOOKSTRUCT*>(lParam);

                // Translate virtual key to character
                BYTE keyboardState[256];
                GetKeyboardState(keyboardState);

                char buffer[2];
                if (ToAscii(pKeyStruct->vkCode, pKeyStruct->scanCode, keyboardState, (LPWORD)buffer, 0) == 1) {
                    for (auto callbackItr = callbacks.begin();
                        callbackItr != callbacks.end();)
                    {
                        if (callbackItr->eventType == EventType::KEYBOARD &&
                            GetForegroundWindow() == callbackItr->handle)
                        {
                            KeyboardEvent ev;
                            ev.type = EventType::KEYBOARD;
                            ev.keyCode = static_cast<int>(buffer[0]);
                            ev.handle = callbackItr->handle;

                            // If the callback "handles" this event, then we
                            // should erase the iterator.
                            if (callbackItr->callback(&ev))
                            {
                                callbackItr = callbacks.erase(callbackItr);
                            }
                            else
                            {
                                callbackItr++;
                            }
                        }
                        else
                        {
                            callbackItr++;
                        }
                    }
                }
            }
        }

        return CallNextHookEx(dispatchHook, nCode, wParam, lParam);
    }

    Win32ContextObserver::Win32ContextObserver(void* handle)
    : ContextObserver(handle), m_handle(nullptr)
    {
        if (handle)
        {
            m_handle = static_cast<HWND>(handle);
        }
    }

    Win32ContextObserver::~Win32ContextObserver()
    {
        UnhookWindowsHookEx(dispatchHook);
    }

    void Win32ContextObserver::subscribe(EventType eventType,
        std::function<bool(ContextEvent*)> callback)
    {
        if (!dispatchHook)
        {
            dispatchHook = SetWindowsHookEx(WH_MOUSE_LL, dispatchCallback, NULL, 0);
            dispatchHook = SetWindowsHookEx(WH_KEYBOARD_LL, dispatchCallback, NULL, 0);
        }

        if (!dispatchHook)
        {
            std::cerr << "Failed to set hook on event type! " << std::endl;
            return;
        }
        callbacks.push_back({ eventType, m_handle, callback });
    }
}
