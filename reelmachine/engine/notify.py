"""Windows notifications. Only for things you need to know: a gate failed for good, the budget
stopped a job, Higgsfield or the network is down, or a reel is finished."""
import logging

try:
    from windows_toasts import Toast, WindowsToaster
    _toaster = WindowsToaster("Coloring Reel Machine")
except Exception:            # not on Windows, or the library is missing: log instead of crashing
    _toaster = None


def notify(title, body, on_click=None):
    logging.info("notify: %s — %s", title, body)
    if _toaster is None:
        return
    try:
        t = Toast()
        t.text_fields = [title, body]
        if on_click:
            t.on_activated = lambda _args: on_click()
        _toaster.show_toast(t)
    except Exception as e:
        logging.warning("notification failed: %s", e)
