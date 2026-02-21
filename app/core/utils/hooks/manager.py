import logging

logger = logging.getLogger(__name__)


class HookManager:
    """
    A simple yet more robust hook management system in plain Python.
    """

    def __init__(self):
        self._hooks = {}  # Event name -> list of (order, hook_id, function)
        self._next_id = 0  # For unique identification of hooks

    def _generate_id(self):
        self._next_id += 1
        return self._next_id

    def register(self, event_name, func=None, order=0):
        """
        Registers a function (hook) to be called for a specific event.
        Can be used as a decorator if func is None.

        Args:
            event_name (str): The name of the event to hook into.
            func (callable, optional): The function to register.
            order (int): Execution order; lower numbers execute earlier.

        Returns:
            If used as a decorator, returns the decorator.
            Otherwise, returns the hook_id of the registered function.
        """
        if (
            func is None
        ):  # Being used as a decorator @hook_manager.register("event_name")

            def decorator(f):
                self._register_hook(event_name, f, order)
                return f  # Return the original function

            return decorator
        else:  # Being used as a direct call hook_manager.register(...)
            return self._register_hook(event_name, func, order)

    def _register_hook(self, event_name, func, order):
        """Internal registration logic."""
        hook_id = self._generate_id()
        if event_name not in self._hooks:
            self._hooks[event_name] = []

        self._hooks[event_name].append((order, hook_id, func))
        self._hooks[event_name].sort(key=lambda x: x[0])  # Sort by order
        logger.info(
            f"HookManager: Registered hook '{func.__name__}' "
            f"(ID: {hook_id}) for event '{event_name}' with order {order}."
        )
        return hook_id  # Return the ID for potential unregistration

    def unregister(
        self, event_name, func_to_unregister=None, hook_id_to_unregister=None
    ):
        """
        Unregisters a specific hook for an event.
        Provide either the function object or its registration ID.
        """
        if event_name not in self._hooks:
            logger.info(
                f"HookManager: Cannot unregister. "
                f"No hooks for event '{event_name}'."
            )
            return False

        initial_count = len(self._hooks[event_name])
        if hook_id_to_unregister is not None:
            self._hooks[event_name] = [
                h for h in self._hooks[event_name] if h[1] != hook_id_to_unregister
            ]
        elif func_to_unregister is not None:
            self._hooks[event_name] = [
                h for h in self._hooks[event_name] if h[2] is not func_to_unregister
            ]
        else:
            logger.info(
                "HookManager: Cannot unregister. "
                "Must provide function object or hook ID."
            )
            return False

        removed_count = initial_count - len(self._hooks[event_name])
        if removed_count > 0:
            logger.info(
                f"HookManager: Unregistered {removed_count} "
                f"hook(s) for event '{event_name}'."
            )
            if not self._hooks[event_name]:
                del self._hooks[event_name]
            return True
        else:
            logger.info(
                f"HookManager: No matching hook found to unregister "
                f"for event '{event_name}'."
            )
            return False

    def trigger(self, event_name, sender=None, *args, **kwargs):
        """
        Triggers an event, calling all registered hooks for that event.

        Args:
            event_name (str): The name of the event to trigger.
            sender (object, optional): The object or context triggering
                                      the event. Passed as the first
                                      argument to the hook if provided.
            *args: Positional arguments to pass to the hook functions.
            **kwargs: Keyword arguments to pass to the hook functions.

        Returns:
            list: A list of results from executed hooks
                  (None if a hook doesn't return).
        """
        results = []
        if event_name in self._hooks:
            # Iterate over a copy in case a hook tries to modify the list
            # during iteration
            for order, hook_id, func in list(self._hooks[event_name]):
                try:
                    if sender is not None:
                        # Pass sender as the first positional argument
                        result = func(sender, *args, **kwargs)
                    else:
                        # Call without the sender argument
                        result = func(*args, **kwargs)
                    results.append(result)
                except Exception as e:
                    logger.info(
                        f"HookManager ERROR: Executing hook "
                        f"'{func.__name__}' (ID: {hook_id}, Order: {order}) "
                        f"for event '{event_name}': {e}"
                    )
            # Optional: logger.info(f"--- Event {event_name} finished ---")
        else:
            # Optional: logger.info(f"HookManager: No hooks registered for
            # event: {event_name}")
            pass
        return results


# Create a global or application-specific hook manager instance
# This instance will be used by the decorator.
hook_manager = HookManager()
