"""Release external GPU allocations before the serial worker changes tasks."""
import atexit
from weakref import WeakSet

_resources = WeakSet()


def register(resource):
    _resources.add(resource)


def release_except(group=None):
    for resource in list(_resources):
        if group is None or resource.resource_group != group:
            resource.close()


atexit.register(release_except)
