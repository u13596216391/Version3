"""
ASGI config for integrated_mine project.
"""
import os
from django.core.asgi import get_asgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'integrated_mine.settings')
application = get_asgi_application()
