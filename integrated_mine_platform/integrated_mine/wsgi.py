"""
WSGI config for integrated_mine project.
"""
import os
from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'integrated_mine.settings')
application = get_wsgi_application()
