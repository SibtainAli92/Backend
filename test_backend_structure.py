#!/usr/bin/env python
"""
Test script to validate backend structure and imports.
"""
import sys
sys.path.insert(0, '.')

from app import app
from connection import health_check

print('=== Backend Structure Validation ===')
print(f'App Title: {app.title}')
print(f'App Version: {app.version}')

routes = [route for route in app.routes if hasattr(route, 'path')]
print(f'API Routes: {len(routes)}')
for route in routes:
    if hasattr(route, 'methods'):
        print(f'  {list(route.methods)[0] if route.methods else ""} {route.path}')

print()
print('OK: Backend imports successfully')
