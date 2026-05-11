import os, sys, logging
from waitress import serve

sys.path.insert(0, r'C:\inetpub\wwwroot\VoiceOfSales\src')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')

logging.basicConfig(
    filename=r'C:\inetpub\wwwroot\VoiceOfSales\src\logs\waitress.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)

from core.wsgi import application

if __name__ == '__main__':
    print("Starting VoiceOfSales on port 8002...")
    serve(
        application,
        host='127.0.0.1',
        port=8002,
        threads=8,
        channel_timeout=120,
        cleanup_interval=30,
        connection_limit=1000,
        max_request_body_size=1073741824
    )