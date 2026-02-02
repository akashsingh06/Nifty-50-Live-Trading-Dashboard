"""
Angel One Configuration
=======================

Your Angel One SmartAPI credentials.
IMPORTANT: Keep this file secure and never commit to version control!
"""

# Angel One API Credentials
ANGEL_ONE_CONFIG = {
    'api_key': 'N9iMIiXP',
    'secret_key': '2009917a-0162-4dba-83be-e294be03c3ba',
    
    # You need to provide these for authentication:
    'client_id': '',      # Your Angel One User ID (e.g., 'A12345678')
    'password': '',       # Your Angel One Password
    'totp_secret': '',    # Your TOTP secret from authenticator setup
                          # OR a 6-digit TOTP code (valid for 30 seconds)
}

"""
SETUP INSTRUCTIONS:
==================

1. Install required packages:
   pip install smartapi-python pyotp

2. Fill in your credentials above:
   - client_id: Your Angel One login ID
   - password: Your Angel One password
   - totp_secret: TOTP secret from authenticator app setup
     (or generate a 6-digit code from your authenticator app)

3. To enable TOTP:
   - Login to Angel One web portal
   - Go to Profile > Security Settings
   - Enable 2FA and scan QR code with Google Authenticator
   - Save the secret key shown during setup

4. Usage:
   
   from quant_trading.execution import AngelOneAPI
   from quant_trading.angel_one_config import ANGEL_ONE_CONFIG
   
   broker = AngelOneAPI(
       api_key=ANGEL_ONE_CONFIG['api_key'],
       secret_key=ANGEL_ONE_CONFIG['secret_key'],
       client_id=ANGEL_ONE_CONFIG['client_id'],
       password=ANGEL_ONE_CONFIG['password'],
       totp=ANGEL_ONE_CONFIG['totp_secret']
   )
   
   if broker.connect():
       print("Connected to Angel One!")
       
       # Get account info
       print(broker.get_account_info())
       
       # Get positions
       print(broker.get_positions())
"""
