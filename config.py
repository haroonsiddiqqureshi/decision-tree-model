import os

# Define the host address for the application, usually set to localhost (127.0.0.1)
APP_HOST: str = "0.0.0.0"

# Enable or disable debug mode for the application; True means debug mode is on
APP_DEBUG_MODE: bool = True

# Set the port number on which the application will run; 5000 is a common default for web apps
APP_PORT: int = int(os.environ.get("PORT", 5000))

# Path to the folder where HTML templates are stored for rendering
APP_TEMPLATE_FOLDER: str = "templates"

# Define a constant for the folder that contains static files (like CSS, JS, images)
APP_STATIC_FOLDER: str = "static"

# Define a constant for the base URL path to access static files
APP_STATIC_URL_PATH: str = "/"
