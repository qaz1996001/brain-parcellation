import os
import requests
from dotenv import load_dotenv
import json

# Load environment variables from .env file
load_dotenv()

# RabbitMQ connection parameters from .env
rabbitmq_admin_user = os.getenv('RABBITMQ_DEFAULT_USER', 'admin')
rabbitmq_admin_pass = os.getenv('RABBITMQ_DEFAULT_PASS', 'pass.123')
rabbitmq_host = os.getenv('RABBITMQ_HOST', '127.0.0.1')
rabbitmq_ui_port = int(os.getenv('RABBITMQ_UI_PORT', 15672))
rabbitmq_port = int(os.getenv('RABBITMQ_PORT', 5672))

# Target user and vhost to check/create
target_user = os.getenv('RABBITMQ_USER', 'guest')
target_pass = os.getenv('RABBITMQ_PASS', 'guest')
target_vhost = os.getenv('RABBITMQ_VIRTUAL_HOST', 'celery')

# Management API base URL
api_base = f"http://{rabbitmq_host}:{rabbitmq_ui_port}/api"

# Authentication for API requests
auth = (rabbitmq_admin_user, rabbitmq_admin_pass)


def check_and_create_vhost(vhost_name):
    """Check if the virtual host exists, create it if it doesn't"""
    # Check if vhost exists
    response = requests.get(f"{api_base}/vhosts/{vhost_name}", auth=auth)

    if response.status_code == 404:
        print(f"Virtual host '{vhost_name}' does not exist. Creating...")
        # Create the vhost
        response = requests.put(f"{api_base}/vhosts/{vhost_name}", auth=auth)
        if response.status_code == 201:
            print(f"Virtual host '{vhost_name}' created successfully.")
        else:
            print(f"Failed to create virtual host '{vhost_name}'. Status code: {response.status_code}")
            print(response.text)
    elif response.status_code == 200:
        print(f"Virtual host '{vhost_name}' already exists.")
    else:
        print(f"Error checking virtual host '{vhost_name}'. Status code: {response.status_code}")
        print(response.text)


def check_and_create_user(username, password):
    """Check if the user exists, create it if it doesn't"""
    # Check if user exists
    response = requests.get(f"{api_base}/users/{username}", auth=auth)

    if response.status_code == 404:
        print(f"User '{username}' does not exist. Creating...")
        # Create the user
        user_data = {
            "password": password,
            "tags": "administrator"
        }
        response = requests.put(
            f"{api_base}/users/{username}",
            auth=auth,
            headers={"Content-Type": "application/json"},
            data=json.dumps(user_data)
        )
        if response.status_code == 201:
            print(f"User '{username}' created successfully.")
        else:
            print(f"Failed to create user '{username}'. Status code: {response.status_code}")
            print(response.text)
    elif response.status_code == 200:
        print(f"User '{username}' already exists.")
    else:
        print(f"Error checking user '{username}'. Status code: {response.status_code}")
        print(response.text)


def set_permissions(username, vhost_name):
    """Set permissions for the user on the virtual host"""
    permissions = {
        "configure": ".*",
        "write": ".*",
        "read": ".*"
    }

    response = requests.put(
        f"{api_base}/permissions/{vhost_name}/{username}",
        auth=auth,
        headers={"Content-Type": "application/json"},
        data=json.dumps(permissions)
    )

    if response.status_code in (201, 204):
        print(f"Permissions for user '{username}' on vhost '{vhost_name}' set successfully.")
    else:
        print(
            f"Failed to set permissions for user '{username}' on vhost '{vhost_name}'. Status code: {response.status_code}")
        print(response.text)


def main():
    try:
        print("Checking RabbitMQ configuration...")

        # Check and create virtual host if needed
        check_and_create_vhost(target_vhost)

        # Check and create user if needed
        check_and_create_user(target_user, target_pass)

        # Set permissions for the user on the virtual host
        set_permissions(target_user, target_vhost)

        print("RabbitMQ configuration check completed.")

        # Verify connection with the created user and vhost without using pika
        print(f"Configuration complete. To verify the connection manually, you can use:")
        print(f"- RabbitMQ Management UI: http://{rabbitmq_host}:{rabbitmq_ui_port}/")
        print(f"- Login with: {target_user}/{target_pass}")
        print(f"- Virtual host: {target_vhost}")

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()