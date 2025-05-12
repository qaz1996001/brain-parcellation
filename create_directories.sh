#!/bin/bash

create_directories_from_env() {
    local env_file="$1"

    # Check if .env file exists
    if [ ! -f "$env_file" ]; then
        echo "Error: .env file not found at $env_file"
        exit 1
    fi

    echo "Parsed environment variables:"
    echo "-----------------------------"

    # Read .env file line by line
    while IFS= read -r line
    do
        # Skip empty lines and comments
        [[ -z "$line" || "$line" == \#* ]] && continue

        # Extract key and value while preserving intended spaces
        if [[ "$line" =~ ^([^=]+)=(.*)$ ]]; then
            key="${BASH_REMATCH[1]}"
            value="${BASH_REMATCH[2]}"

            # Remove leading/trailing whitespace from key
            key=$(echo "$key" | xargs)

            # Skip if key is empty
            [[ -z "$key" ]] && continue

            # Process the value - remove quotes and trailing whitespace
            # First, trim trailing whitespace from the value
            value=$(echo "$value" | sed 's/[[:space:]]*$//')

            # Then remove any quotes
            value="${value#\"}"
            value="${value%\"}"
            value="${value#\'}"
            value="${value%\'}"

            # Final trim of any remaining trailing whitespace (that might have been inside quotes)
            value=$(echo "$value" | sed 's/[[:space:]]*$//')

            # Display all valid environment variables
            if [[ -n "$key" && -n "$value" ]]; then
                # Expand environment variables
                expanded_value=$(eval printf '%s' "$value")

                # Display the key and expanded value
                echo "$key = '$expanded_value'" # Quoting to make any trailing spaces visible

                # Export the variable to the environment
                export "$key=$value"
            fi

            # Check if the key contains 'PATH' and the value is not empty
            if [[ "$key" == *PATH* && -n "$value" ]]; then
                # Expand environment variables safely
                expanded_path=$(eval printf '%s' "$value")

                # Debug: Show exact path with visible spaces
                echo "DEBUG - Path with markers: |$expanded_path|"

                # Check if the expanded path is not empty
                if [ -n "$expanded_path" ]; then
                    # Create directory if it doesn't exist
                    if [ ! -d "$expanded_path" ]; then
                        echo "Creating directory: |$expanded_path|"
                        mkdir -p "$expanded_path"

                        # Set permissions to be readable, writable, and executable by all
                        chmod 777 "$expanded_path"
                    else
                        echo "Directory already exists: |$expanded_path|"
                    fi
                fi
            fi
        fi
    done < "$env_file"

    echo "-----------------------------"
}

# Main function
main() {
    # Get the directory of the script
    SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
    ENV_FILE="${SCRIPT_DIR}/.env"

    # Check if a custom .env file is provided as an argument
    if [ $# -eq 1 ]; then
        ENV_FILE="$1"
    fi

    echo "Using .env file: $ENV_FILE"

    # Call the function to create directories
    create_directories_from_env "$ENV_FILE"

    echo "Directory creation completed."
}

# Run the main function with all script arguments
main "$@"