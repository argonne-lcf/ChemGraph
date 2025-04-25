import globus_sdk
from globus_sdk.login_flows import LocalServerLoginFlowManager # Needed to access globus_sdk.gare
import os.path

# Globus UserApp name
APP_NAME = "inference_app"

# Public inference auth client
AUTH_CLIENT_ID = "58fdd3bc-e1c3-4ce5-80ea-8d6b87cfb944"

# Inference gateway API scope
GATEWAY_CLIENT_ID = "681c10cc-f684-4540-bcd7-0b4df3bc26ef"
GATEWAY_SCOPE = f"https://auth.globus.org/scopes/{GATEWAY_CLIENT_ID}/action_all"

# Path where access and refresh tokens are stored
TOKENS_PATH = f"{os.path.expanduser('~')}/.globus/app/{AUTH_CLIENT_ID}/{APP_NAME}/tokens.json"

# Allowed identity provider domains
ALLOWED_DOMAINS = ["anl.gov", "alcf.anl.gov"]

# Globus authorizer parameters to point to specific identity providers
GA_PARAMS = globus_sdk.gare.GlobusAuthorizationParameters(session_required_single_domain=ALLOWED_DOMAINS)


# Error handler to guide user through specific identity providers 
class DomainBasedErrorHandler:
    def __call__(self, app, error):
        print(f"Encountered error '{error}', initiating login...")
        app.login(auth_params=GA_PARAMS)


# Get refresh authorizer object
def get_auth_object(force=False):
    """
    Create a Globus UserApp with the inference service scope
    and trigger the authentication process. If authentication
    has already happened, existing tokens will be reused.
    """

    # Create Globus user application
    app = globus_sdk.UserApp(
        APP_NAME,
        client_id=AUTH_CLIENT_ID,
        scope_requirements={GATEWAY_CLIENT_ID: [GATEWAY_SCOPE]},
        config=globus_sdk.GlobusAppConfig(
            request_refresh_tokens=True,
            token_validation_error_handler=DomainBasedErrorHandler()
        ),
    )

    # Force re-login if required
    if force:
        app.login(auth_params=GA_PARAMS)

    # Authenticate using your Globus account or reuse existing tokens
    auth = app.get_authorizer(GATEWAY_CLIENT_ID)

    # Return the Globus refresh token authorizer
    return auth


# Get access token
def get_access_token():
    """
    Load existing tokens, refresh the access token if necessary,
    and return the valid access token. If there is no token stored
    in the home directory, or if the refresh token is expired following
    6 months of inactivity, an authentication will be triggered.
    """

    # Get authorizer object and authenticate if need be
    auth = get_auth_object()

    # Make sure the stored access token if valid, and refresh otherwise
    auth.ensure_valid_token()

    # Return the access token
    return auth.access_token


# If this file is executed as a script ...
if __name__ == "__main__":

    # Imports
    import argparse

    # Exception to raise in case of errors
    class InferenceAuthError(Exception):
        pass

    # Constant
    AUTHENTICATE_ACTION = "authenticate"
    GET_ACCESS_TOKEN_ACTION = "get_access_token"

    # Define possible arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=[AUTHENTICATE_ACTION, GET_ACCESS_TOKEN_ACTION])
    parser.add_argument("-f", "--force", action="store_true", help="authenticate from scratch")
    args = parser.parse_args()

    # Authentication
    if args.action == AUTHENTICATE_ACTION:
        
        # Authenticate using Globus account
        _ = get_auth_object(force=args.force)

    # Get token
    elif args.action == GET_ACCESS_TOKEN_ACTION:

        # Make sure tokens exist
        # This is important otherwise the CLI will print more than just the access token
        if not os.path.isfile(TOKENS_PATH):
            raise InferenceAuthError('Access token does not exist. '
                'Please authenticate by running "python3 inference_auth_token.py authenticate".')
        
        # Make sure no force flag was provided
        if args.force:
            raise InferenceAuthError(f"The --force flag cannot be used with the {GET_ACCESS_TOKEN_ACTION} action.")

        # Load tokens, refresh token if necessary, and print access token
        print(get_access_token())