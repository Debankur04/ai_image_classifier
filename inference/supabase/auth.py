from supabase.auth import sign_up, sign_in_with_password, sign_out,reset_password_for_email, update_user


def signup(email,password):
    response = sign_up({
        'email': email,
        'password': password
    })
    return response

def signin(email,password):
    response = sign_in_with_password(
    {
        "email": email,
        "password": password,
    }
    )
    return response

def signout():
    return sign_out()

