from django.contrib import admin
from django.urls import path, include
from django.shortcuts import redirect
from django.contrib.auth import views as auth_views
from cable import views as cable_views


def redirect_root(request):
    """
    Redirect root URL based on authentication status.
    Authenticated users go to dashboard; others go to login.
    """
    return redirect('dashboard' if request.user.is_authenticated else 'login')


urlpatterns = [
    # Root redirection
    path('', redirect_root, name='home'),

    # Admin panel
    path('admin/', admin.site.urls),

    # Authentication routes
    path('login/', auth_views.LoginView.as_view(
        template_name='registration/login.html',
        redirect_authenticated_user=True
    ), name='login'),

    path('logout/', auth_views.LogoutView.as_view(next_page='login'), name='logout'),
    path('signup/', cable_views.signup, name='signup'),

    # App-specific routes
    path('', include('cable.urls')),
]
