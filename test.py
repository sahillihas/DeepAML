from django.contrib import admin
from django.urls import path, include
from django.shortcuts import redirect
from django.contrib.auth import views as auth_views
from cable import views as cable_views

# Redirect root URL based on authentication status
def home_redirect(request):
    return redirect('dashboard' if request.user.is_authenticated else 'login')

urlpatterns = [
    # Root URL: redirect to dashboard or login
    path('', home_redirect, name='home'),

    # Admin site
    path('admin/', admin.site.urls),

    # Authentication
    path('login/', auth_views.LoginView.as_view(
        template_name='registration/login.html',
        redirect_authenticated_user=True
    ), name='login'),
    path('logout/', auth_views.LogoutView.as_view(next_page='login'), name='logout'),
    path('signup/', cable_views.signup, name='signup'),

    # App routes
    path('', include('cable.urls')),
]
