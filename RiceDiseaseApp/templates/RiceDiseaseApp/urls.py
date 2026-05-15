from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),   # root URL → index.html
    path('index.html', views.index, name='index'),
    path('Login.html', views.Login, name='Login'),
    path('UserLogin', views.UserLogin, name='UserLogin'),
    path('Register.html', views.Register, name='Register'),
    path('Signup', views.Signup, name='Signup'),
    path('Upload.html', views.Upload, name='Upload'),
    path('UploadImage', views.UploadImage, name='UploadImage'),
    path('Train', views.Train, name='Train'),
]
